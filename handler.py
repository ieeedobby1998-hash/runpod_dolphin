"""
RunPod Serverless handler.

Boots llama.cpp's `llama-server` (OpenAI-compatible) with a GGUF model that is
downloaded on first cold start to a network volume, then proxies OpenAI-style
requests (/v1/chat/completions, /v1/completions, /v1/models) to it.

Designed for large *sharded* GGUF quants (e.g. Dolphin-2.9.2-Mixtral-8x22B Q5_K_M,
which is 3 shards in a subfolder). Point MODEL_FILE at the FIRST shard; llama.cpp
loads the rest automatically as long as all shards sit in the same directory.

Request format (job["input"]):
    {
        "endpoint": "/v1/chat/completions",   # optional, this is the default
        "stream":   true,                      # optional
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 256,
        ...any other OpenAI Chat Completions field...
    }
You may also pass the OpenAI body under "openai_input" and the route under
"openai_route" (RunPod's OpenAI-compat convention); both are handled.
"""

import os
import json
import time
import subprocess

import requests
import runpod
from huggingface_hub import snapshot_download, hf_hub_download

# ---------------------------------------------------------------- config
MODEL_REPO           = os.environ["MODEL_REPO"]
MODEL_FILE           = os.environ["MODEL_FILE"]          # first shard, repo-relative
MODEL_ALLOW_PATTERNS = os.environ.get("MODEL_ALLOW_PATTERNS", "").strip()
MODEL_DIR            = os.environ.get("MODEL_DIR", "/runpod-volume/models")

N_GPU_LAYERS = os.environ.get("N_GPU_LAYERS", "-1")
CTX_SIZE     = os.environ.get("CTX_SIZE", "8192")
PARALLEL     = os.environ.get("PARALLEL", "1")
LLAMA_PORT   = os.environ.get("LLAMA_PORT", "8080")
TENSOR_SPLIT = os.environ.get("TENSOR_SPLIT", "").strip()
HF_TOKEN     = os.environ.get("HF_TOKEN") or None

BASE_URL   = f"http://127.0.0.1:{LLAMA_PORT}"
BOOT_LIMIT = int(os.environ.get("BOOT_LIMIT_SECONDS", "1800"))  # big model = slow load

_server_proc = None


# ---------------------------------------------------------------- model download
def ensure_model() -> str:
    """Make sure every shard is on the volume. Returns the local first-shard path."""
    local_path = os.path.join(MODEL_DIR, MODEL_FILE)
    if os.path.exists(local_path):
        gb = os.path.getsize(local_path) / (1024 ** 3)
        print(f"[model] found cached first shard ({gb:.1f} GB): {local_path}", flush=True)
        return local_path

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[model] downloading from {MODEL_REPO} into {MODEL_DIR} ...", flush=True)

    if MODEL_ALLOW_PATTERNS:
        # Sharded quant: grab every file in the quant folder in one pass.
        patterns = [p.strip() for p in MODEL_ALLOW_PATTERNS.split(",") if p.strip()]
        snapshot_download(
            repo_id=MODEL_REPO,
            allow_patterns=patterns,
            local_dir=MODEL_DIR,
            token=HF_TOKEN,
        )
    else:
        # Single-file quant.
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=MODEL_DIR,
            token=HF_TOKEN,
        )

    if not os.path.exists(local_path):
        raise RuntimeError(
            f"Download finished but first shard missing at {local_path}. "
            f"Check MODEL_FILE / MODEL_ALLOW_PATTERNS."
        )
    print(f"[model] ready: {local_path}", flush=True)
    return local_path


# ---------------------------------------------------------------- server lifecycle
def start_llama_server(model_path: str) -> None:
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        return  # already running

    cmd = [
        "llama-server",
        "-m", model_path,
        "--host", "127.0.0.1",
        "--port", str(LLAMA_PORT),
        "-ngl", str(N_GPU_LAYERS),
        "-c", str(CTX_SIZE),
        "--parallel", str(PARALLEL),
    ]
    if TENSOR_SPLIT:
        cmd += ["--tensor-split", TENSOR_SPLIT]

    print(f"[llama] launching: {' '.join(cmd)}", flush=True)
    _server_proc = subprocess.Popen(cmd)

    # Wait for /health to report ok (model load for 8x22B can take many minutes).
    deadline = time.time() + BOOT_LIMIT
    while time.time() < deadline:
        if _server_proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {_server_proc.returncode})")
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print("[llama] server healthy", flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(2)

    _server_proc.terminate()
    raise RuntimeError(f"llama-server did not become healthy within {BOOT_LIMIT}s")


# ---------------------------------------------------------------- request proxy
def _extract(job_input: dict):
    """Return (route, body, stream) from a variety of accepted input shapes."""
    if "openai_input" in job_input:                 # RunPod OpenAI-compat convention
        body = job_input["openai_input"]
        route = job_input.get("openai_route", "/v1/chat/completions")
    else:
        body = {k: v for k, v in job_input.items() if k not in ("endpoint", "stream")}
        route = job_input.get("endpoint", "/v1/chat/completions")
    stream = bool(job_input.get("stream", body.get("stream", False)))
    body["stream"] = stream
    return route, body, stream


def handler(job):
    job_input = job.get("input") or {}
    route, body, stream = _extract(job_input)
    url = f"{BASE_URL}{route}"

    if stream:
        with requests.post(url, json=body, stream=True, timeout=600) as resp:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                # llama-server emits SSE "data: {...}" lines; pass the JSON payload through.
                if line.startswith("data: "):
                    data = line[len("data: "):]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        yield {"raw": data}
                else:
                    yield {"raw": line}
    else:
        resp = requests.post(url, json=body, timeout=600)
        try:
            return resp.json()
        except json.JSONDecodeError:
            return {"error": "non-json response", "status": resp.status_code, "body": resp.text}


# ---------------------------------------------------------------- boot
_model_path = ensure_model()
start_llama_server(_model_path)

runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
