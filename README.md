# Dolphin-2.9.2-Mixtral-8x22B (Q5_K_M GGUF) — RunPod Serverless (llama.cpp)

An uncensored Mixtral-8x22B served via `llama.cpp` (`llama-server`, OpenAI-compatible)
on RunPod Serverless. The GGUF is **not** baked into the image; it is pulled once to a
**network volume** on the first cold start and reused thereafter.

- **Model repo:** `bartowski/dolphin-2.9.2-mixtral-8x22b-GGUF`
- **Quant:** `Q5_K_M` — 3 shards (~99 GB) in subfolder `dolphin-2.9.2-mixtral-8x22b-Q5_K_M.gguf/`
- **Base image:** `ghcr.io/ggml-org/llama.cpp:server-cuda` (NVIDIA/CUDA only)

> Why this route: RunPod's one-click "Deploy LLM from Hugging Face" (vLLM) worker
> does **not** support GGUF (only AWQ/SqueezeLLM/GPTQ), and no AWQ/GPTQ build of this
> 8x22B exists. llama.cpp is the reliable way to serve this sharded MoE GGUF cheaply.

---

## Files
- `Dockerfile` — builds the ~2-3 GB worker image (no weights inside).
- `handler.py` — downloads shards, boots `llama-server`, proxies OpenAI requests.
- `env.reference.txt` — the env vars to paste into the RunPod endpoint.

---

## Step 1 — Build & push the image  (⚠️ do this on your machine)
You need Docker + a registry (Docker Hub or GHCR). RunPod cannot build for you.

```bash
cd dolphin-8x22b-runpod
# Docker Hub example — replace YOURUSER
docker build -t YOURUSER/dolphin-8x22b-llamacpp:latest .
docker push YOURUSER/dolphin-8x22b-llamacpp:latest
```

(If you'd rather not build locally: fork `github.com/hegner123/gemma4`, drop these two
files in, and let its GitHub Action build/push to GHCR. Ask me and I'll adapt the workflow.)

## Step 2 — Create a Network Volume  (in RunPod console)
- Storage → Network Volumes → New. Size **≥ 150 GB** (99 GB weights + headroom).
- **Region must match** where your endpoint's GPU will run. Note the region.
- First cold start downloads ~99 GB (one time); later starts just mount it.

## Step 3 — Create the Serverless endpoint from the Docker image
- Serverless → New Endpoint → **Deploy from a Docker image**.
- Image: `YOURUSER/dolphin-8x22b-llamacpp:latest`
- **GPU:** pick ONE of:
  - `1 × H200 (141 GB)` or `1 × MI300X`-class **NVIDIA** 141 GB+ → simplest, all on one GPU.
  - `2 × 80 GB` (A100/H100 80GB) → also fine; llama.cpp splits layers across both.
    If you use 2 GPUs and want an explicit split, set `TENSOR_SPLIT=1,1`.
  - `1 × 80 GB` is **too small** for Q5_K_M fully on GPU; avoid unless you accept
    heavy CPU offload (slow) via a lower `N_GPU_LAYERS`.
- **Attach the network volume** from Step 2 (mounts at `/runpod-volume`).
- **Container start command:** leave default (image `CMD` runs `handler.py`).
- Env vars: paste from `env.reference.txt` (defaults already baked into the image,
  so you only need to override if you change GPU/context).
- Workers: Min 0 / Max 1 to start. Set a generous **idle timeout** and raise the
  **execution timeout** — first request waits on the ~99 GB download + model load.

## Step 4 — Test
```bash
curl https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"input":{"endpoint":"/v1/chat/completions",
               "messages":[{"role":"user","content":"Say hello."}],
               "max_tokens":64}}'
```
The very first call is slow (cold download + load). Subsequent calls are fast.

---

## Notes / gotchas
- **Cold start:** downloading 99 GB + loading an 8x22B takes many minutes the first time.
  `handler.py` waits up to `BOOT_LIMIT_SECONDS` (default 1800). Keep at least 1 worker
  warm (Min workers = 1) if you want low latency, at the cost of idle GPU billing.
- **Context size:** `CTX_SIZE=8192` default. Larger context needs more VRAM for KV cache —
  leave headroom beyond the 99 GB of weights.
- **CUDA only:** the base image is CUDA. Do not select an AMD/ROCm GPU unless you swap
  the base image to a ROCm llama.cpp build.
- **Streaming:** pass `"stream": true` in the input; the handler relays SSE chunks.
