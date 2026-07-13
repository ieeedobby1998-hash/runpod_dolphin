# RunPod Serverless worker: llama.cpp (llama-server) serving a GGUF model.
# Image is model-agnostic — the actual model is chosen at runtime via env vars.
# Weights are NOT baked in; they are pulled to a RunPod network volume on first cold start.

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# We control startup ourselves from handler.py
ENTRYPOINT []

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}"

RUN pip install --no-cache-dir --break-system-packages \
        runpod requests huggingface-hub

COPY handler.py /app/handler.py

# ---- Model config (override any of these in the RunPod endpoint env vars) ----
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
# Dolphin-2.9.2-Mixtral-8x22B, Q5_K_M (sharded, ~99 GB across 3 files in a subfolder)
ENV MODEL_REPO=bartowski/dolphin-2.9.2-mixtral-8x22b-GGUF
# Path to the FIRST shard, relative to the repo root (llama.cpp auto-loads 00002/00003)
ENV MODEL_FILE=dolphin-2.9.2-mixtral-8x22b-Q5_K_M.gguf/dolphin-2.9.2-mixtral-8x22b-Q5_K_M-00001-of-00003.gguf
# Glob of every file to download for this quant (all 3 shards live in this folder)
ENV MODEL_ALLOW_PATTERNS=dolphin-2.9.2-mixtral-8x22b-Q5_K_M.gguf/*
ENV MODEL_DIR=/runpod-volume/models

# ---- llama-server runtime config ----
ENV N_GPU_LAYERS=-1
ENV CTX_SIZE=8192
ENV PARALLEL=1
ENV LLAMA_PORT=8080
# Optional: multi-GPU weight split, e.g. "1,1" for two equal GPUs. Empty = llama.cpp default.
ENV TENSOR_SPLIT=
# Optional HF token for gated/private repos (bartowski's is public, so leave unset)
# ENV HF_TOKEN=

CMD ["python3", "/app/handler.py"]

# trigger CI
