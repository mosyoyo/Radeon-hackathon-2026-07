#!/bin/bash
# Launch vLLM OpenAI server on AMD gfx1100 (ROCm).
# Quirk handling:
#   - HF_HUB_OFFLINE=1 so transformers/Hub don't phone home for speculator configs
#   - VLLM_ATTENTION_BACKEND=ROCM_ATTN bypasses the default flash_attn that
#     expects NVIDIA's flash_attn_2_cuda
#   - python -m vllm.entrypoints.openai.api_server (this vLLM version's CLI does
#     not accept `serve <path>` as a positional; we pass --model explicitly)

cd "$(dirname "$0")/.."

exec env \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_ENDPOINT=https://hf-mirror.com \
  VLLM_ATTENTION_BACKEND=ROCM_ATTN \
  /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /workspace/persistence/hackathon/models/Qwen2.5-14B-Instruct \
    --served-model-name Qwen2.5-14B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
