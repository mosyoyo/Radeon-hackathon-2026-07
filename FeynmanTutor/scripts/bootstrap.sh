#!/bin/bash
#
# bootstrap.sh — one-shot environment setup for FeynmanTutor on a fresh
# AMD Radeon Cloud instance (or any ROCm box with one gfx1100-class GPU).
#
# Idempotent: safe to run multiple times. Everything after step 1 is
# resumable — if a step fails (e.g. model download interrupted) just run
# the script again.
#
# Usage:
#   bash FeynmanTutor/scripts/bootstrap.sh
#
# What it does:
#   1. Clone the repo (skip if already present)
#   2. Ensure /opt/venv (the prebuilt ROCm venv) exists, else create a venv
#   3. Install agent-side python deps
#   4. Uninstall NVIDIA flash-attn if present (ROCM_ATTN fix)
#   5. Download Qwen2.5-14B-Instruct (hf-mirror accelerated)
#   6. Download BAAI/bge-base-en-v1.5
#   7. Launch vLLM + web UI (optional, via --start)

set -eu

# ---- config ----------------------------------------------------------------
REPO_DIR="${FT_REPO_DIR:-/workspace/persistence/hackathon/Radeon-hackathon-2026-07}"
MODELS_DIR="${FT_MODELS_DIR:-/workspace/persistence/hackathon/models}"
FORK_URL="${FT_FORK_URL:-https://github.com/mosyoyo/Radeon-hackathon-2026-07.git}"
BRANCH="${FT_BRANCH:-feature/feynmantutor}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

MODEL_CHAT="Qwen/Qwen2.5-14B-Instruct"
MODEL_EMB="BAAI/bge-base-en-v1.5"

log()  { printf '\n\033[1;36m[FeynmanTutor]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[FeynmanTutor] FAILED:\033[0m %s\n' "$*"; exit 1; }

# ---- python ----------------------------------------------------------------
if [[ -x /opt/venv/bin/python ]]; then
    PY=/opt/venv/bin/python
    VENV=/opt/venv
else
    VENV="${FT_VENV:-$HOME/ft-venv}"
    if [[ ! -x "$VENV/bin/python" ]]; then
        log "creating venv at $VENV"
        python3 -m venv "$VENV"
    fi
    PY="$VENV/bin/python"
    log "upgrading pip"
    "$PY" -m pip install -q --upgrade pip -i "$PIP_INDEX"
fi
log "using python: $PY"
"$PY" -c "import torch, vllm; print('  torch', torch.__version__, '| vllm', vllm.__version__)" \
    || log "  (torch/vllm not importable yet — will be installed below if needed)"

# ---- clone repo (if not present) -------------------------------------------
if [[ ! -d "$REPO_DIR/.git" ]]; then
    log "cloning fork $FORK_URL"
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone "$FORK_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
if [[ -n "$BRANCH" ]]; then
    git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
fi

# ---- python deps -------------------------------------------------------------
log "installing agent-side deps"
"$PY" -m pip install -q \
    -i "$PIP_INDEX" \
    -r FeynmanTutor/agent/requirements.txt || log "  some agent deps failed — check network"

# ---- flash-attn uninstall (ROCM_ATTN fix) ------------------------------------
if "$PY" -m pip show flash-attn >/dev/null 2>&1; then
    log "uninstalling NVIDIA flash-attn (ROCM_ATTN fix)"
    "$PY" -m pip uninstall -y -q flash-attn
else
    log "flash-attn not present — good"
fi

# ---- model download ------------------------------------------------------------
mkdir -p "$MODELS_DIR"
export HF_ENDPOINT
export HF_HUB_OFFLINE=0

for spec in "$MODEL_CHAT:$MODELS_DIR/Qwen2.5-14B-Instruct" "$MODEL_EMB:$MODELS_DIR/bge-base-en-v1.5"; do
    repo="${spec%%:*}"
    dest="${spec##*:}"
    if [[ -f "$dest/config.json" && -n "$(ls "$dest"/*.safetensors 2>/dev/null)" ]]; then
        log "model already present: $dest (skipping)"
        continue
    fi
    log "downloading $repo → $dest"
    "$PY" -m huggingface_hub.commands.huggingface_cli download "$repo" --local-dir "$dest" \
        || fail "model download failed for $repo (network?) — re-run to resume"
done

# ---- sanity ---------------------------------------------------------------------
log "embedding model files:"
ls "$MODELS_DIR/bge-base-en-v1.5" 2>/dev/null | head -3
log "chat model files:"
ls "$MODELS_DIR/Qwen2.5-14B-Instruct"/*.safetensors 2>/dev/null | wc -l

cat <<EOF

\033[1;32m✓ Bootstrap complete.\033[0m

Next steps:
  1. Start the model server:   bash $REPO_DIR/FeynmanTutor/server/start_vllm.sh
  2. Health check:             curl http://127.0.0.1:8000/v1/models
  3. Web UI:                   bash $REPO_DIR/FeynmanTutor/server/start_web.sh
  4. Tunnel:                   ~/.local/bin/rc-tunnel expose --port 8500
  5. Scripted demo:            $PY $REPO_DIR/FeynmanTutor/scripts/run_demo.py
EOF
