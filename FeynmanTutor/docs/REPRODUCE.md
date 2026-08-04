# Reproduction Guide — FeynmanTutor on AMD Radeon GPU

This document lets an evaluator reproduce FeynmanTutor end-to-end on a Radeon
Cloud instance (or any machine with one **AMD gfx1100-class GPU** + ROCm 6.x).

> **Total time estimate:** ~25–40 minutes (most of it is unattended model
> download and pip install). Hands-on time is ~5 minutes.

---

## 0. Hardware & software prerequisites

| Component    | Required                            | Verified-on                      |
| ------------ | ----------------------------------- | -------------------------------- |
| GPU          | 1× AMD gfx1100 (RDNA3), ≥ 24 GB VRAM | Radeon Cloud `gfx1100`, 48 GB   |
| ROCm         | 6.x (rock module loaded)            | ROCk 6.16.13 / HIP runtime 1.18 |
| CPU          | x86_64, ≥ 8 cores                   | AMD EPYC 9334, 128 threads       |
| RAM          | ≥ 32 GB free                        | 503 GB                            |
| Disk         | ≥ 80 GB free (model weights)        | 2.8 TB available                 |
| OS           | Ubuntu 22.04 / 24.04                | Ubuntu 24.04.4 LTS               |
| Python       | 3.10–3.12                           | 3.12.3                            |

Verify the GPU is reachable from ROCm:

```bash
rocm-smi --showproductname --showmeminfo vram
# Expect: 1 GPU, gfx1100, ~48 GB VRAM
rocminfo | grep -E "Name:|Device Type:|Compute Unit:" | head
# Expect: one GPU agent named gfx1100
```

If `rocm-smi` reports **0 GPUs**, the instance was launched without GPU
passthrough — pick a GPU template in Radeon Cloud → Profile → My Templates,
**enable GPU**, then re-launch.

---

## 1. Clone the repository

```bash
git clone https://github.com/<your-fork>/Radeon-hackathon-2026-07.git
cd Radeon-hackathon-2026-07/FeynmanTutor
```

If `git clone` fails with a certificate error inside a Radeon Cloud container,
the platform uses an MITM proxy. Work around it:

```bash
git config --global http.https://github.com/.sslVerify false
git clone https://github.com/<your-fork>/Radeon-hackathon-2026-07.git
```

---

## 2. Create a Python virtual environment

Use a fresh venv so we don't pollute the host Python:

```bash
python3 -m venv ~/ft-venv
source ~/ft-venv/bin/activate
python -m pip install --upgrade pip
```

---

## 3. Install dependencies

### 3.1 vLLM (ROCm)

We install vLLM via the AMD-maintained ROCm wheel index. ROCm 6.x wheels are
published at `https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/`.

```bash
# inside the venv, with ROCm available on the host
pip install vllm \
    --extra-index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/ \
    --trusted-host repo.radeon.com
```

> **If `repo.radeon.com` is unreachable from your network**, fall back to the
> PyPI build (slower but works):
> ```bash
> pip install vllm
> HSA_OVERRIDE_GFX_VERSION=11.0.0 python -c "import vllm; print(vllm.__version__)"
> ```
> gfx1100 is a RDNA3 consumer GPU. vLLM 0.6.x+ auto-detects it; if anything
> complains "unsupported arch", set `HSA_OVERRIDE_GFX_VERSION=11.0.0`.

### 3.2 Agent-side dependencies

```bash
pip install -r agent/requirements.txt
# openai (the SDK, used as a client)
# sentence-transformers  (RAG embeddings, runs on the same GPU)
# faiss-cpu              (vector store, CPU-only is enough here)
# pypdf python-docx markdown
# ddgs                   (DuckDuckGo search)
# rich                  (terminal UI)
```

### 3.3 (CRITICAL for ROCm) Remove NVIDIA-only `flash_attn` if present

Some Radeon Cloud images ship with the **NVIDIA CUDA build** of
`flash-attn` pre-installed (it can't actually load on AMD GPUs but vLLM
checks for it during init). If you see:

```
ModuleNotFoundError: No module named 'flash_attn_2_cuda'
```

The fix is one command:

```bash
pip uninstall -y flash-attn
```

After that, vLLM's `VLLM_ATTENTION_BACKEND=ROCM_ATTN` (set for you in
`server/start_vllm.sh`) kicks in and uses the AMD-native attention kernel.

---

## 4. Download the model (Qwen2.5-14B-Instruct, bf16)

Pure HuggingFace is sometimes blocked from certain networks; `hf-mirror.com`
works in Radeon Cloud's region:

```bash
# one-time
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com

# ~28 GB, cached under ~/.cache/huggingface by default
huggingface-cli download Qwen/Qwen2.5-14B-Instruct \
    --local-dir /workspace/persistence/hackathon/models/Qwen2.5-14B-Instruct
```

---

## 5. Launch the vLLM server

```bash
bash server/start_vllm.sh
```

What this script does (see the file for the exact command):

- exports `HSA_OVERRIDE_GFX_VERSION=11.0.0` (only if needed by your vLLM build)
- starts `vllm serve Qwen/Qwen2.5-14B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager` (eager mode avoids long graph compile on first run)

Health check:

```bash
curl http://127.0.0.1:8000/v1/models
# expect {"data":[{"id":"Qwen/Qwen2.5-14B-Instruct",...}]}
```

---

## 6. Run the agent

End-to-end scripted demo (the one shown in our demo video):

```bash
python agent/run_cli.py --topic "Raft consensus algorithm"
```

Interactive REPL:

```bash
python agent/run_cli.py
# > I want to learn "the CAP theorem"
```

## 7. Web front-end

A FastAPI + SSE chat UI is included:

```bash
bash server/start_web.sh            # binds 127.0.0.1:8500
```

Open <http://127.0.0.1:8500> in a browser. Type "我想学 Raft 共识算法" and
watch the agent decompose the skill, search for material, Feynman-explain,
and quiz you — streaming tool calls live.

## 8. (Optional) Expose the server to the internet

### 8a. rc-tunnel (official Radeon Cloud)

The Radeon Cloud platform ships `rc-tunnel` (see the platform's user guide):

```bash
/var/run/secrets/frp-self-service/install
$HOME/.local/bin/rc-tunnel expose --port 8500
# returns https://rc-<random>.radeon.firstdg.ai
```

Point any OpenAI-compatible client at `https://rc-<random>.radeon.firstdg.ai/v1`
and model `Qwen/Qwen2.5-14B-Instruct`.

> ⚠️ rc-tunnel only works on Notebook instances created **after** the feature
> was enabled. If `/var/run/secrets/frp-self-service/install` fails with
> `FRP_BROKER_URL is not injected`, you are on an old Pod — destroy the
> instance and relaunch (with Persistent PVC storage) to get a new one.

### 8b. Cloudflare Tunnel (third-party alternative)

If you can't recreate the instance, a Cloudflare Quick Tunnel needs no
account and no public server:

```bash
# download cloudflared (any arch)
curl -L -o /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /tmp/cloudflared

# get a public HTTPS url in seconds
/tmp/cloudflared tunnel --url http://127.0.0.1:8500
# prints: https://<random>.trycloudflare.com  ← this is your public URL
```

Both tunnels keep the OpenAI-compatible API reachable remotely.

---

## 9. Troubleshooting

| Symptom                                                | Fix                                                                                          |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `rocm-smi` returns 0 GPUs                              | Re-launch the instance with GPU enabled in the template                                       |
| `vllm` errors "gfx1100 unsupported"                    | `export HSA_OVERRIDE_GFX_VERSION=11.0.0` before starting vLLM                                 |
| First inference is very slow (~2–3 min)                | That's the eager/all-gather warmup; subsequent calls are fast                                 |
| `huggingface.co` unreachable                           | `export HF_ENDPOINT=https://hf-mirror.com` then re-download                                    |
| OOM during vLLM start (CUDA/HIP out of memory)         | Lower `--gpu-memory-utilization` to 0.7, or `--max-model-len 4096`                             |
| Embedding model download fails                         | Same `HF_ENDPOINT` fix; model is `BAAI/bge-base-en-v1.5`                                      |
| Tool call isn't fired (model just emits text)          | Make sure `--enable-auto-tool-choice --tool-call-parser hermes` are passed to vllm serve      |

---

## 10. What you should see at the end

A successful end-to-end run yields something like the saved transcript in
`examples/session_raft.jsonl`:

1. The agent decomposes *Raft* into **Leader election / Log replication / Safety / Membership change**.
2. It calls `web_research` to fetch 5 reference links and shows them as the reading list.
3. It walks you through **Feynman explanation** of Leader Election, asking
   you to re-explain in your own words twice.
4. It generates a **5-question quiz**, scores your answers, and stores the
   per-sub-skill mastery.
5. It schedules the **next review** for the Ebbinghaus-smoothed dates and
   writes a **Cornell note** to `notes/raft.md`.

That is the full FeynmanTutor loop — entirely on the local Radeon GPU.
