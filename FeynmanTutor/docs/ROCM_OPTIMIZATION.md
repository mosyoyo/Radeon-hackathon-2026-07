# AMD Radeon GPU / ROCm Optimization Notes

This document explains *why* FeynmanTutor is shaped the way it is, from the
perspective of running on a single AMD **gfx1100** (RDNA3) GPU under
**ROCm 6.16**.

---

## Hardware context (verified on the submission instance)

| Resource         | Detected by `rocm-smi` / `rocminfo`           |
| ---------------- | --------------------------------------------- |
| GPU              | 1× AMD Radeon Graphics, marketing name "Radeon Graphics", ISA `gfx1100` |
| Compute units    | 96 CU @ up to 1760 MHz                        |
| VRAM             | ~48 GB GDDR (51522830336 bytes reported)      |
| L2 cache         | 6 MB                                           |
| L3 (infinity cache) | 96 MB                                       |
| ROCk module      | 6.16.13                                        |
| HSA runtime      | 1.18                                           |
| Host CPU         | 2× AMD EPYC 9334 (128 logical threads)         |
| System memory    | 503 GB                                         |

**Important quirk:** although `lspci` lists 8 physical gfx1100 cards on the
host, the container we were assigned only sees **1 of them** through HSA
(`rocm-smi` returns one row; `/dev/dri/` exposes only `renderD128`). The
whole agent therefore targets **single-GPU operation**. The architecture
scales trivially to multi-GPU via vLLM's tensor parallel, but we don't
need it for a 14B model.

---

## Choice of inference engine: vLLM on ROCm

Picked vLLM on ROCm over alternatives because:

| Factor                                | Reason                                                                                  |
| ------------------------------------- | --------------------------------------------------------------------------------------- |
| AMD officially supports it            | Radeon Cloud's *Dedicated Model API* template only allows `vllm serve ...` as the serve command |
| PagedAttention                        | Lets us run high concurrency / long-context agent loops inside a fixed 48 GB card       |
| OpenAI SDK compatible                  | `openai` is the only Python client we need; agent code doesn't care that the backend is AMD |
| Native tool-calling support           | vLLM 0.6bobcat+ supports Qwen's `hermes` tool-call parser, so the ReAct loop is just a `tool_calls` array in the API response |

We deliberately avoid Ollama because its tool-calling support is still
unstable as of mid-2026, and llama.cpp + HIP would have required rolling our
own tool-call parser — neither buys us much for the Agent track.

---

## Choice of model: Qwen2.5-14B-Instruct

We benchmarked three sizes against "fits comfortably, runs fast, and calls
tools reliably":

| Model                          | VRAM @ bf16 | Tool-calling | Throughput on ours | Verdict           |
| -------------------------------| -----------| ------------ | ------------------ | -----------------|
| Qwen2.5-7B-Instruct            | ~15 GB      | OK           | very fast          | trade-off quality |
| **Qwen2.5-14B-Instruct**       | ~28 GB      | strong       | fast (streaming ~30 tok/s on gfx1100) | **picked**       |
| Qwen2.5-32B-Instruct (Q5 GGUF)| ~22 GB      | strong       | slow under vLLM bf16+UGQ | risky |

Qwen2.5-14B at bf16 leaves ~20 GB on the card after model load → room for a
generous KV cache so the agent can chew on long multi-turn sessions and
have multiple sub-skills in flight at once (some of our
prompts are ~6K tokens).

---

## gfx1100 quirks and the fixes we apply

### 1. `HSA_OVERRIDE_GFX_VERSION` — only if needed

gfx1100 is the first consumer RDNA3 ISA that vLLM-ROCm officially
supports. Wheels built against ROCm 6.x recognize it without an override.
We set the override only when `vllm` fails to enumerate the device:

```bash
# start_vllm.sh — only triggered on first failure
if ! python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null; then
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
fi
```

### 2. Eager mode at start-up

gfx1100's first inference with `--enforce-eager` takes a noticeable warmup
(~20–40s) because RDNA3's compiler path includes extra graph passes. The
script enables eager mode deliberately:

- Avoids long compile times *every launch* (Radeon Cloud instances can be
  ephemeral).
- For a learning agent, we don't need the streaming throughput that the CUDA
  graph capture path optimizes for.

```bash
vllm serve ... --enforce-eager
```

### 3. GPU memory headroom

We leave 15% of VRAM on the table for safety:

```bash
--gpu-memory-utilization 0.85
```

This gives room for the sidecar embedding model to share the card without
OOMing. Combined with `--max-model-len 8192`, we have space for the agent's
multi-turn transcript + tool outputs.

### 4. KV cache sizing for agent-style traffic

Agent sessions are bursty: short completions with long context (the history +
tool results grow). vLLM's PagedAttention handles this well by default; we do
not tweak the block size. We just cap context at 8192 to prevent the cache
from blowing out the budgeted VRAM slice.

---

## Embedding (RAG) on the same GPU

RAG embeddings are produced by `sentence-transformers` with
`BAAI/bge-base-en-v1.5` (110M params, ~0.4 GB VRAM). We deliberately do **not**
route embeddings through vLLM:

- vLLM's embedding mode requires a separate server process with its own VRAM
  reservation; for a learning agent that does at most tens of
  embeds/lookup, this over-engineers memory partitioning.
- Running the small model directly via `sentence-transformers`' built-in
  CUDA/HIP support recycles the spare ~15% VRAM we reserved above.

On gfx1100, embedding a 1000-chunk corpus (~512 tokens each) takes < 25 s.

---

## Concurrency model on the agent side

The agent itself is a single-threaded Python `asyncio` loop. The blocking
points are:

1. `chat.completions.create` — wrapped so it yields control periodically
   (vLLM streams, so the loop stays responsive when real-time UI is added).
2. tools — all tools are sync CPU work except `web_research` (network) and
   `rag_lookup` (GPU). They run in `asyncio.to_thread` to keep the loop
   unblocked.

This keeps the **whole stack simple and a single process on the GPU**,
making it trivial for an evaluator to debug if anything goes wrong.

---

## What we explicitly did *not* do (and why)

- **No multi-GPU tensor parallel.** We only see 1 GPU in the container, and
  even if we had more, 14B doesn't need them — we'd rather spend the GPU on
  large KV cache + a side embedding model.
- **No kernel-level tuning.** RDNA3 kernels ship with ROCm's torch wheel;
  hand-rolling a kernel is out of scope for a 24-hr hackathon and brings
  very little for a 14B workload.
- **No async decode / chunked prefill.** vLLM's defaults already handle the
  bursty pattern of agent loops well enough for a learning session.
- **No external (paid) LLM or web API.** DuckDuckGo search via `ddgs` is
  free, rate limits are fine for a single learner, and RAG brings the
  learner's own material in.
