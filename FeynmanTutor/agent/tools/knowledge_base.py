"""knowledge_base — local FAISS vector store + RAG retrieval.

Why we don't use `sentence-transformers`:
- Installing `sentence-transformers` pulls a fresh copy of `torch`, which
  would clobber the prebuilt ROCm torch in /opt/venv.
- We can use `transformers` (already installed) directly with a HF
  embedding model — the only thing we need is AutoModel + AutoTokenizer and
  a mean-pool on last_hidden_state.

The embedding model is `BAAI/bge-base-en-v1.5` (110M params, ~0.4 GB VRAM).
It runs on the same gfx1100 GPU as vLLM, occupying the spare ~15% VRAM
we'd reserved.
"""

from __future__ import annotations
import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import faiss

# --- model singleton (lazy loaded, shared across calls) -------------------

_EMB_MODEL = None
_EMB_TOK = None
_EMB_DEVICE = None
_EMB_DIM = 768  # bge-base hidden size


def _ensure_embedder():
    """Lazily load BAAI/bge-base-en-v1.5 onto the GPU once per process."""
    global _EMB_MODEL, _EMB_TOK, _EMB_DEVICE
    if _EMB_MODEL is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_id = "BAAI/bge-base-en-v1.5"
    cache_root = Path(os.environ.get(
        "FT_EMBED_CACHE",
        "/workspace/persistence/hackathon/models/bge-base-en-v1.5",
    ))
    cache_root.mkdir(parents=True, exist_ok=True)

    _EMB_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _EMB_TOK = AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_root))
    _EMB_MODEL = AutoModel.from_pretrained(model_id, cache_dir=str(cache_root)).to(_EMB_DEVICE)
    _EMB_MODEL.eval()


def _embed(texts: list[str]) -> np.ndarray:
    """Mean-pooled last hidden states, L2-normalized. n_texts × dim."""
    import torch
    _ensure_embedder()
    out = np.zeros((len(texts), _EMB_DIM), dtype="float32")
    bs = 16
    for i in range(0, len(texts), bs):
        batch = texts[i: i + bs]
        toks = _EMB_TOK(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(_EMB_DEVICE)
        with torch.no_grad():
            hs = _EMB_MODEL(**toks).last_hidden_state  # B,T,H
        mask = toks.attention_mask.unsqueeze(-1).float()  # B,T,1
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        out[i: i + bs] = pooled.cpu().numpy()
    return out


# --- store ----------------------------------------------------------------

KB_ROOT = Path(os.environ.get("FT_KB_ROOT", "/workspace/persistence/hackathon/FeynmanTutor/data/kb"))
KB_ROOT.mkdir(parents=True, exist_ok=True)


def _corpus_path(corpus_id: str) -> Path:
    safe = hashlib.sha1(corpus_id.encode()).hexdigest()[:12]
    return KB_ROOT / f"{safe}_{corpus_id}"


def index_material(corpus_id: str, chunks: list[str]) -> str:
    """Embed `chunks` and store them under `corpus_id` in a local FAISS index.
    Returns the number of vectors stored. Idempotent: re-indexing the same
    corpus replaces the previous one.
    """
    if not chunks:
        return json.dumps({"error": "no chunks to index", "corpus_id": corpus_id})
    vecs = _embed(chunks)
    index = faiss.IndexFlatIP(_EMB_DIM)
    index.add(vecs)

    out = _corpus_path(corpus_id)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "index.faiss"))
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    return json.dumps({"corpus_id": corpus_id, "n_vectors": int(index.ntotal), "path": str(out)})


def rag_lookup(query: str, corpus_id: str, top_k: int = 4) -> str:
    """Look up top-K relevant passages from `corpus_id`. Returns JSON with
    passage + score for each hit. If the corpus doesn't exist, returns an
    empty result list (so the agent can fall back to its own knowledge)."""
    top_k = max(1, min(int(top_k), 16))
    p = _corpus_path(corpus_id)
    if not (p / "index.faiss").exists():
        return json.dumps({"query": query, "corpus_id": corpus_id, "results": []})
    index = faiss.read_index(str(p / "index.faiss"))
    chunks = json.loads((p / "chunks.json").read_text())
    qv = _embed([query])
    scores, idx = index.search(qv, top_k)
    hits = []
    for s, i in zip(scores[0], idx[0]):
        if i < 0 or i >= len(chunks):
            continue
        hits.append({"score": float(s), "passage": chunks[i][:1000]})
    return json.dumps({"query": query, "corpus_id": corpus_id, "results": hits}, ensure_ascii=False)


INDEX_MATERIAL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "index_material",
        "description": (
            "Embed text chunks and store them in a local FAISS vector index so "
            "the agent can later retrieve relevant passages via `rag_lookup`. "
            "Always call this after `parse_uploaded_material` or when you have "
            "fresh snippets you want to keep for the session."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "corpus_id": {"type": "string", "description": "a short name for the corpus (e.g. 'raft-paper')"},
                "chunks": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["corpus_id", "chunks"],
        },
    },
}

RAG_LOOKUP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "rag_lookup",
        "description": (
            "Retrieve the most relevant passages from a previously indexed corpus. "
            "Use this before answering learner questions when you want to ground "
            "your answer in their uploaded material."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "corpus_id": {"type": "string"},
                "top_k": {"type": "integer", "default": 4},
            },
            "required": ["query", "corpus_id"],
        },
    },
}
