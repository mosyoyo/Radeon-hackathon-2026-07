"""material_loader — parse a learner-uploaded PDF/DOCX/MD file into chunks."""

from __future__ import annotations
import os
import json
from pathlib import Path


def parse_uploaded_material(file_path: str, chunk_chars: int = 2000, overlap: int = 200) -> str:
    """Parse a user-uploaded file (PDF/DOCX/MD/TXT) and split it into chunks.
    Returns a JSON object with `n_pages`, `n_chunks`, `preview`, and the
    full chunks list. The agent can then call `index_material` to store them
    in the local RAG knowledge base.
    """
    p = Path(file_path)
    if not p.exists():
        return json.dumps({"error": f"file not found: {file_path}"})

    text = ""
    ext = p.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        n_pages = len(reader.pages)
    elif ext == ".docx":
        import docx  # python-docx
        d = docx.Document(str(p))
        text = "\n\n".join(para.text for para in d.paragraphs if para.text.strip())
        n_pages = max(1, len(text) // 3500)
    elif ext in {".md", ".markdown"}:
        text = p.read_text(encoding="utf-8", errors="ignore")
        n_pages = max(1, len(text) // 3500)
    elif ext == ".txt":
        text = p.read_text(encoding="utf-8", errors="ignore")
        n_pages = max(1, len(text) // 3500)
    else:
        return json.dumps({"error": f"unsupported file type: {ext}"})

    chunks = []
    start = 0
    n = len(text)
    step = max(1, chunk_chars - overlap)
    while start < n:
        end = min(start + chunk_chars, n)
        chunks.append(text[start:end])
        start += step

    preview = text[:300] + ("..." if len(text) > 300 else "")
    return json.dumps({
        "file": str(p),
        "n_pages": n_pages,
        "n_chunks": len(chunks),
        "preview": preview,
        "chunks": chunks,
    }, ensure_ascii=False)


MATERIAL_LOADER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "parse_uploaded_material",
        "description": (
            "Parse a user-uploaded learning material file (PDF/DOCX/MD/TXT) "
            "and split the text into reusable chunks. Call this AFTER the "
            "learner provides a file path, then call `index_material` to "
            "store the chunks in the RAG knowledge base."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "absolute or relative path to the file"},
                "chunk_chars": {"type": "integer", "default": 2000},
                "overlap": {"type": "integer", "default": 200},
            },
            "required": ["file_path"],
        },
    },
}
