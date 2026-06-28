"""Minimal RAG pipeline: PDF -> chunks -> local embeddings -> FAISS -> retrieval.

Kept dependency-light and in-memory so it's easy to demo. For a 12-hour build this
is plenty; swap FAISS for a persistent store later if you need it.
"""
import numpy as np
import faiss
from pypdf import PdfReader

from bedrock_client import embed


def load_pdf(file) -> list[dict]:
    """Read a PDF into a list of {"text", "page"} records (one per page)."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "page": i})
    return pages


def chunk_pages(pages: list[dict], size: int = 1200, overlap: int = 200) -> list[dict]:
    """Split page text into overlapping character chunks, preserving page numbers."""
    chunks = []
    for p in pages:
        text, page_no = p["text"], p["page"]
        start = 0
        while start < len(text):
            piece = text[start : start + size]
            chunks.append({"text": piece, "page": page_no})
            start += size - overlap
    return chunks


class VectorStore:
    """In-memory FAISS index over chunk embeddings, with cosine similarity."""

    def __init__(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("No chunks to index — the document had no extractable text.")
        self.chunks = chunks
        vectors = np.array(embed([c["text"] for c in chunks]), dtype="float32")
        faiss.normalize_L2(vectors)  # so inner product == cosine similarity
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = np.array(embed(query), dtype="float32")
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            hit = dict(self.chunks[idx])
            hit["score"] = float(score)
            results.append(hit)
        return results


def build_prompt(question: str, contexts: list[dict]) -> list[dict]:
    """Assemble grounded chat messages with numbered, page-cited context."""
    blocks = []
    for i, c in enumerate(contexts, start=1):
        blocks.append(f"[{i}] (page {c['page']}) {c['text']}")
    context_text = "\n\n".join(blocks)

    system = (
        "You answer strictly from the provided document excerpts. "
        "Cite the sources you use with their page numbers like (page 3). "
        "If the answer isn't in the excerpts, say you don't know."
    )
    user = f"Document excerpts:\n\n{context_text}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
