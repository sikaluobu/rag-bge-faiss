"""Retrieval core: load the FAISS index and answer by similarity search."""

from __future__ import annotations

import pickle
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from . import config


class RAGEngine:
    """Thin wrapper around a FAISS inner-product index of BGE embeddings."""

    def __init__(self, index_dir=None, model_name: str | None = None) -> None:
        index_dir = index_dir or config.INDEX_DIR
        index_path = index_dir / "faiss.index"
        store_path = index_dir / "store.pkl"

        if not index_path.exists() or not store_path.exists():
            raise RuntimeError(
                f"Index not found in {index_dir}. Run `python scripts/build_index.py` first."
            )

        self.index = faiss.read_index(str(index_path))
        with open(store_path, "rb") as fh:
            store = pickle.load(fh)

        self.documents: List[str] = store["documents"]
        self.metas: List[Dict[str, Any]] = store.get("metas", [])
        self.model_name = model_name or store.get("model_name", config.EMBEDDING_MODEL)

        if self.index.ntotal != len(self.documents):
            raise RuntimeError(
                f"Index/document mismatch: {self.index.ntotal} vectors vs "
                f"{len(self.documents)} chunks. Rebuild the index."
            )

        self._embedder = None  # lazily loaded: importing is cheap, encoding is not

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Return the top-k chunks as ``{score, text, meta}`` dicts."""
        top_k = top_k or config.TOP_K
        scores, indices = self.index.search(self.embed([query]), top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 when the index is smaller than k
                continue
            results.append(
                {
                    "score": float(score),
                    "text": self.documents[idx],
                    "meta": self.metas[idx] if idx < len(self.metas) else {},
                }
            )
        return results

    @staticmethod
    def build_prompt(query: str, retrieved: List[Dict[str, Any]]) -> str:
        """Combine evidence and question into a grounding-constrained prompt."""
        context = "\n\n".join(
            f"[{i}] (source: {r['meta'].get('source', 'unknown')}) {r['text']}"
            for i, r in enumerate(retrieved, start=1)
        )
        return (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above. "
            "If the context does not contain the answer, say you don't know."
        )
