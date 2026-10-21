"""Build the FAISS index from the plain-text corpus in ``data/``.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --chunk-size 150 --overlap 30
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402


def chunk_words(words, size: int, overlap: int, min_words: int):
    """Split a word list into overlapping fixed-size chunks."""
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if len(piece) >= min_words:
            yield start, piece


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FAISS index.")
    parser.add_argument("--data-dir", default=str(config.DATA_DIR))
    parser.add_argument("--index-dir", default=str(config.INDEX_DIR))
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--min-words", type=int, default=config.MIN_CHUNK_WORDS)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {data_dir}")

    documents, metas = [], []
    for path in files:
        text = " ".join(path.read_text(encoding="utf-8").split())
        words = text.split()
        for n, (start, piece) in enumerate(
            chunk_words(words, args.chunk_size, args.overlap, args.min_words)
        ):
            documents.append(" ".join(piece))
            metas.append({"source": path.name, "chunk_id": n, "start_word": start})

    if not documents:
        raise SystemExit("Chunking produced no chunks — check --chunk-size / --min-words.")

    print(f"[1/3] {len(files)} documents -> {len(documents)} chunks "
          f"(size={args.chunk_size}, overlap={args.overlap})")

    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        documents, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)

    print(f"[2/3] embedded {embeddings.shape[0]} chunks, dim={embeddings.shape[1]}")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_dir / "faiss.index"))

    with open(index_dir / "store.pkl", "wb") as fh:
        pickle.dump(
            {
                "documents": documents,
                "metas": metas,
                "model_name": args.model,
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
            },
            fh,
        )

    print(f"[3/3] index written to {index_dir} ({index.ntotal} vectors)")


if __name__ == "__main__":
    main()
