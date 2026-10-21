"""Runtime configuration.

Every value can be overridden with an environment variable, so no secret or
machine-specific path is ever hard-coded in the source tree.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # optional convenience: read a local .env file if python-dotenv is present
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - the project runs fine without it
    pass

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT / "data"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", ROOT / "index"))

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Indexing
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "250"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
MIN_CHUNK_WORDS = int(os.getenv("RAG_MIN_CHUNK_WORDS", "30"))

# Retrieval
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# Generation
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.getenv("RAG_LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.2"))
