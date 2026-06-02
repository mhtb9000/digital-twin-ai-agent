from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db"
DOC_INDEX_DIR = DATA_DIR / "doc_index"
MEMORY_DIR = DATA_DIR / "memory"

for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR, CHROMA_DIR, DOC_INDEX_DIR, MEMORY_DIR):
    path.mkdir(parents=True, exist_ok=True)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    chroma_path: str = os.getenv("CHROMA_PATH", str(CHROMA_DIR))
    raw_data_dir: str = os.getenv("RAW_DATA_DIR", str(RAW_DIR))
    max_top_k: int = env_int("MAX_TOP_K", 5)
    memory_top_k: int = env_int("MEMORY_TOP_K", 5)
    recent_turns: int = env_int("RECENT_TURNS", 8)
    chunk_size: int = env_int("CHUNK_SIZE", 1100)
    chunk_overlap: int = env_int("CHUNK_OVERLAP", 180)
    max_context_chars: int = env_int("MAX_CONTEXT_CHARS", 25000)


SETTINGS = Settings()
