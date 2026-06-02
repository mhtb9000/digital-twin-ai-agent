


from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from typing import List

from config import SETTINGS
from rag.chunking import chunk_text
from rag.embeddings import Embedder
from rag.loaders import load_documents_from_dir
from rag.retriever import ChromaRetriever


def build_chunks(raw_dir: Path, api_key: str) -> List[dict]:
    docs = load_documents_from_dir(raw_dir)
    embedder = Embedder(api_key=api_key, model=SETTINGS.embedding_model)

    chunk_rows: List[dict] = []
    for doc in docs:
        chunks = chunk_text(doc.text, chunk_size=SETTINGS.chunk_size, overlap=SETTINGS.chunk_overlap)
        for i, chunk in enumerate(chunks):
            embedding = embedder.embed_text(chunk)
            chunk_rows.append(
                {
                    "id": f"{Path(doc.path).stem}_{i}",
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": {
                        "source": doc.path,
                        "title": doc.title,
                        "kind": doc.kind,
                        "chunk_id": i,
                    },
                }
            )
    return chunk_rows


def ingest_corpus(api_key: str, raw_dir: str | None = None) -> int:
    raw_path = Path(raw_dir or SETTINGS.raw_data_dir)
    retriever = ChromaRetriever(api_key=api_key)
    chunk_rows = build_chunks(raw_path, api_key=api_key)
    retriever.add_chunks(chunk_rows)
    return len(chunk_rows)


if __name__ == "__main__":
    import os

    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set GEMINI_API_KEY first.")
    count = ingest_corpus(api_key=key)
    print(f"Ingested {count} chunks.")
