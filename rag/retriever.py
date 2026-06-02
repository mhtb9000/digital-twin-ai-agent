from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import chromadb

from config import SETTINGS
from rag.embeddings import Embedder


@dataclass
class RetrievalResult:
    text: str
    source: str
    title: str
    chunk_id: int
    distance: float
    metadata: dict


class ChromaRetriever:
    def __init__(
        self,
        api_key: str,
        chroma_path: str | None = None,
        embed_model: str | None = None,
        collection_name: str = "neil_corpus",
    ) -> None:
        self.embedder = Embedder(api_key=api_key, model=embed_model or SETTINGS.embedding_model)
        self.client = chromadb.PersistentClient(path=chroma_path or SETTINGS.chroma_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: List[dict]) -> None:
        if not chunks:
            return

        ids = []
        docs = []
        metas = []
        embeds = []
        for item in chunks:
            ids.append(item["id"])
            docs.append(item["text"])
            metas.append(item["metadata"])
            embeds.append(item["embedding"])

        self.collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeds,
        )

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        query_embedding = self.embedder.embed_text(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        items: List[RetrievalResult] = []
        if not result.get("documents"):
            return items

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            items.append(
                RetrievalResult(
                    text=doc,
                    source=str(meta.get("source", "")),
                    title=str(meta.get("title", "")),
                    chunk_id=int(meta.get("chunk_id", 0)),
                    distance=float(dist),
                    metadata=dict(meta),
                )
            )
        return items
