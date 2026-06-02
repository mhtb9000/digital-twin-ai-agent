from __future__ import annotations

from dataclasses import dataclass
from typing import List

import chromadb

from config import SETTINGS
from rag.embeddings import Embedder


@dataclass
class MemoryHit:
    text: str
    importance: int
    memory_type: str
    distance: float
    metadata: dict


class LongTermMemoryStore:
    def __init__(
        self,
        api_key: str,
        chroma_path: str | None = None,
        embed_model: str | None = None,
        collection_name: str = "neil_memories",
    ) -> None:
        self.embedder = Embedder(api_key=api_key, model=embed_model or SETTINGS.embedding_model)
        self.client = chromadb.PersistentClient(path=chroma_path or SETTINGS.chroma_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_memories(self, user_id: str, memories: List[dict]) -> None:
        if not memories:
            return

        ids = []
        docs = []
        metas = []
        embeds = []

        for idx, mem in enumerate(memories):
            text = str(mem.get("text", "")).strip()
            if not text:
                continue
            ids.append(f"{user_id}_{len(ids)}_{abs(hash(text))}")
            docs.append(text)
            embeds.append(self.embedder.embed_text(text))
            metas.append(
                {
                    "user_id": user_id,
                    "importance": int(mem.get("importance", 3)),
                    "memory_type": str(mem.get("memory_type", "context")),
                }
            )

        if ids:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)

    def search(self, user_id: str, query: str, top_k: int = 5) -> List[MemoryHit]:
        query_embedding = self.embedder.embed_text(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"],
        )

        items: List[MemoryHit] = []
        if not result.get("documents"):
            return items

        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            items.append(
                MemoryHit(
                    text=str(doc),
                    importance=int(meta.get("importance", 3)),
                    memory_type=str(meta.get("memory_type", "context")),
                    distance=float(dist),
                    metadata=dict(meta),
                )
            )
        return items

    def has_any(self, user_id: str) -> bool:
        result = self.collection.get(where={"user_id": user_id}, limit=1)
        return bool(result.get("ids"))
