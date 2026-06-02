from __future__ import annotations

from dataclasses import dataclass
from typing import List

from llm.gemini_client import GeminiClient
from llm.prompts import build_response_prompt
from memory.long_term import LongTermMemoryStore
from memory.short_term import ShortTermMemory
from rag.retriever import ChromaRetriever


@dataclass
class GenerationBundle:
    prompt: str
    sources: List[dict]
    memories: List[dict]


class ResponseGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
        retriever: ChromaRetriever,
        memory_store: LongTermMemoryStore,
    ) -> None:
        self.client = GeminiClient(api_key=api_key, model=model)
        self.retriever = retriever
        self.memory_store = memory_store

    def build_context(self, user_id: str, query: str, short_term: ShortTermMemory) -> GenerationBundle:
        docs = self.retriever.search(query, top_k=5)
        memories = self.memory_store.search(user_id=user_id, query=query, top_k=5)

        doc_block = []
        sources = []
        for d in docs:
            sources.append(
                {
                    "title": d.title,
                    "source": d.source,
                    "chunk_id": d.chunk_id,
                    "distance": d.distance,
                }
            )
            doc_block.append(
                f"[{d.title} | chunk {d.chunk_id} | {d.source}]\n{d.text}"
            )

        memory_block = []
        memory_items = []
        for m in memories:
            memory_items.append(
                {
                    "text": m.text,
                    "importance": m.importance,
                    "memory_type": m.memory_type,
                }
            )
            memory_block.append(f"- ({m.memory_type}, importance {m.importance}) {m.text}")

        prompt = build_response_prompt(
            user_query=query,
            short_term_summary=short_term.summary,
            recent_messages=short_term.recent_user_assistant_pairs(),
            retrieved_documents="\n\n---\n\n".join(doc_block),
            retrieved_memories="\n".join(memory_block),
        )
        return GenerationBundle(prompt=prompt, sources=sources, memories=memory_items)

    def generate(self, user_id: str, query: str, short_term: ShortTermMemory) -> tuple[str, GenerationBundle]:
        bundle = self.build_context(user_id=user_id, query=query, short_term=short_term)
        answer = self.client.generate_text(bundle.prompt, temperature=0.35).strip()
        return answer, bundle
