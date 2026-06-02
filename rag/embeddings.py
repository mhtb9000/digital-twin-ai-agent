from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from google import genai


@dataclass
class Embedder:
    api_key: str
    model: str = "gemini-embedding-001"

    def __post_init__(self) -> None:
        self.client = genai.Client(api_key=self.api_key)

    def embed_text(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )
        return list(response.embeddings[0].values)

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]
