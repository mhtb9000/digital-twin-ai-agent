from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from google import genai


@dataclass
class GeminiClient:
    api_key: str
    model: str = "gemini-2.5-flash"

    def __post_init__(self) -> None:
        self.client = genai.Client(api_key=self.api_key)

    def generate_text(self, prompt: str, temperature: float = 0.4) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return getattr(response, "text", "") or ""
