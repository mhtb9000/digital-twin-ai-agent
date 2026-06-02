from __future__ import annotations

from typing import List

from llm.gemini_client import GeminiClient
from llm.prompts import MEMORY_EXTRACTION_PROMPT
from utils.json_utils import safe_json_loads


def extract_memories(client: GeminiClient, user_message: str, assistant_message: str) -> List[dict]:
    prompt = MEMORY_EXTRACTION_PROMPT.format(
        user_message=user_message.strip(),
        assistant_message=assistant_message.strip(),
    )
    text = client.generate_text(prompt, temperature=0.1)
    try:
        payload = safe_json_loads(text)
    except Exception:
        return []

    memories = payload.get("memories", []) if isinstance(payload, dict) else []
    cleaned = []
    for mem in memories:
        if not isinstance(mem, dict):
            continue
        candidate = str(mem.get("text", "")).strip()
        if not candidate:
            continue
        cleaned.append(
            {
                "text": candidate,
                "importance": int(mem.get("importance", 3)),
                "memory_type": str(mem.get("memory_type", "context")),
            }
        )
    return cleaned
