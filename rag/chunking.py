from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Chunk:
    text: str
    chunk_id: int
    source: str
    title: str
    extra: dict


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts


def chunk_text(text: str, chunk_size: int = 1100, overlap: int = 180) -> List[str]:
    text = clean_text(text)
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())

        if len(para) <= chunk_size:
            current = para
            continue

        start = 0
        while start < len(para):
            end = min(len(para), start + chunk_size)
            part = para[start:end].strip()
            if part:
                chunks.append(part)
            start = max(end - overlap, end)

        current = ""

    if current.strip():
        chunks.append(current.strip())

    return chunks
