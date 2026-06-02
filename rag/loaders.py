from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import fitz  # PyMuPDF


@dataclass
class LoadedDocument:
    path: str
    title: str
    text: str
    kind: str


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf_file(path: Path) -> str:
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n\n".join(pages)


def load_document(path: Path) -> LoadedDocument | None:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = load_text_file(path)
        return LoadedDocument(path=str(path), title=path.stem, text=text, kind=suffix.lstrip("."))
    if suffix == ".pdf":
        text = load_pdf_file(path)
        return LoadedDocument(path=str(path), title=path.stem, text=text, kind="pdf")
    return None


def load_documents_from_dir(raw_dir: Path) -> List[LoadedDocument]:
    docs: List[LoadedDocument] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file():
            doc = load_document(path)
            if doc and doc.text.strip():
                docs.append(doc)
    return docs
