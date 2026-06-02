from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ShortTermMemory:
    max_messages: int = 12
    messages: List[Message] = field(default_factory=list)
    summary: str = ""

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        self.messages = []
        self.summary = ""

    def as_text(self) -> str:
        return "\n".join(f"{m.role.title()}: {m.content}" for m in self.messages)

    def recent_user_assistant_pairs(self, n_pairs: int = 4) -> str:
        if not self.messages:
            return ""
        relevant = self.messages[-(n_pairs * 2) :]
        return "\n".join(f"{m.role.title()}: {m.content}" for m in relevant)
