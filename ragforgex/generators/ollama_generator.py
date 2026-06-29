"""Ollama generator adapter."""

from __future__ import annotations

from typing import Any

import requests

from ragforgex.core.schema import SearchResult
from ragforgex.generators.base import BaseGenerator


class OllamaGenerator(BaseGenerator):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        allow_fallback: bool = True,
        **_: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.allow_fallback = allow_fallback

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        context_text = "\n\n".join(item.text for item in contexts)
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": f"Context:\n{context_text}\n\nQuestion: {question}", "stream": False},
                timeout=90,
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception:
            if self.allow_fallback:
                return f"Question: {question}\n\nGrounded context:\n{context_text}"
            raise

