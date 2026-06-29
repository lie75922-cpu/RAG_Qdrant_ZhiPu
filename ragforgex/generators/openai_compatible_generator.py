"""OpenAI-compatible chat completion adapter."""

from __future__ import annotations

from typing import Any

import requests

from ragforgex.core.schema import SearchResult
from ragforgex.generators.base import BaseGenerator


class OpenAICompatibleGenerator(BaseGenerator):
    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        allow_fallback: bool = True,
        **_: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.allow_fallback = allow_fallback

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        context_text = "\n\n".join(item.text for item in contexts)
        if not self.base_url or not self.api_key:
            if self.allow_fallback:
                return f"Question: {question}\n\nGrounded context:\n{context_text}"
            raise ValueError("OpenAI-compatible generator requires base_url and api_key.")
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": "Answer using only the provided context."},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
                ],
            },
            timeout=90,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

