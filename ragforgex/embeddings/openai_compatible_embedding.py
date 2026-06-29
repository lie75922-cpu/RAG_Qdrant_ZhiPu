"""OpenAI-compatible embedding adapter."""

from __future__ import annotations

from typing import Any

import requests

from ragforgex.embeddings.base import BaseEmbedding
from ragforgex.embeddings.sentence_transformers_embedding import deterministic_vector


class OpenAICompatibleEmbedding(BaseEmbedding):
    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        allow_fallback: bool = True,
        **_: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.allow_fallback = allow_fallback

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.base_url or not self.api_key:
            if self.allow_fallback:
                return [deterministic_vector(text, self.dimension) for text in texts]
            raise ValueError("OpenAI-compatible embedding requires base_url and api_key.")
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in data]

