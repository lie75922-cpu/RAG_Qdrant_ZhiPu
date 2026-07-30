"""Vector store interface."""

from __future__ import annotations

from pathlib import Path

from ragforgex.core.schema import Chunk, SearchResult


class BaseStore:
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise NotImplementedError

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        raise NotImplementedError

    @property
    def chunks(self) -> list[Chunk]:
        return []

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("This store does not support local persistence.")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError("This store does not support local persistence.")

