"""Vector store interface."""

from ragforgex.core.schema import Chunk, SearchResult


class BaseStore:
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise NotImplementedError

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        raise NotImplementedError

