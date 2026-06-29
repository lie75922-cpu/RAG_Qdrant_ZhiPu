"""Reranker interface."""

from typing import Any

from ragforgex.core.schema import SearchResult


class BaseReranker:
    def rerank(self, question: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        raise NotImplementedError


class NoOpReranker(BaseReranker):
    def __init__(self, **_: Any) -> None:
        pass

    def rerank(self, question: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        return results[:top_k]

