"""Retriever interface."""

from ragforgex.core.schema import SearchResult


class BaseRetriever:
    def retrieve(self, question: str, top_k: int = 5) -> list[SearchResult]:
        raise NotImplementedError

