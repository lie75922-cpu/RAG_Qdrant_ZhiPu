"""Generator interface."""

from ragforgex.core.schema import SearchResult


class BaseGenerator:
    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        raise NotImplementedError

