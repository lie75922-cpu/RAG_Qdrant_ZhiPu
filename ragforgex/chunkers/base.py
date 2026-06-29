"""Chunker interface."""

from ragforgex.core.schema import Chunk, Document


class BaseChunker:
    def split(self, documents: list[Document]) -> list[Chunk]:
        raise NotImplementedError

