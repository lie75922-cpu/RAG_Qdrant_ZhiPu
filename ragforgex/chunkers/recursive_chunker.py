"""Recursive character chunking."""

from __future__ import annotations

from typing import Any

from ragforgex.chunkers.base import BaseChunker
from ragforgex.core.schema import Chunk, Document


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120, **_: Any) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        for doc in documents:
            text = doc.text.strip()
            for start in range(0, len(text), step):
                piece = text[start : start + self.chunk_size].strip()
                if piece:
                    metadata = dict(doc.metadata)
                    metadata.update({"chunk_start": start})
                    chunks.append(Chunk(text=piece, metadata=metadata))
        return chunks

