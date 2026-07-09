"""Lightweight semantic chunking.

The implementation is dependency-light by design. It groups adjacent sentences by
lexical overlap so examples and tests run without embedding models. Production
users can replace this with an embedding-backed semantic splitter later.
"""

from __future__ import annotations

import re
from typing import Any

from ragforgex.chunkers.base import BaseChunker
from ragforgex.core.schema import Chunk, Document

_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _tokens(text: str) -> set[str]:
    return {token.strip(".,!?;:()[]{}\"'").lower() for token in text.split() if token.strip()}


class SemanticChunker(BaseChunker):
    def __init__(
        self,
        max_sentences: int = 5,
        similarity_threshold: float = 0.15,
        **_: Any,
    ) -> None:
        self.max_sentences = max_sentences
        self.similarity_threshold = similarity_threshold

    def split(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            sentences = [item.strip() for item in _SENTENCE_PATTERN.split(doc.text.strip()) if item.strip()]
            group: list[str] = []
            group_tokens: set[str] = set()
            for sentence in sentences:
                sentence_tokens = _tokens(sentence)
                overlap = len(group_tokens & sentence_tokens) / max(len(group_tokens | sentence_tokens), 1)
                should_start_new = (
                    group
                    and (len(group) >= self.max_sentences or overlap < self.similarity_threshold)
                )
                if should_start_new:
                    chunks.append(self._chunk(doc, group, len(chunks)))
                    group = []
                    group_tokens = set()
                group.append(sentence)
                group_tokens |= sentence_tokens
            if group:
                chunks.append(self._chunk(doc, group, len(chunks)))
        return chunks

    @staticmethod
    def _chunk(document: Document, sentences: list[str], index: int) -> Chunk:
        metadata = dict(document.metadata)
        metadata.update({"chunker": "semantic", "semantic_group": index})
        return Chunk(text=" ".join(sentences), metadata=metadata)

