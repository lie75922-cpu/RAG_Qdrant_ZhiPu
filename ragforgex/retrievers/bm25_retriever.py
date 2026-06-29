"""BM25 retrieval with dependency fallback."""

from __future__ import annotations

from collections import Counter
from math import log
from typing import Any

from ragforgex.core.schema import Chunk, SearchResult
from ragforgex.retrievers.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(self, chunks: list[Chunk], **_: Any) -> None:
        self.chunks = chunks
        self.tokens = [chunk.text.lower().split() for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self.tokens)
        except Exception:
            self._bm25 = None

    def retrieve(self, question: str, top_k: int = 5) -> list[SearchResult]:
        query = question.lower().split()
        if self._bm25 is not None:
            scores = self._bm25.get_scores(query)
        else:
            scores = self._fallback_scores(query)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            SearchResult(text=self.chunks[idx].text, score=float(score), metadata=self.chunks[idx].metadata)
            for idx, score in ranked
            if score > 0
        ]

    def _fallback_scores(self, query: list[str]) -> list[float]:
        doc_freq = Counter(token for tokens in self.tokens for token in set(tokens))
        total = max(len(self.tokens), 1)
        scores: list[float] = []
        for tokens in self.tokens:
            counts = Counter(tokens)
            score = 0.0
            for token in query:
                if token in counts:
                    score += counts[token] * log((total + 1) / (doc_freq[token] + 0.5))
            scores.append(score)
        return scores

