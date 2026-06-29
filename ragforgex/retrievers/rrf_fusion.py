"""Reciprocal Rank Fusion."""

from __future__ import annotations

from ragforgex.core.schema import SearchResult


def rrf_fusion(result_sets: list[list[SearchResult]], k: int = 60, top_k: int = 5) -> list[SearchResult]:
    scores: dict[str, float] = {}
    items: dict[str, SearchResult] = {}
    for results in result_sets:
        for rank, result in enumerate(results, start=1):
            key = result.text
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items[key] = result
    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        SearchResult(text=items[text].text, score=score, metadata=items[text].metadata)
        for text, score in fused
    ]

