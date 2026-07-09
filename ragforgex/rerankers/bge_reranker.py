"""BGE reranker adapter with a lexical fallback."""

from __future__ import annotations

from typing import Any

from ragforgex.core.schema import SearchResult
from ragforgex.rerankers.base import BaseReranker


class BGEReranker(BaseReranker):
    def __init__(self, model: str = "BAAI/bge-reranker-base", allow_fallback: bool = True, **_: Any) -> None:
        self.model_name = model
        self._model = None
        try:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(model, use_fp16=False)
        except Exception:
            if not allow_fallback:
                raise

    def rerank(self, question: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if self._model is not None:
            pairs = [[question, result.text] for result in results]
            scores = self._model.compute_score(pairs)
            if not isinstance(scores, list):
                scores = [float(scores)]
            reranked = [
                SearchResult(text=result.text, score=float(score), metadata=result.metadata)
                for result, score in zip(results, scores, strict=False)
            ]
        else:
            query_terms = {term.lower() for term in question.split()}
            reranked = []
            for result in results:
                text_terms = {term.lower().strip(".,!?;:") for term in result.text.split()}
                overlap = len(query_terms & text_terms) / max(len(query_terms), 1)
                reranked.append(
                    SearchResult(
                        text=result.text,
                        score=float(result.score + overlap),
                        metadata=result.metadata,
                    )
                )
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]
