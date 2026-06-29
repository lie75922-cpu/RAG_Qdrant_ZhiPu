"""Simple retrieval metrics."""

from __future__ import annotations

from typing import Any

from ragforgex.core.schema import RAGAnswer
from ragforgex.evaluators.base import BaseEvaluator


class RetrievalMetricsEvaluator(BaseEvaluator):
    def __init__(self, **_: Any) -> None:
        pass

    def evaluate(self, answer: RAGAnswer, expected_contexts: list[str] | None = None) -> dict[str, Any]:
        if not expected_contexts:
            return {"context_count": len(answer.contexts), "top_score": answer.contexts[0].score if answer.contexts else 0.0}
        retrieved = {item.text for item in answer.contexts}
        expected = set(expected_contexts)
        hits = len(retrieved & expected)
        return {"recall": hits / max(len(expected), 1), "precision": hits / max(len(retrieved), 1)}

