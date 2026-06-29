"""Ragas evaluator adapter."""

from __future__ import annotations

from typing import Any

from ragforgex.core.schema import RAGAnswer
from ragforgex.evaluators.base import BaseEvaluator


class RagasEvaluator(BaseEvaluator):
    def __init__(self, **_: Any) -> None:
        try:
            import ragas  # noqa: F401

            self.available = True
        except Exception:
            self.available = False

    def evaluate(self, answer: RAGAnswer, expected_contexts: list[str] | None = None) -> dict[str, Any]:
        if not self.available:
            return {"ragas_available": False, "message": "Install ragas to enable Ragas evaluation."}
        return {"ragas_available": True, "context_count": len(answer.contexts)}

