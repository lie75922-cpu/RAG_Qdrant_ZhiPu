"""DeepEval evaluator adapter."""

from __future__ import annotations

from typing import Any

from ragforgex.core.schema import RAGAnswer
from ragforgex.evaluators.base import BaseEvaluator


class DeepEvalEvaluator(BaseEvaluator):
    def __init__(self, **_: Any) -> None:
        try:
            import deepeval  # noqa: F401

            self.available = True
        except Exception:
            self.available = False

    def evaluate(self, answer: RAGAnswer, expected_contexts: list[str] | None = None) -> dict[str, Any]:
        if not self.available:
            return {
                "deepeval_available": False,
                "message": "Install deepeval to enable DeepEval metrics.",
            }
        return {
            "deepeval_available": True,
            "context_count": len(answer.contexts),
            "answer_length": len(answer.answer),
        }
