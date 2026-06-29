"""Evaluator interface."""

from typing import Any

from ragforgex.core.schema import RAGAnswer


class BaseEvaluator:
    def evaluate(self, answer: RAGAnswer, expected_contexts: list[str] | None = None) -> dict[str, Any]:
        raise NotImplementedError

