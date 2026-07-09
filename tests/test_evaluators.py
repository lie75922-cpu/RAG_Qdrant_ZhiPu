from ragforgex.core.schema import RAGAnswer, SearchResult
from ragforgex.evaluators.deepeval_evaluator import DeepEvalEvaluator


def test_deepeval_evaluator_reports_availability():
    evaluator = DeepEvalEvaluator()
    answer = RAGAnswer("q", "a", [SearchResult("context", 1.0)])

    report = evaluator.evaluate(answer)

    assert "deepeval_available" in report
