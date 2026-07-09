from ragforgex.core.registry import EVALUATORS
from ragforgex.evaluators.deepeval_evaluator import DeepEvalEvaluator
from ragforgex.evaluators.ragas_evaluator import RagasEvaluator
from ragforgex.evaluators.retrieval_metrics import RetrievalMetricsEvaluator

EVALUATORS.register("deepeval", DeepEvalEvaluator)
EVALUATORS.register("retrieval_metrics", RetrievalMetricsEvaluator)
EVALUATORS.register("ragas", RagasEvaluator)

