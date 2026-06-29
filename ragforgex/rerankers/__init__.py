from ragforgex.core.registry import RERANKERS
from ragforgex.rerankers.base import NoOpReranker

RERANKERS.register("none", NoOpReranker)
RERANKERS.register("noop", NoOpReranker)

