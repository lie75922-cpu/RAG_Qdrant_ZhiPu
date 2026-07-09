from ragforgex.rerankers.bge_reranker import BGEReranker
from ragforgex.core.registry import RERANKERS
from ragforgex.rerankers.base import NoOpReranker

RERANKERS.register("bge", BGEReranker)
RERANKERS.register("none", NoOpReranker)
RERANKERS.register("noop", NoOpReranker)

