from ragforgex.core.registry import RETRIEVERS
from ragforgex.retrievers.bm25_retriever import BM25Retriever
from ragforgex.retrievers.dense_retriever import DenseRetriever
from ragforgex.retrievers.hybrid_retriever import HybridRetriever
from ragforgex.retrievers.query_rewrite import QueryRewriteRetriever

RETRIEVERS.register("dense", DenseRetriever)
RETRIEVERS.register("bm25", BM25Retriever)
RETRIEVERS.register("hybrid", HybridRetriever)
RETRIEVERS.register("query_rewrite", QueryRewriteRetriever)

