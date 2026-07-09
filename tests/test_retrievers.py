from ragforgex.core.schema import Chunk
from ragforgex.embeddings.sentence_transformers_embedding import SentenceTransformersEmbedding
from ragforgex.retrievers.bm25_retriever import BM25Retriever
from ragforgex.retrievers.query_rewrite import QueryRewriteRetriever
from ragforgex.retrievers.rrf_fusion import rrf_fusion
from ragforgex.core.schema import SearchResult
from ragforgex.stores.faiss_store import FAISSStore


def test_bm25_retriever_finds_lexical_match():
    retriever = BM25Retriever([Chunk("dense vector search"), Chunk("graph database")])

    results = retriever.retrieve("vector", top_k=1)

    assert results[0].text == "dense vector search"


def test_rrf_fusion_merges_results():
    fused = rrf_fusion([[SearchResult("a", 1.0)], [SearchResult("b", 1.0), SearchResult("a", 0.5)]])

    assert fused[0].text == "a"


def test_query_rewrite_retriever_runs_multiple_queries():
    chunks = [Chunk("RAGForgeX compares retrieval pipelines")]
    embedding = SentenceTransformersEmbedding(dimension=16)
    vectors = embedding.embed_texts([chunk.text for chunk in chunks])
    store = FAISSStore(dimension=16)
    store.add(chunks, vectors)
    retriever = QueryRewriteRetriever(embedding=embedding, store=store)

    assert retriever.retrieve("RAGForgeX", top_k=1)

