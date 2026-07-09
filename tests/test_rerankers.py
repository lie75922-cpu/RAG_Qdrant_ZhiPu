from ragforgex.core.schema import SearchResult
from ragforgex.rerankers.bge_reranker import BGEReranker


def test_bge_reranker_fallback_prioritizes_overlap():
    reranker = BGEReranker(model="missing-local-model")
    results = [
        SearchResult("unrelated context", 0.1),
        SearchResult("rag retrieval context", 0.1),
    ]

    reranked = reranker.rerank("rag retrieval", results, top_k=2)

    assert reranked[0].text == "rag retrieval context"
