from ragforgex.core.schema import Chunk, Document, RAGAnswer, SearchResult


def test_schema_defaults():
    document = Document(text="hello")
    chunk = Chunk(text=document.text)
    result = SearchResult(text=chunk.text, score=1.0)
    answer = RAGAnswer(question="q", answer="a", contexts=[result])

    assert document.metadata == {}
    assert answer.contexts[0].score == 1.0

