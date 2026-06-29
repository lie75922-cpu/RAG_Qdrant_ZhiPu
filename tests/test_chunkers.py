from ragforgex.chunkers.recursive_chunker import RecursiveChunker
from ragforgex.chunkers.token_chunker import TokenChunker
from ragforgex.core.schema import Document


def test_recursive_chunker_splits_text():
    chunks = RecursiveChunker(chunk_size=10, chunk_overlap=2).split([Document(text="abcdefghijklmnopqrstuvwxyz")])

    assert len(chunks) > 1
    assert chunks[0].metadata["chunk_start"] == 0


def test_token_chunker_splits_tokens():
    chunks = TokenChunker(chunk_size=3, chunk_overlap=1).split([Document(text="one two three four five")])

    assert [chunk.text for chunk in chunks][:2] == ["one two three", "three four five"]

