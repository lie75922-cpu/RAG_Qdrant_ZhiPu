from ragforgex.chunkers.recursive_chunker import RecursiveChunker
from ragforgex.chunkers.parent_child_chunker import ParentChildChunker
from ragforgex.chunkers.semantic_chunker import SemanticChunker
from ragforgex.chunkers.token_chunker import TokenChunker
from ragforgex.core.schema import Document


def test_recursive_chunker_splits_text():
    chunks = RecursiveChunker(chunk_size=10, chunk_overlap=2).split([Document(text="abcdefghijklmnopqrstuvwxyz")])

    assert len(chunks) > 1
    assert chunks[0].metadata["chunk_start"] == 0


def test_token_chunker_splits_tokens():
    chunks = TokenChunker(chunk_size=3, chunk_overlap=1).split([Document(text="one two three four five")])

    assert [chunk.text for chunk in chunks][:2] == ["one two three", "three four five"]


def test_semantic_chunker_groups_sentences():
    text = "RAG uses retrieval. Retrieval adds context. Bananas are fruit."
    chunks = SemanticChunker(max_sentences=2, similarity_threshold=0.0).split([Document(text=text)])

    assert len(chunks) == 2
    assert chunks[0].metadata["chunker"] == "semantic"


def test_parent_child_chunker_preserves_parent_context():
    chunks = ParentChildChunker(parent_size=30, child_size=15, child_overlap=5).split(
        [Document(text="abcdefghijklmnopqrstuvwxyz0123456789")]
    )

    assert chunks
    assert "parent_text" in chunks[0].metadata
    assert chunks[0].metadata["child_id"].startswith("parent-")

