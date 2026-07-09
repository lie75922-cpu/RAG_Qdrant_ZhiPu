"""Parent-child chunking for retrievers that need fine chunks and broad context."""

from __future__ import annotations

from typing import Any

from ragforgex.chunkers.base import BaseChunker
from ragforgex.chunkers.recursive_chunker import RecursiveChunker
from ragforgex.core.schema import Chunk, Document


class ParentChildChunker(BaseChunker):
    def __init__(
        self,
        parent_size: int = 1600,
        child_size: int = 400,
        child_overlap: int = 80,
        **_: Any,
    ) -> None:
        self.parent_splitter = RecursiveChunker(chunk_size=parent_size, chunk_overlap=0)
        self.child_size = child_size
        self.child_overlap = child_overlap

    def split(self, documents: list[Document]) -> list[Chunk]:
        children: list[Chunk] = []
        parent_chunks = self.parent_splitter.split(documents)
        child_splitter = RecursiveChunker(chunk_size=self.child_size, chunk_overlap=self.child_overlap)
        for parent_index, parent in enumerate(parent_chunks):
            child_docs = [Document(text=parent.text, metadata=parent.metadata)]
            for child_index, child in enumerate(child_splitter.split(child_docs)):
                metadata = dict(child.metadata)
                metadata.update(
                    {
                        "chunker": "parent_child",
                        "parent_id": f"parent-{parent_index}",
                        "child_id": f"parent-{parent_index}-child-{child_index}",
                        "parent_text": parent.text,
                    }
                )
                children.append(Chunk(text=child.text, metadata=metadata))
        return children
