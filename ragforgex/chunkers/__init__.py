from ragforgex.chunkers.recursive_chunker import RecursiveChunker
from ragforgex.chunkers.parent_child_chunker import ParentChildChunker
from ragforgex.chunkers.semantic_chunker import SemanticChunker
from ragforgex.chunkers.token_chunker import TokenChunker
from ragforgex.core.registry import CHUNKERS

CHUNKERS.register("recursive", RecursiveChunker)
CHUNKERS.register("token", TokenChunker)
CHUNKERS.register("semantic", SemanticChunker)
CHUNKERS.register("parent_child", ParentChildChunker)

