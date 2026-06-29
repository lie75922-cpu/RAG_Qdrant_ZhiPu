from ragforgex.chunkers.recursive_chunker import RecursiveChunker
from ragforgex.chunkers.token_chunker import TokenChunker
from ragforgex.core.registry import CHUNKERS

CHUNKERS.register("recursive", RecursiveChunker)
CHUNKERS.register("token", TokenChunker)

