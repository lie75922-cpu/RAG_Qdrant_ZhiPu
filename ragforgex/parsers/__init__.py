from ragforgex.core.registry import PARSERS
from ragforgex.parsers.docling_parser import DoclingParser
from ragforgex.parsers.llamaindex_parser import LlamaIndexParser

PARSERS.register("docling", DoclingParser)
PARSERS.register("llamaindex", LlamaIndexParser)
PARSERS.register("llama_index", LlamaIndexParser)

