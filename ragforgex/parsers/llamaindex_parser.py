"""LlamaIndex parser adapter with a text fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragforgex.core.schema import Document
from ragforgex.parsers.base import BaseParser


class LlamaIndexParser(BaseParser):
    def __init__(self, **_: Any) -> None:
        self._reader_cls = None
        try:
            from llama_index.core import SimpleDirectoryReader

            self._reader_cls = SimpleDirectoryReader
        except Exception:
            self._reader_cls = None

    def parse(self, path: str | Path) -> list[Document]:
        source = Path(path)
        if self._reader_cls is not None and source.is_dir():
            docs = self._reader_cls(input_dir=str(source)).load_data()
            return [
                Document(text=doc.get_content(), metadata={"source": str(source), "parser": "llamaindex"})
                for doc in docs
            ]
        if source.is_dir():
            return [
                Document(text=file.read_text(encoding="utf-8"), metadata={"source": str(file), "parser": "llamaindex"})
                for file in sorted(source.rglob("*.txt"))
            ]
        return [Document(text=source.read_text(encoding="utf-8"), metadata={"source": str(source), "parser": "llamaindex"})]

