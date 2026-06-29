"""Docling parser adapter with a text fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragforgex.core.schema import Document
from ragforgex.parsers.base import BaseParser


class DoclingParser(BaseParser):
    def __init__(self, **_: Any) -> None:
        self._converter = None
        try:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        except Exception:
            self._converter = None

    def parse(self, path: str | Path) -> list[Document]:
        source = Path(path)
        if self._converter is not None and source.suffix.lower() != ".txt":
            result = self._converter.convert(str(source))
            text = result.document.export_to_markdown()
        else:
            text = source.read_text(encoding="utf-8")
        return [Document(text=text, metadata={"source": str(source), "parser": "docling"})]

