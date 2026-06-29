"""Parser interface."""

from __future__ import annotations

from pathlib import Path

from ragforgex.core.schema import Document


class BaseParser:
    def parse(self, path: str | Path) -> list[Document]:
        raise NotImplementedError

