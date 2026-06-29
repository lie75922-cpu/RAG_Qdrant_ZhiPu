"""Small component registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self) -> None:
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., T]) -> Callable[..., T]:
        self._items[name] = factory
        return factory

    def create(self, name: str, **kwargs: Any) -> Any:
        try:
            factory = self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "none"
            raise KeyError(f"Unknown component `{name}`. Available: {available}") from exc
        return factory(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)


PARSERS = Registry()
CHUNKERS = Registry()
EMBEDDINGS = Registry()
STORES = Registry()
RETRIEVERS = Registry()
RERANKERS = Registry()
GENERATORS = Registry()
EVALUATORS = Registry()

