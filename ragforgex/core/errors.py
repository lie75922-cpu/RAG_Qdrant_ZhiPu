"""Project-specific exceptions."""


class RAGForgeXError(Exception):
    """Base exception for RAGForgeX."""


class OptionalDependencyError(RAGForgeXError):
    """Raised when an optional integration dependency is missing."""


def missing_dependency(package: str, extra: str | None = None) -> OptionalDependencyError:
    hint = f" Install with `pip install {package}`."
    if extra:
        hint = f" Install with `pip install -e '.[{extra}]'` or `pip install {package}`."
    return OptionalDependencyError(f"Optional dependency `{package}` is required.{hint}")

