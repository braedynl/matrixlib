from __future__ import annotations

__all__ = [
    "SupportsBool",
    "SupportsDunderLT",
    "SupportsDunderGT",
    "SupportsDunderLE",
    "SupportsDunderGE",
    "Sortable",
]

from typing import Any, Protocol


class SupportsBool(Protocol):
    def __bool__(self) -> bool: ...

class SupportsDunderLT[T](Protocol):
    def __lt__(self, other: T, /) -> SupportsBool: ...

class SupportsDunderGT[T](Protocol):
    def __gt__(self, other: T, /) -> SupportsBool: ...

class SupportsDunderLE[T](Protocol):
    def __le__(self, other: T, /) -> SupportsBool: ...

class SupportsDunderGE[T](Protocol):
    def __ge__(self, other: T, /) -> SupportsBool: ...

type Sortable = SupportsDunderLT[Any] | SupportsDunderGT[Any]
