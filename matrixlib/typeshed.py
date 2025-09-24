from __future__ import annotations

__all__ = [
    "SupportsIter",
    "SupportsReversed",
    "SupportsIterAndReversed",
]

from collections.abc import Iterator
from typing import Protocol


class SupportsIter[T](Protocol):
    def __iter__(self) -> Iterator[T]: ...

class SupportsReversed[T](Protocol):
    def __reversed__(self) -> Iterator[T]: ...

class SupportsIterAndReversed[T](SupportsIter[T], SupportsReversed[T], Protocol):
    ...
