from __future__ import annotations

__all__ = [
    "DefaultAccessor",
    "MutableDefaultAccessor",
]

import copy
from collections.abc import Iterable, Iterator
from typing import Any, Self, override

from .abstracts import AbstractMutableVectorAccessor, AbstractVectorAccessor


class DefaultAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractVectorAccessor[M, N, T]):

    __slots__ = (
        "array",
        "shape",  # pyright: ignore[reportIncompatibleMethodOverride]
    )
    array: tuple[T, ...]
    shape: tuple[M, N]

    def __new__(cls, array: Iterable[T], shape: tuple[M, N]) -> Self:
        self = super(DefaultAccessor, cls).__new__(cls)
        self.array = tuple(array)
        self.shape = shape
        return self

    def __hash__(self) -> int:
        return hash((self.array, self.shape))

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        return self

    __copy__ = __deepcopy__

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (self.__class__, (self.array, self.shape))

    @override
    def __len__(self) -> int:
        return len(self.array)

    @override
    def __iter__(self) -> Iterator[T]:
        return iter(self.array)

    @override
    def __reversed__(self) -> Iterator[T]:
        return reversed(self.array)

    @property
    @override
    def row_count(self) -> M:
        return self.shape[0]

    @property
    @override
    def col_count(self) -> N:
        return self.shape[1]

    @override
    def vector_access(self, index: int) -> T:
        return self.array[index]


class MutableDefaultAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractMutableVectorAccessor[M, N, T]):

    __slots__ = (
        "array",
        "shape",
    )
    array: list[T]
    shape: tuple[M, N]

    def __init__(self, array: Iterable[T], shape: tuple[M, N]) -> None:
        self.array = list(array)
        self.shape = shape  # pyright: ignore[reportIncompatibleMethodOverride]

    # No hash

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        result = object.__new__(self.__class__)
        result.array = copy.deepcopy(self.array, memo)
        result.shape = self.shape
        return result

    def __copy__(self) -> Self:
        result = object.__new__(self.__class__)
        result.array = copy.copy(self.array)
        result.shape = self.shape
        return result

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (self.__class__, (self.array, self.shape))

    @override
    def __len__(self) -> int:
        return len(self.array)

    @override
    def __iter__(self) -> Iterator[T]:
        return iter(self.array)

    @override
    def __reversed__(self) -> Iterator[T]:
        return reversed(self.array)

    @property
    @override
    def row_count(self) -> M:
        return self.shape[0]

    @property
    @override
    def col_count(self) -> N:
        return self.shape[1]

    @override
    def vector_access(self, index: int) -> T:
        return self.array[index]

    @override
    def vector_modify(self, index: int, value: T) -> None:
        self.array[index] = value
