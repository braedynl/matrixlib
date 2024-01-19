from __future__ import annotations

__all__ = [
    "MatrixAccessor",
    "ValueAccessor",
    "RowVectorAccessor",
    "ColVectorAccessor",
]

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Generic, Literal, TypeVar

from typing_extensions import override

from .abc import BaseVectorAccessor

T_co = TypeVar("T_co", covariant=True)


class ArrayedAccessor(BaseVectorAccessor[T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __hash__(self) -> int:
        return hash((self.array, self.shape))

    @override
    def __len__(self) -> int:
        return len(self.array)

    @override
    def __iter__(self) -> Iterator[T_co]:
        return iter(self.array)

    @override
    def __reversed__(self) -> Iterator[T_co]:
        return reversed(self.array)

    @override
    def __contains__(self, value: object) -> bool:
        return value in self.array

    @property
    @abstractmethod
    def array(self) -> tuple[T_co, ...]:
        raise NotImplementedError

    @override
    def collect(self) -> tuple[T_co, ...]:
        return self.array

    @override
    def vector_access(self, index: int) -> T_co:
        return self.array[index]


class MatrixAccessor(ArrayedAccessor[T_co], Generic[T_co]):

    __slots__ = ("array", "shape")
    array: tuple[T_co, ...]
    shape: tuple[int, int]

    def __init__(self, array: tuple[T_co, ...], shape: tuple[int, int]) -> None:
        self.array = array  # pyright: ignore[reportIncompatibleMethodOverride]
        self.shape = shape  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"MatrixAccessor(array={self.array!r}, shape={self.shape!r})"

    @property
    @override
    def row_count(self) -> int:
        return self.shape[0]

    @property
    @override
    def col_count(self) -> int:
        return self.shape[1]


class RowVectorAccessor(ArrayedAccessor[T_co], Generic[T_co]):

    __slots__ = ("array")
    array: tuple[T_co, ...]

    def __init__(self, array: tuple[T_co, ...]) -> None:
        self.array = array  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"RowVectorAccessor(array={self.array!r})"

    @property
    @override
    def row_count(self) -> Literal[1]:
        return 1

    @property
    @override
    def col_count(self) -> int:
        return len(self.array)


class ColVectorAccessor(ArrayedAccessor[T_co], Generic[T_co]):

    __slots__ = ("array")
    array: tuple[T_co, ...]

    def __init__(self, array: tuple[T_co, ...]) -> None:
        self.array = array  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"ColVectorAccessor(array={self.array!r})"

    @property
    @override
    def row_count(self) -> int:
        return len(self.array)

    @property
    @override
    def col_count(self) -> Literal[1]:
        return 1


class ValueAccessor(ArrayedAccessor[T_co], Generic[T_co]):

    __slots__ = ("value")
    value: T_co

    def __init__(self, array: tuple[T_co]) -> None:
        self.value = array[0]

    def __repr__(self) -> str:
        return f"ValueAccessor(array={self.array!r})"

    @override
    def __len__(self) -> Literal[1]:
        return 1

    @override
    def __iter__(self) -> Iterator[T_co]:
        yield self.value

    @override
    def __reversed__(self) -> Iterator[T_co]:
        yield self.value

    @override
    def __contains__(self, value: object) -> bool:
        return value is self.value or value == self.value

    @property
    @override
    def array(self) -> tuple[T_co]:
        return (self.value,)

    @property
    @override
    def row_count(self) -> Literal[1]:
        return 1

    @property
    @override
    def col_count(self) -> Literal[1]:
        return 1
