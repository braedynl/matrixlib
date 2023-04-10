from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Any, Generic, Literal, TypeVar

from .abc import VectorSieve

__all__ = [
    "Grid",
    "Box",
    "Row",
    "Col",
    "Empty",
    "EmptyRow",
    "EmptyCol",
]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class ArrayedSieve(VectorSieve[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __len__(self) -> int:
        return len(self.array)

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.array)

    def __reversed__(self) -> Iterator[T_co]:
        return reversed(self.array)

    def __contains__(self, value: object) -> bool:
        return value in self.array

    @property
    @abstractmethod
    def array(self) -> tuple[T_co, ...]:
        raise NotImplementedError

    def sieve(self) -> tuple[T_co, ...]:
        return self.array

    def vector_sieve(self, index: int) -> T_co:
        return self.array[index]


class EmptyArrayedSieve(ArrayedSieve[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    array: tuple[T_co, ...] = ()


class Grid(ArrayedSieve[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("array", "shape")

    array: tuple[T_co, ... ]
    shape: tuple[M_co, N_co]

    def __init__(self, array: tuple[T_co, ...], shape: tuple[M_co, N_co]) -> None:
        self.array = array
        self.shape = shape

    def __repr__(self) -> str:
        return f"Grid(array={self.array!r}, shape={self.shape!r})"

    @property
    def nrows(self) -> M_co:
        return self.shape[0]

    @property
    def ncols(self) -> N_co:
        return self.shape[1]


class Box(ArrayedSieve[Literal[1], Literal[1], T_co], Generic[T_co]):

    __slots__ = ("value")

    value: T_co
    nrows: Literal[1] = 1
    ncols: Literal[1] = 1

    def __init__(self, array: tuple[T_co]) -> None:
        self.value = array[0]

    def __repr__(self) -> str:
        return f"Box(array={self.array!r})"

    @property
    def array(self) -> tuple[T_co]:
        return (self.value,)


class Row(ArrayedSieve[Literal[1], int, T_co], Generic[T_co]):

    __slots__ = ("array")

    array: tuple[T_co, ...]
    nrows: Literal[1] = 1

    def __init__(self, array: tuple[T_co, ...]) -> None:
        self.array = array

    def __repr__(self) -> str:
        return f"Row(array={self.array!r})"

    @property
    def ncols(self) -> int:
        return len(self.array)


class Col(ArrayedSieve[int, Literal[1], T_co], Generic[T_co]):

    __slots__ = ("array")

    array: tuple[T_co, ...]
    ncols: Literal[1] = 1

    def __init__(self, array: tuple[T_co, ...]) -> None:
        self.array = array

    def __repr__(self) -> str:
        return f"Col(array={self.array!r})"

    @property
    def nrows(self) -> int:
        return len(self.array)


class Empty(EmptyArrayedSieve[Literal[0], Literal[0], T_co], Generic[T_co]):

    __slots__ = ()

    nrows: Literal[0] = 0
    ncols: Literal[0] = 0

    def __repr__(self) -> str:
        return "Empty()"


class EmptyRow(EmptyArrayedSieve[Literal[0], N_co, T_co], Generic[N_co, T_co]):

    __slots__ = ("ncols")

    ncols: N_co
    nrows: Literal[0] = 0

    def __init__(self, shape: tuple[Any, N_co]) -> None:
        self.ncols = shape[1]

    def __repr__(self) -> str:
        return f"EmptyRow(shape={self.shape!r})"


class EmptyCol(EmptyArrayedSieve[M_co, Literal[0], T_co], Generic[M_co, T_co]):

    __slots__ = ("nrows")

    nrows: M_co
    ncols: Literal[0] = 0

    def __init__(self, shape: tuple[M_co, Any]) -> None:
        self.nrows = shape[0]

    def __repr__(self) -> str:
        return f"EmptyCol(shape={self.shape!r})"
