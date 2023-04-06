from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Any, Generic, Literal, TypeVar, final

from typing_extensions import override

from .abc import InnerMesh

__all__ = [
    "Grid",
    "Box",
    "Row",
    "Col",
    "Nil",
    "NilRow",
    "NilCol",
]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class ArrayedMesh(InnerMesh[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

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
    def inner_get(self, index: int) -> T_co:
        return self.array[index]


@final
class Grid(ArrayedMesh[M_co, N_co, T_co]):

    __slots__ = ("array", "shape")

    array: tuple[T_co, ... ]
    shape: tuple[M_co, N_co]

    def __init__(self, array: tuple[T_co, ...], shape: tuple[M_co, N_co]) -> None:
        self.array = array
        self.shape = shape

    @override
    def __repr__(self) -> str:
        return f"Grid(array={self.array!r}, shape={self.shape!r})"

    @property
    @override
    def nrows(self) -> M_co:
        return self.shape[0]

    @property
    @override
    def ncols(self) -> N_co:
        return self.shape[1]


@final
class Box(ArrayedMesh[Literal[1], Literal[1], T_co], Generic[T_co]):

    __slots__ = ("value")

    value: T_co

    def __init__(self, array: tuple[T_co]) -> None:
        self.value = array[0]

    def __repr__(self) -> str:
        return f"Box(array={self.array!r})"

    @property
    @override
    def array(self) -> tuple[T_co]:
        return (self.value,)

    @property
    @override
    def nrows(self) -> Literal[1]:
        return 1

    @property
    @override
    def ncols(self) -> Literal[1]:
        return 1


@final
class Row(ArrayedMesh[Literal[1], int, T_co], Generic[T_co]):

    __slots__ = ("array")

    array: tuple[T_co, ...]

    def __init__(self, array: tuple[T_co, ...]) -> None:
        self.array = array

    @override
    def __repr__(self) -> str:
        return f"Row(array={self.array!r})"

    @property
    @override
    def nrows(self) -> Literal[1]:
        return 1

    @property
    @override
    def ncols(self) -> int:
        return len(self.array)


@final
class Col(ArrayedMesh[int, Literal[1], T_co], Generic[T_co]):

    __slots__ = ("array")

    array: tuple[T_co, ...]

    def __init__(self, array: tuple[T_co, ...]) -> None:
        self.array = array

    @override
    def __repr__(self) -> str:
        return f"Col(array={self.array!r})"

    @property
    @override
    def nrows(self) -> int:
        return len(self.array)

    @property
    @override
    def ncols(self) -> Literal[1]:
        return 1


class NilMesh(InnerMesh[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    @override
    def __len__(self) -> Literal[0]:
        return 0

    @override
    def __iter__(self) -> Iterator[T_co]:
        return iter(())

    @override
    def __reversed__(self) -> Iterator[T_co]:
        return iter(())

    @override
    def __contains__(self, value: object) -> bool:
        return False

    @property
    def array(self) -> tuple[T_co, ...]:
        return ()

    @override
    def inner_get(self, index: int) -> T_co:
        raise IndexError("index out of range")  # Should be unreachable


@final
class Nil(NilMesh[Literal[0], Literal[0], T_co], Generic[T_co]):

    __slots__ = ()

    @override
    def __repr__(self) -> str:
        return "Nil()"

    @property
    @override
    def nrows(self) -> Literal[0]:
        return 0

    @property
    @override
    def ncols(self) -> Literal[0]:
        return 0


@final
class NilRow(NilMesh[Literal[0], N_co, T_co], Generic[N_co, T_co]):

    __slots__ = ("ncols")

    ncols: N_co

    def __init__(self, shape: tuple[Any, N_co]) -> None:
        self.ncols = shape[1]

    @override
    def __repr__(self) -> str:
        return f"NilRow(shape={self.shape!r})"

    @property
    @override
    def nrows(self) -> Literal[0]:
        return 0


@final
class NilCol(NilMesh[M_co, Literal[0], T_co], Generic[M_co, T_co]):

    __slots__ = ("nrows")

    nrows: M_co

    def __init__(self, shape: tuple[M_co, Any]) -> None:
        self.nrows = shape[0]

    @override
    def __repr__(self) -> str:
        return f"NilCol(shape={self.shape!r})"

    @property
    @override
    def ncols(self) -> Literal[0]:
        return 0
