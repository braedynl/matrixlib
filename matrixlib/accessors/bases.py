from __future__ import annotations

__all__ = [
    "MatrixAccessor",
    "RowVectorAccessor",
    "ColVectorAccessor",
    "ValueAccessor",
    "ColCountAccessor",
    "RowCountAccessor",
    "NullAccessor",
    "NULLARY_ACCESSOR_1x0",
    "NULLARY_ACCESSOR_0x1",
    "NULLARY_ACCESSOR_0x0",
]

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Any, Final, Generic, Literal, TypeVar, final

from typing_extensions import override

from .abstracts import AbstractAccessor, AbstractVectorAccessor

T_co = TypeVar("T_co", covariant=True)


class ArrayedAccessor(AbstractVectorAccessor[T_co], metaclass=ABCMeta):

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
    def materialize(self) -> tuple[T_co, ...]:
        return self.array

    @override
    def vector_access(self, index: int) -> T_co:
        return self.array[index]


class NullaryAccessor(AbstractAccessor[T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __hash__(self) -> int:
        return hash(self.shape)

    @override
    def __len__(self) -> Literal[0]:
        return 0

    @override
    def __iter__(self) -> Iterator[T_co]:
        return
        yield

    @override
    def __reversed__(self) -> Iterator[T_co]:
        return
        yield

    @override
    def __contains__(self, value: object) -> Literal[False]:
        return False

    @override
    def materialize(self) -> tuple[()]:
        return ()

    @override
    def vector_access(self, index: int) -> T_co:
        raise IndexError

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        raise IndexError


@final
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


@final
class RowVectorAccessor(ArrayedAccessor[T_co], Generic[T_co]):

    __slots__ = ("array")
    array: tuple[T_co, ...]
    row_count: Final[Literal[1]] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, array: tuple[T_co, ...]) -> None:
        self.array = array  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"RowVectorAccessor(array={self.array!r})"

    @property
    @override
    def shape(self) -> tuple[Literal[1], int]:
        return (1, len(self.array))

    @property
    @override
    def col_count(self) -> int:
        return len(self.array)


@final
class ColVectorAccessor(ArrayedAccessor[T_co], Generic[T_co]):

    __slots__ = ("array")
    array: tuple[T_co, ...]
    col_count: Final[Literal[1]] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, array: tuple[T_co, ...]) -> None:
        self.array = array  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"ColVectorAccessor(array={self.array!r})"

    @property
    @override
    def shape(self) -> tuple[int, Literal[1]]:
        return (len(self.array), 1)

    @property
    @override
    def row_count(self) -> int:
        return len(self.array)


@final
class ValueAccessor(AbstractAccessor[T_co], Generic[T_co]):

    __slots__ = ("value")
    value: T_co
    shape: Final[tuple[Literal[1], Literal[1]]] = (1, 1)  # pyright: ignore[reportIncompatibleMethodOverride]
    row_count: Final[Literal[1]] = 1  # pyright: ignore[reportIncompatibleMethodOverride]
    col_count: Final[Literal[1]] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, value: T_co) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"ValueAccessor(value={self.value!r})"

    def __hash__(self) -> int:
        return hash(self.value)

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

    @override
    def materialize(self) -> tuple[T_co]:
        return (self.value,)

    @override
    def vector_access(self, index: int) -> T_co:
        assert index == 0
        return self.value

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        assert row_index == 0
        assert col_index == 0
        return self.value


@final
class ColCountAccessor(NullaryAccessor[T_co], Generic[T_co]):

    __slots__ = ("col_count")
    col_count: int
    row_count: Final[Literal[0]] = 0  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, col_count: int) -> None:
        self.col_count = col_count  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"ColCountAccessor(col_count={self.col_count!r})"

    @property
    @override
    def shape(self) -> tuple[Literal[0], int]:
        return (0, self.col_count)


@final
class RowCountAccessor(NullaryAccessor[T_co], Generic[T_co]):

    __slots__ = ("row_count")
    row_count: int
    col_count: Final[Literal[0]] = 0  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, row_count: int) -> None:
        self.row_count = row_count  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"RowCountAccessor(row_count={self.row_count!r})"

    @property
    @override
    def shape(self) -> tuple[int, Literal[0]]:
        return (self.row_count, 0)


@final
class NullAccessor(NullaryAccessor[T_co], Generic[T_co]):

    __slots__ = ()
    shape: Final[tuple[Literal[0], Literal[0]]] = (0, 0)  # pyright: ignore[reportIncompatibleMethodOverride]
    row_count: Final[Literal[0]] = 0  # pyright: ignore[reportIncompatibleMethodOverride]
    col_count: Final[Literal[0]] = 0  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return "NullAccessor()"

    @override
    def __hash__(self) -> Literal[0]:
        return 0


NULLARY_ACCESSOR_1x0: Final[RowCountAccessor[Any]] = RowCountAccessor(1)
NULLARY_ACCESSOR_0x1: Final[ColCountAccessor[Any]] = ColCountAccessor(1)
NULLARY_ACCESSOR_0x0: Final[NullAccessor[Any]] = NullAccessor()
