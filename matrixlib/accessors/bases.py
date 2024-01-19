from __future__ import annotations

__all__ = [
    "BaseMatrixAccessor",
    "BaseRowVectorAccessor",
    "BaseColVectorAccessor",
    "BaseValueAccessor",
    "BaseEmptyAccessor",
]

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Generic, Literal, TypeVar, final

from typing_extensions import override

from .abstracts import AbstractAccessor, AbstractVectorAccessor

T_co = TypeVar("T_co", covariant=True)


class BaseArrayedAccessor(AbstractVectorAccessor[T_co], metaclass=ABCMeta):

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


@final
class BaseMatrixAccessor(BaseArrayedAccessor[T_co], Generic[T_co]):

    __slots__ = ("array", "shape")
    array: tuple[T_co, ...]
    shape: tuple[int, int]

    def __init__(self, array: tuple[T_co, ...], shape: tuple[int, int]) -> None:
        self.array = array  # pyright: ignore[reportIncompatibleMethodOverride]
        self.shape = shape  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"BaseMatrixAccessor(array={self.array!r}, shape={self.shape!r})"

    @property
    @override
    def row_count(self) -> int:
        return self.shape[0]

    @property
    @override
    def col_count(self) -> int:
        return self.shape[1]


@final
class BaseRowVectorAccessor(BaseArrayedAccessor[T_co], Generic[T_co]):

    __slots__ = ("array")
    array: tuple[T_co, ...]

    def __init__(self, array: tuple[T_co, ...]) -> None:
        self.array = array  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"BaseRowVectorAccessor(array={self.array!r})"

    @property
    @override
    def shape(self) -> tuple[Literal[1], int]:
        return (1, len(self.array))

    @property
    @override
    def row_count(self) -> Literal[1]:
        return 1

    @property
    @override
    def col_count(self) -> int:
        return len(self.array)


@final
class BaseColVectorAccessor(BaseArrayedAccessor[T_co], Generic[T_co]):

    __slots__ = ("array")
    array: tuple[T_co, ...]

    def __init__(self, array: tuple[T_co, ...]) -> None:
        self.array = array  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"BaseColVectorAccessor(array={self.array!r})"

    @property
    @override
    def shape(self) -> tuple[int, Literal[1]]:
        return (len(self.array), 1)

    @property
    @override
    def row_count(self) -> int:
        return len(self.array)

    @property
    @override
    def col_count(self) -> Literal[1]:
        return 1


@final
class BaseValueAccessor(AbstractAccessor[T_co], Generic[T_co]):

    __slots__ = ("value")
    value: T_co

    def __init__(self, value: T_co) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"BaseValueAccessor(value={self.value!r})"

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

    @property
    @override
    def shape(self) -> tuple[Literal[1], Literal[1]]:
        return (1, 1)

    @property
    @override
    def row_count(self) -> Literal[1]:
        return 1

    @property
    @override
    def col_count(self) -> Literal[1]:
        return 1

    @override
    def collect(self) -> tuple[T_co, ...]:
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
class BaseEmptyAccessor(AbstractAccessor[T_co], Generic[T_co]):

    __slots__ = ()

    def __repr__(self) -> str:
        return "BaseEmptyAccessor()"

    def __hash__(self) -> Literal[0]:
        return 0

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

    @property
    @override
    def shape(self) -> tuple[Literal[0], Literal[0]]:
        return (0, 0)

    @property
    @override
    def row_count(self) -> Literal[0]:
        return 0

    @property
    @override
    def col_count(self) -> Literal[0]:
        return 0

    @override
    def collect(self) -> tuple[()]:
        return ()

    @override
    def vector_access(self, index: int) -> T_co:
        raise IndexError("empty accessor")

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        raise IndexError("empty accessor")
