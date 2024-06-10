from __future__ import annotations

__all__ = [
    "MatrixAccessor",
    "RowVectorAccessor",
    "ColVectorAccessor",
    "ValueAccessor",
    "ZeroRowAccessor",
    "ZeroColAccessor",
    "ZeroAccessor",
    "NULLARY_ACCESSOR_1x0",
    "NULLARY_ACCESSOR_0x1",
    "NULLARY_ACCESSOR_0x0",
]

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Any, Final, Generic, Literal, TypeVar, cast, final

from typing_extensions import Never, override

from .abstracts import AbstractAccessor, AbstractVectorAccessor

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)

T_co = TypeVar("T_co", covariant=True)


class AbstractArrayedAccessor(AbstractVectorAccessor[M_co, N_co, T_co], metaclass=ABCMeta):
    """Abstract for accessors that hold their values in memory as a built-in
    ``tuple``.
    """

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


class AbstractNullaryAccessor(AbstractAccessor[M_co, N_co, T_co], metaclass=ABCMeta):
    """Abstract for accessors that have a size of 0.

    Sub-classes of ``AbstractNullaryAccessor`` must have at least one 0 dimension.
    """

    __slots__ = ()

    def __hash__(self) -> int:
        return hash(self.shape)

    @override
    @final
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
    def vector_access(self, index: int) -> Never:
        raise IndexError

    @override
    def matrix_access(self, row_index: int, col_index: int) -> Never:
        raise IndexError


@final
class MatrixAccessor(AbstractArrayedAccessor[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):
    """Concrete base accessor for matrices of shape M by N, where both M and N
    are greater than 1.
    """

    __slots__ = ("array", "shape")
    array: tuple[T_co, ...]
    shape: tuple[M_co, N_co]

    def __init__(self, array: tuple[T_co, ...], shape: tuple[M_co, N_co]) -> None:
        assert len(array) == shape[0] * shape[1]
        self.array = array  # pyright: ignore[reportIncompatibleMethodOverride]
        self.shape = shape  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"MatrixAccessor(array={self.array!r}, shape={self.shape!r})"

    @property
    @override
    def row_count(self) -> M_co:
        return self.shape[0]

    @property
    @override
    def col_count(self) -> N_co:
        return self.shape[1]


@final
class RowVectorAccessor(AbstractArrayedAccessor[Literal[1], N_co, T_co], Generic[N_co, T_co]):
    """Concrete base accessor for matrices of shape 1 by N."""

    __slots__ = ("array")
    array: tuple[T_co, ...]
    row_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, array: tuple[T_co, ...]) -> None:
        self.array = array  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"RowVectorAccessor(array={self.array!r})"

    @property
    @override
    def col_count(self) -> N_co:
        return cast(N_co, len(self.array))


@final
class ColVectorAccessor(AbstractArrayedAccessor[M_co, Literal[1], T_co], Generic[M_co, T_co]):
    """Concrete base accessor for matrices of shape M by 1."""

    __slots__ = ("array")
    array: tuple[T_co, ...]
    col_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, array: tuple[T_co, ...]) -> None:
        self.array = array  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"ColVectorAccessor(array={self.array!r})"

    @property
    @override
    def row_count(self) -> M_co:
        return cast(M_co, len(self.array))


@final
class ValueAccessor(AbstractAccessor[Literal[1], Literal[1], T_co], Generic[T_co]):
    """Concrete base accessor for matrices of shape 1 by 1."""

    __slots__ = ("value")
    value: T_co
    shape: tuple[Literal[1], Literal[1]] = (1, 1)  # pyright: ignore[reportIncompatibleMethodOverride]
    row_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]
    col_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

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
class ZeroRowAccessor(AbstractNullaryAccessor[Literal[0], N_co, T_co], Generic[N_co, T_co]):
    """Concrete base accessor for matrices of shape 0 by N."""

    __slots__ = ("col_count")
    col_count: N_co
    row_count: Literal[0] = 0  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, col_count: N_co) -> None:
        self.col_count = col_count  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"ZeroRowAccessor(col_count={self.col_count!r})"


@final
class ZeroColAccessor(AbstractNullaryAccessor[M_co, Literal[0], T_co], Generic[M_co, T_co]):
    """Concrete base accessor for matrices of shape M by 0."""

    __slots__ = ("row_count")
    row_count: M_co
    col_count: Literal[0] = 0  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, row_count: M_co) -> None:
        self.row_count = row_count  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"ZeroColAccessor(row_count={self.row_count!r})"


@final
class ZeroAccessor(AbstractNullaryAccessor[Literal[0], Literal[0], T_co], Generic[T_co]):
    """Concrete base accessor for matrices of shape 0 by 0."""

    __slots__ = ()
    shape: tuple[Literal[0], Literal[0]] = (0, 0)  # pyright: ignore[reportIncompatibleMethodOverride]
    row_count: Literal[0] = 0  # pyright: ignore[reportIncompatibleMethodOverride]
    col_count: Literal[0] = 0  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return "ZeroAccessor()"

    @override
    def __hash__(self) -> Literal[0]:
        return 0


NULLARY_ACCESSOR_1x0: Final[ZeroColAccessor[Literal[1], Any]] = ZeroColAccessor(1)
NULLARY_ACCESSOR_0x1: Final[ZeroRowAccessor[Literal[1], Any]] = ZeroRowAccessor(1)
NULLARY_ACCESSOR_0x0: Final[ZeroAccessor[Any]] = ZeroAccessor()
