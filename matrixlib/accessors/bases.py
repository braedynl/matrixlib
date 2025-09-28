from __future__ import annotations

__all__ = [
    "MatrixAccessor",
    "RowVectorAccessor",
    "ColVectorAccessor",
    "ValueAccessor",
    "IdentityAccessor",
    "DiagonalAccessor",
    "SparseAccessor",
]

import itertools
from abc import ABCMeta, abstractmethod
from array import array as Array
from collections import Counter
from collections.abc import Iterator, Mapping
from typing import Final, Literal, cast, final, override

from .abstracts import (AbstractAccessor, AbstractMatrixAccessor,
                        AbstractVectorAccessor)

SLL_TYPE_CODE: Final[Literal["q"]] = "q"


class AbstractArrayAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractVectorAccessor[M, N, T], metaclass=ABCMeta):
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
    def __iter__(self) -> Iterator[T]:
        return iter(self.array)

    @override
    def __reversed__(self) -> Iterator[T]:
        return reversed(self.array)

    @override
    def __contains__(self, value: object) -> bool:
        return value in self.array

    @property
    @abstractmethod
    def array(self) -> tuple[T, ...]:
        raise NotImplementedError

    @override
    def to_tuple(self) -> tuple[T, ...]:
        return self.array

    @override
    def vector_access(self, index: int) -> T:
        return self.array[index]


@final
class MatrixAccessor[M: int = int, N: int = int, T: object = object](AbstractArrayAccessor[M, N, T]):
    """Concrete base accessor for matrices of shape M by N, where both M and N
    are greater than 1.
    """

    __slots__ = ("array", "shape")
    array: tuple[T, ...]
    shape: tuple[M, N]

    def __init__(self, array: tuple[T, ...], shape: tuple[M, N]) -> None:
        assert len(array) == shape[0] * shape[1]
        self.array = array  # pyright: ignore[reportIncompatibleMethodOverride]
        self.shape = shape  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"MatrixAccessor(array={self.array!r}, shape={self.shape!r})"

    @property
    @override
    def row_count(self) -> M:
        return self.shape[0]

    @property
    @override
    def col_count(self) -> N:
        return self.shape[1]


@final
class RowVectorAccessor[N: int = int, T: object = object](AbstractArrayAccessor[Literal[1], N, T]):
    """Concrete base accessor for matrices of shape 1 by N."""

    __slots__ = ("array",)
    array: tuple[T, ...]
    row_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, array: tuple[T, ...]) -> None:
        self.array = array  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"RowVectorAccessor(array={self.array!r})"

    @property
    @override
    def col_count(self) -> N:
        return cast(N, len(self.array))


@final
class ColVectorAccessor[M: int = int, T: object = object](AbstractArrayAccessor[M, Literal[1], T]):
    """Concrete base accessor for matrices of shape M by 1."""

    __slots__ = ("array",)
    array: tuple[T, ...]
    col_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, array: tuple[T, ...]) -> None:
        self.array = array  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"ColVectorAccessor(array={self.array!r})"

    @property
    @override
    def row_count(self) -> M:
        return cast(M, len(self.array))


@final
class ValueAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractAccessor[M, N, T]):

    __slots__ = ("value", "shape")
    value: T
    shape: tuple[M, N]

    def __init__(self, value: T, shape: tuple[M, N]) -> None:
        self.value = value
        self.shape = shape  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"ValueAccessor(value={self.value!r}, shape={self.shape!r})"

    def __hash__(self) -> int:
        return hash((self.value, self.shape))

    @override
    def __iter__(self) -> Iterator[T]:
        return itertools.repeat(self.value, times=len(self))

    __reversed__ = override(__iter__)

    @override
    def __contains__(self, value: object) -> bool:
        return value is self.value or value == self.value

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
        return self.value

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T:
        return self.value


@final
class IdentityAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractMatrixAccessor[M, N, T]):

    __slots__ = ("non_zero_value", "zero_value", "shape")
    non_zero_value: T
    zero_value: T
    shape: tuple[M, N]

    def __init__(self, non_zero_value: T, shape: tuple[M, N], *, zero_value: T = 0) -> None:
        self.non_zero_value = non_zero_value
        self.shape = shape  # pyright: ignore[reportIncompatibleMethodOverride]
        self.zero_value = zero_value

    @property
    @override
    def row_count(self) -> M:
        return self.shape[0]

    @property
    @override
    def col_count(self) -> N:
        return self.shape[1]

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T:
        if row_index == col_index:
            return self.non_zero_value
        return self.zero_value


@final
class DiagonalAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
    S: object = object,
](AbstractMatrixAccessor[M, N, T | S]):

    __slots__ = ("diagonal_values", "off_diagonal_value")
    diagonal_values: tuple[T, ...]
    off_diagonal_value: S

    def __init__(self, diagonal_values: tuple[T, ...], *, off_diagonal_value: S = 0) -> None:
        self.diagonal_values = diagonal_values
        self.off_diagonal_value = off_diagonal_value

    @property
    @override
    def row_count(self) -> M:
        return cast(M, len(self.diagonal_values))

    @property
    @override
    def col_count(self) -> N:
        return cast(N, len(self.diagonal_values))

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T | S:
        if row_index == col_index:
            return self.diagonal_values[row_index]
        return self.off_diagonal_value


@final
class SparseAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractMatrixAccessor[M, N, T]):

    __slots__ = (
        "non_zero_values",
        "zero_value",
        "shape",
        "non_zero_row_offsets",
        "non_zero_col_indices",
    )
    non_zero_values: tuple[T, ...]
    zero_value: T
    shape: tuple[M, N]
    non_zero_row_offsets: Array[int]
    non_zero_col_indices: Array[int]

    def __init__(
        self,
        non_zero_value_map: Mapping[tuple[int, int], T],
        shape: tuple[M, N],
        *,
        zero_value: T = 0,
    ) -> None:
        non_zero_pairs   = sorted(non_zero_value_map.items(), key=lambda item: item[0])
        non_zero_indices = tuple(map(lambda pair: pair[0], non_zero_pairs))
        non_zero_values  = tuple(map(lambda pair: pair[1], non_zero_pairs))

        self.non_zero_row_offsets = Array(SLL_TYPE_CODE, (0,))
        self.non_zero_col_indices = Array(SLL_TYPE_CODE, map(lambda index: index[1], non_zero_indices))

        row_counts = Counter(map(lambda index: index[0], non_zero_indices))
        for row_index in range(shape[0]):
            self.non_zero_row_offsets.append(
                row_counts[row_index]
                + self.non_zero_row_offsets[-1]
            )

        self.non_zero_values = non_zero_values
        self.zero_value = zero_value
        self.shape = shape  # pyright: ignore[reportIncompatibleMethodOverride]

    @override
    def __iter__(self) -> Iterator[T]:
        non_zero_values = self.non_zero_values
        zero_value = self.zero_value
        non_zero_row_offsets = self.non_zero_row_offsets
        non_zero_col_indices = self.non_zero_col_indices

        row_indices = range(self.row_count)
        col_indices = range(self.col_count)

        for row_index in row_indices:
            offset = non_zero_row_offsets[row_index]
            offset_end = non_zero_row_offsets[row_index + 1]
            for col_index in col_indices:
                if offset < offset_end and non_zero_col_indices[offset] == col_index:
                    yield non_zero_values[offset]
                    offset += 1
                else:
                    yield zero_value

    @override
    def __reversed__(self) -> Iterator[T]:
        non_zero_values = self.non_zero_values
        zero_value = self.zero_value
        non_zero_row_offsets = self.non_zero_row_offsets
        non_zero_col_indices = self.non_zero_col_indices

        row_indices = range(self.row_count - 1, -1, -1)
        col_indices = range(self.col_count - 1, -1, -1)

        for row_index in row_indices:
            offset = non_zero_row_offsets[row_index + 1] - 1
            offset_end = non_zero_row_offsets[row_index] - 1
            for col_index in col_indices:
                if offset > offset_end and non_zero_col_indices[offset] == col_index:
                    yield non_zero_values[offset]
                    offset -= 1
                else:
                    yield zero_value

    @property
    @override
    def row_count(self) -> M:
        return self.shape[0]

    @property
    @override
    def col_count(self) -> N:
        return self.shape[1]

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T:
        non_zero_row_offsets = self.non_zero_row_offsets
        non_zero_col_indices = self.non_zero_col_indices
        for offset in range(
            non_zero_row_offsets[row_index],
            non_zero_row_offsets[row_index + 1],
        ):
            if non_zero_col_indices[offset] == col_index:
                return self.non_zero_values[offset]
        return self.zero_value
