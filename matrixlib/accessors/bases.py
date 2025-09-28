from __future__ import annotations

__all__ = [
    "MatrixAccessor",
    "RowVectorAccessor",
    "ColVectorAccessor",
    "ValueAccessor",
]

import itertools
from abc import ABCMeta, abstractmethod
from array import array as Array
from collections.abc import Iterator, Mapping
from typing import Literal, cast, final, override

from .abstracts import AbstractAccessor, AbstractVectorAccessor, AbstractMatrixAccessor


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
class SparseAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
    S: object = object,
](AbstractMatrixAccessor[M, N, T | S]):

    __slots__ = (
        "non_zeroes",
        "zero",
        "shape",
        "_nz_row_offsets",
        "_nz_col_indices",
    )
    non_zeroes: tuple[T, ...]
    zero: S
    shape: tuple[M, N]
    _nz_row_offsets: Array[int]
    _nz_col_indices: Array[int]

    def __init__(
        self,
        non_zeroes: Mapping[tuple[int, int], T],
        zero: S,
        shape: tuple[M, N],
    ) -> None:
        sorted_non_zeroes = sorted(non_zeroes.items(), key=lambda item: item[0])

        nz_row_offsets = Array("q", (0,))
        begin = 0
        end = len(sorted_non_zeroes)
        for row_index in range(shape[0]):
            row_offset = nz_row_offsets[-1]
            for i in range(begin, end):
                matrix_index = sorted_non_zeroes[i][0]
                if matrix_index[0] == row_index:
                    row_offset += 1
                else:
                    begin = i
                    break
            nz_row_offsets.append(row_offset)

        assert len(nz_row_offsets) == shape[0] + 1

        self._nz_row_offsets = nz_row_offsets
        self._nz_col_indices = Array(
            "q",
            (pair[0][1] for pair in sorted_non_zeroes),
        )

        self.non_zeroes = tuple(pair[1] for pair in sorted_non_zeroes)
        self.zero = zero
        self.shape = shape  # pyright: ignore[reportIncompatibleMethodOverride]

    @override
    def __iter__(self) -> Iterator[T | S]:
        non_zeroes = self.non_zeroes
        zero = self.zero
        nz_row_offsets = self._nz_row_offsets
        nz_col_indices = self._nz_col_indices

        row_indices = range(self.row_count)
        col_indices = range(self.col_count)

        for row_index in row_indices:
            offset = nz_row_offsets[row_index]
            offset_end = nz_row_offsets[row_index + 1]

            for col_index in col_indices:
                if offset < offset_end and nz_col_indices[offset] == col_index:
                    yield non_zeroes[offset]
                    offset += 1
                else:
                    yield zero

    @override
    def __reversed__(self) -> Iterator[T | S]:
        raise NotImplementedError

    @property
    @override
    def row_count(self) -> M:
        return self.shape[0]

    @property
    @override
    def col_count(self) -> N:
        return self.shape[1]

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T | S:
        raise NotImplementedError
