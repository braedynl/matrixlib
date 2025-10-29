from __future__ import annotations

__all__ = [
    "SliceAccessor",
    "RowSliceAccessor",
    "ColSliceAccessor",
    "MatrixSliceAccessor",
]

from typing import Literal, Self, cast, final, override

from .abstracts import (AbstractAccessor, AbstractMatrixAccessor,
                        AbstractVectorAccessor)


@final
class SliceAccessor[N: int = int, T: object = object](AbstractVectorAccessor[Literal[1], N, T]):

    __slots__ = ("target", "window")
    target: AbstractAccessor[int, N, T]
    window: range
    row_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __new__(cls, target: AbstractAccessor[int, N, T], window: range) -> Self:
        self = super(SliceAccessor, cls).__new__(cls)
        self.target = target
        self.window = window
        return self

    def __hash__(self) -> int:
        return hash((self.target, self.window))

    def __reduce__(self) -> tuple[object, ...]:
        return (self.__class__, (self.target, self.window))

    @property
    @override
    def col_count(self) -> N:
        return cast(N, len(self.window))

    @override
    def vector_access(self, index: int) -> T:
        return self.target.vector_access(self.window[index])


@final
class RowSliceAccessor[N: int = int, T: object = object](AbstractMatrixAccessor[Literal[1], N, T]):

    __slots__ = ("target", "row_index", "col_window")
    target: AbstractAccessor[int, N, T]
    row_index: int
    col_window: range
    row_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __new__(cls, target: AbstractAccessor[int, N, T], row_index: int, col_window: range) -> Self:
        self = super(RowSliceAccessor, cls).__new__(cls)
        self.target = target
        self.row_index = row_index
        self.col_window = col_window
        return self

    def __hash__(self) -> int:
        return hash((self.target, self.row_index, self.col_window))

    def __reduce__(self) -> tuple[object, ...]:
        return (self.__class__, (self.target, self.row_index, self.col_window))

    @property
    @override
    def col_count(self) -> N:
        return cast(N, len(self.col_window))

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T:
        return self.target.matrix_access(
            self.row_index + row_index,
            self.col_window[col_index],
        )


@final
class ColSliceAccessor[M: int = int, T: object = object](AbstractMatrixAccessor[M, Literal[1], T]):

    __slots__ = ("target", "row_window", "col_index")
    target: AbstractAccessor[M, int, T]
    row_window: range
    col_index: int
    col_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __new__(cls, target: AbstractAccessor[M, int, T], row_window: range, col_index: int) -> Self:
        self = super(ColSliceAccessor, cls).__new__(cls)
        self.target = target
        self.row_window = row_window
        self.col_index = col_index
        return self

    def __hash__(self) -> int:
        return hash((self.target, self.row_window, self.col_index))

    def __reduce__(self) -> tuple[object, ...]:
        return (self.__class__, (self.target, self.row_window, self.col_index))

    @property
    @override
    def row_count(self) -> M:
        return cast(M, len(self.row_window))

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T:
        return self.target.matrix_access(
            self.row_window[row_index],
            self.col_index + col_index,
        )


@final
class MatrixSliceAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractMatrixAccessor[M, N, T]):

    __slots__ = ("target", "row_window", "col_window")
    target: AbstractAccessor[int, int, T]
    row_window: range
    col_window: range

    def __new__(
        cls,
        target: AbstractAccessor[int, int, T],
        *,
        row_window: range,
        col_window: range,
    ) -> Self:
        self = super(MatrixSliceAccessor, cls).__new__(cls)
        self.target = target
        self.row_window = row_window
        self.col_window = col_window
        return self

    def __hash__(self) -> int:
        return hash((self.target, self.row_window, self.col_window))

    def __reduce__(self) -> tuple[object, ...]:
        return (self.__class__, (self.target, self.row_window, self.col_window))

    @property
    @override
    def row_count(self) -> M:
        return cast(M, len(self.row_window))

    @property
    @override
    def col_count(self) -> N:
        return cast(N, len(self.col_window))

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T:
        return self.target.matrix_access(
            self.row_window[row_index],
            self.col_window[col_index],
        )
