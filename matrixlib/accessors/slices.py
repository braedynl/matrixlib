from __future__ import annotations

__all__ = [
    "SliceAccessor",
    "RowSliceAccessor",
    "ColSliceAccessor",
    "MatrixSliceAccessor",
]

from typing import Generic, Literal, TypeVar, final

from typing_extensions import override

from .abstracts import (AbstractAccessor, AbstractMatrixAccessor,
                        AbstractVectorAccessor)

T_co = TypeVar("T_co", covariant=True)


@final
class SliceAccessor(AbstractVectorAccessor[T_co], Generic[T_co]):

    __slots__ = ("target", "window")
    target: AbstractAccessor[T_co]
    window: range
    row_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, target: AbstractAccessor[T_co], *, window: range) -> None:
        self.target = target
        self.window = window

    def __repr__(self) -> str:
        return f"SliceAccessor(target={self.target!r}, window={self.window!r})"

    def __hash__(self) -> int:
        return hash((self.target, self.window))

    @property
    @override
    def col_count(self) -> int:
        return len(self.window)

    @override
    def vector_access(self, index: int) -> T_co:
        return self.target.vector_access(self.window[index])


@final
class RowSliceAccessor(AbstractMatrixAccessor[T_co], Generic[T_co]):

    __slots__ = ("target", "row_index", "col_window")
    target: AbstractAccessor[T_co]
    row_index: int
    col_window: range
    row_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, target: AbstractAccessor[T_co], *, row_index: int, col_window: range) -> None:
        self.target = target
        self.row_index = row_index
        self.col_window = col_window

    def __repr__(self) -> str:
        return f"RowSliceAccessor(target={self.target!r}, row_index={self.row_index!r}, col_window={self.col_window!r})"

    def __hash__(self) -> int:
        return hash((self.target, self.row_index, self.col_window))

    @property
    @override
    def col_count(self) -> int:
        return len(self.col_window)

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(
            self.row_index + row_index,
            self.col_window[col_index],
        )


@final
class ColSliceAccessor(AbstractMatrixAccessor[T_co], Generic[T_co]):

    __slots__ = ("target", "row_window", "col_index")
    target: AbstractAccessor[T_co]
    row_window: range
    col_index: int
    col_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, target: AbstractAccessor[T_co], *, row_window: range, col_index: int) -> None:
        self.target = target
        self.row_window = row_window
        self.col_index = col_index

    def __repr__(self) -> str:
        return f"ColSliceAccessor(target={self.target!r}, row_window={self.row_window!r}, col_index={self.col_index!r})"

    def __hash__(self) -> int:
        return hash((self.target, self.row_window, self.col_index))

    @property
    @override
    def row_count(self) -> int:
        return len(self.row_window)

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(
            self.row_window[row_index],
            self.col_index + col_index,
        )


@final
class MatrixSliceAccessor(AbstractMatrixAccessor[T_co], Generic[T_co]):

    __slots__ = ("target", "row_window", "col_window")
    target: AbstractAccessor[T_co]
    row_window: range
    col_window: range

    def __init__(self, target: AbstractAccessor[T_co], *, row_window: range, col_window: range) -> None:
        self.target = target
        self.row_window = row_window
        self.col_window = col_window

    def __repr__(self) -> str:
        return f"MatrixSliceAccessor(target={self.target!r}, row_window={self.row_window!r}, col_window={self.col_window!r})"

    def __hash__(self) -> int:
        return hash((self.target, self.row_window, self.col_window))

    @property
    @override
    def row_count(self) -> int:
        return len(self.row_window)

    @property
    @override
    def col_count(self) -> int:
        return len(self.col_window)

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(
            self.row_window[row_index],
            self.col_window[col_index],
        )
