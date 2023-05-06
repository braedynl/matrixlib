from typing import Any, Generic, Literal, TypeVar

from .abc import BaseAccessor, BaseMatrixAccessor, BaseVectorAccessor

__all__ = [
    "SliceAccessor",
    "RowSliceAccessor",
    "ColSliceAccessor",
    "MatrixSliceAccessor",
]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class SliceAccessor(BaseVectorAccessor[Literal[1], int, T_co], Generic[T_co]):

    __slots__ = ("target", "window")

    target: BaseAccessor[Any, Any, T_co]
    window: range
    nrows: Literal[1] = 1

    def __init__(self, target: BaseAccessor[Any, Any, T_co], *, window: range) -> None:
        self.target = target
        self.window = window

    def __repr__(self) -> str:
        return f"SliceAccessor(target={self.target!r}, window={self.window!r})"

    def __hash__(self) -> int:
        return hash((self.target, self.window))

    @property
    def ncols(self) -> int:
        return len(self.window)

    def vector_access(self, index: int) -> T_co:
        index = self.window[index]
        return self.target.vector_access(index)


class RowSliceAccessor(BaseMatrixAccessor[Literal[1], int, T_co], Generic[T_co]):

    __slots__ = ("target", "row_index", "col_window")

    target: BaseAccessor[Any, Any, T_co]
    row_index: int
    col_window: range
    nrows: Literal[1] = 1

    def __init__(self, target: BaseAccessor[Any, Any, T_co], *, row_index: int, col_window: range) -> None:
        self.target = target
        self.row_index = row_index
        self.col_window = col_window

    def __repr__(self) -> str:
        return f"RowSliceAccessor(target={self.target!r}, row_index={self.row_index!r}, col_window={self.col_window!r})"

    def __hash__(self) -> int:
        return hash((self.target, self.row_index, self.col_window))

    @property
    def ncols(self) -> int:
        return len(self.col_window)

    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        row_index = self.row_index + row_index
        col_index = self.col_window[col_index]
        return self.target.matrix_access(row_index, col_index)


class ColSliceAccessor(BaseMatrixAccessor[int, Literal[1], T_co], Generic[T_co]):

    __slots__ = ("target", "row_window", "col_index")

    target: BaseAccessor[Any, Any, T_co]
    row_window: range
    col_index: int
    ncols: Literal[1] = 1

    def __init__(self, target: BaseAccessor[Any, Any, T_co], *, row_window: range, col_index: int) -> None:
        self.target = target
        self.row_window = row_window
        self.col_index = col_index

    def __repr__(self) -> str:
        return f"ColSliceAccessor(target={self.target!r}, row_window={self.row_window!r}, col_index={self.col_index!r})"

    def __hash__(self) -> int:
        return hash((self.target, self.row_window, self.col_index))

    @property
    def nrows(self) -> int:
        return len(self.row_window)

    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        row_index = self.row_window[row_index]
        col_index = self.col_index + col_index
        return self.target.matrix_access(row_index, col_index)


class MatrixSliceAccessor(BaseMatrixAccessor[int, int, T_co], Generic[T_co]):

    __slots__ = ("target", "row_window", "col_window")

    target: BaseAccessor[Any, Any, T_co]
    row_window: range
    col_window: range

    def __init__(self, target: BaseAccessor[Any, Any, T_co], *, row_window: range, col_window: range) -> None:
        self.target = target
        self.row_window = row_window
        self.col_window = col_window

    def __repr__(self) -> str:
        return f"MatrixSliceAccessor(target={self.target!r}, row_window={self.row_window!r}, col_window={self.col_window!r})"

    def __hash__(self) -> int:
        return hash((self.target, self.row_window, self.col_window))

    @property
    def nrows(self) -> int:
        return len(self.row_window)

    @property
    def ncols(self) -> int:
        return len(self.col_window)

    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        row_index = self.row_window[row_index]
        col_index = self.col_window[col_index]
        return self.target.matrix_access(row_index, col_index)
