from typing import Any, Generic, Literal, TypeVar

from .abc import MatrixSieve, Sieve, VectorSieve

__all__ = [
    "Slice",
    "RowSlice",
    "ColSlice",
    "RowColSlice",
]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class Slice(VectorSieve[Literal[1], int, T_co], Generic[T_co]):

    __slots__ = ("target", "window")

    target: Sieve[Any, Any, T_co]
    window: range
    nrows: Literal[1] = 1

    def __init__(self, target: Sieve[Any, Any, T_co], *, window: range) -> None:
        self.target = target
        self.window = window

    def __repr__(self) -> str:
        return f"Slice(target={self.target!r}, window={self.window!r})"

    @property
    def ncols(self) -> int:
        return len(self.window)

    def vector_sieve(self, index: int) -> T_co:
        index = self.window[index]
        return self.target.vector_sieve(index)


class RowSlice(MatrixSieve[Literal[1], int, T_co], Generic[T_co]):

    __slots__ = ("target", "row_index", "col_window")

    target: Sieve[Any, Any, T_co]
    row_index: int
    col_window: range
    nrows: Literal[1] = 1

    def __init__(self, target: Sieve[Any, Any, T_co], *, row_index: int, col_window: range) -> None:
        self.target = target
        self.row_index = row_index
        self.col_window = col_window

    def __repr__(self) -> str:
        return f"RowSlice(target={self.target!r}, row_index={self.row_index!r}, col_window={self.col_window!r})"

    @property
    def ncols(self) -> int:
        return len(self.col_window)

    def matrix_sieve(self, row_index: int, col_index: int) -> T_co:
        row_index = self.row_index + row_index
        col_index = self.col_window[col_index]
        return self.target.matrix_sieve(row_index, col_index)


class ColSlice(MatrixSieve[int, Literal[1], T_co], Generic[T_co]):

    __slots__ = ("target", "row_window", "col_index")

    target: Sieve[Any, Any, T_co]
    row_window: range
    col_index: int
    ncols: Literal[1] = 1

    def __init__(self, target: Sieve[Any, Any, T_co], *, row_window: range, col_index: int) -> None:
        self.target = target
        self.row_window = row_window
        self.col_index = col_index

    def __repr__(self) -> str:
        return f"ColSlice(target={self.target!r}, row_window={self.row_window!r}, col_index={self.col_index!r})"

    @property
    def nrows(self) -> int:
        return len(self.row_window)

    def matrix_sieve(self, row_index: int, col_index: int) -> T_co:
        row_index = self.row_window[row_index]
        col_index = self.col_index + col_index
        return self.target.matrix_sieve(row_index, col_index)


class RowColSlice(MatrixSieve[int, int, T_co], Generic[T_co]):

    __slots__ = ("target", "row_window", "col_window")

    target: Sieve[Any, Any, T_co]
    row_window: range
    col_window: range

    def __init__(self, target: Sieve[Any, Any, T_co], *, row_window: range, col_window: range) -> None:
        self.target = target
        self.row_window = row_window
        self.col_window = col_window

    def __repr__(self) -> str:
        return f"RowColSlice(target={self.target!r}, row_window={self.row_window!r}, col_window={self.col_window!r})"

    @property
    def nrows(self) -> int:
        return len(self.row_window)

    @property
    def ncols(self) -> int:
        return len(self.col_window)

    def matrix_sieve(self, row_index: int, col_index: int) -> T_co:
        row_index = self.row_window[row_index]
        col_index = self.col_window[col_index]
        return self.target.matrix_sieve(row_index, col_index)
