from typing import Any, Generic, Literal, TypeVar

from .abc import MatrixSieve, Sieve

__all__ = [
    "RowSheer",
    "ColSheer",
]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class RowSheer(MatrixSieve[Literal[1], N_co, T_co], Generic[N_co, T_co]):

    __slots__ = ("target", "row_index")

    target: Sieve[Any, N_co, T_co]
    row_index: int
    nrows: Literal[1] = 1

    def __init__(self, target: Sieve[Any, N_co, T_co], *, row_index: int) -> None:
        self.target = target
        self.row_index = row_index

    def __repr__(self) -> str:
        return f"RowSheer(target={self.target!r}, row_index={self.row_index!r})"

    @property
    def ncols(self) -> N_co:
        return self.target.ncols

    def matrix_collect(self, row_index: int, col_index: int) -> T_co:
        row_index = self.row_index + row_index
        return self.target.matrix_collect(row_index, col_index)


class ColSheer(MatrixSieve[M_co, Literal[1], T_co], Generic[M_co, T_co]):

    __slots__ = ("target", "col_index")

    target: Sieve[M_co, Any, T_co]
    col_index: int
    ncols: Literal[1] = 1

    def __init__(self, target: Sieve[M_co, Any, T_co], *, col_index: int) -> None:
        self.target = target
        self.col_index = col_index

    def __repr__(self) -> str:
        return f"ColSheer(target={self.target!r}, col_index={self.col_index!r})"

    @property
    def nrows(self) -> M_co:
        return self.target.nrows

    def matrix_collect(self, row_index: int, col_index: int) -> T_co:
        col_index = self.col_index + col_index
        return self.target.matrix_collect(row_index, col_index)
