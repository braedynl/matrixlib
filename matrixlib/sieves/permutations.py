from typing import Generic, TypeVar

from .abc import MatrixSieve, Sieve, VectorSieve

__all__ = [
    "Transposition",
    "RowFlip",
    "ColFlip",
    "Rotation090",
    "Rotation180",
    "Rotation270",
]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class Transposition(MatrixSieve[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: Sieve[N_co, M_co, T_co]

    def __init__(self, target: Sieve[N_co, M_co, T_co]) -> None:
        self.target = target

    def __repr__(self) -> str:
        return f"Transposition(target={self.target!r})"

    @property
    def nrows(self) -> M_co:
        return self.target.ncols

    @property
    def ncols(self) -> N_co:
        return self.target.nrows

    def matrix_collect(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_collect(col_index, row_index)


class RowFlip(MatrixSieve[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: Sieve[M_co, N_co, T_co]

    def __init__(self, target: Sieve[M_co, N_co, T_co]) -> None:
        self.target = target

    def __repr__(self) -> str:
        return f"RowFlip(target={self.target!r})"

    @property
    def nrows(self) -> M_co:
        return self.target.nrows

    @property
    def ncols(self) -> N_co:
        return self.target.ncols

    def matrix_collect(self, row_index: int, col_index: int) -> T_co:
        row_index = self.nrows - row_index - 1
        return self.target.matrix_collect(row_index, col_index)


class ColFlip(MatrixSieve[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: Sieve[M_co, N_co, T_co]

    def __init__(self, target: Sieve[M_co, N_co, T_co]) -> None:
        self.target = target

    def __repr__(self) -> str:
        return f"ColFlip(target={self.target!r})"

    @property
    def nrows(self) -> M_co:
        return self.target.nrows

    @property
    def ncols(self) -> N_co:
        return self.target.ncols

    def matrix_collect(self, row_index: int, col_index: int) -> T_co:
        col_index = self.ncols - col_index - 1
        return self.target.matrix_collect(row_index, col_index)


class Rotation090(MatrixSieve[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: Sieve[N_co, M_co, T_co]

    def __init__(self, target: Sieve[N_co, M_co, T_co]) -> None:
        self.target = target

    def __repr__(self) -> str:
        return f"Rotation090(target={self.target!r})"

    @property
    def nrows(self) -> M_co:
        return self.target.ncols

    @property
    def ncols(self) -> N_co:
        return self.target.nrows

    def matrix_collect(self, row_index: int, col_index: int) -> T_co:
        row_index = self.nrows - row_index - 1
        return self.target.matrix_collect(col_index, row_index)


class Rotation180(VectorSieve[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: Sieve[M_co, N_co, T_co]

    def __init__(self, target: Sieve[M_co, N_co, T_co]) -> None:
        self.target = target

    def __repr__(self) -> str:
        return f"Rotation180(target={self.target!r})"

    @property
    def nrows(self) -> M_co:
        return self.target.nrows

    @property
    def ncols(self) -> N_co:
        return self.target.ncols

    def vector_collect(self, index: int) -> T_co:
        index = len(self) - index - 1
        return self.target.vector_collect(index)


class Rotation270(MatrixSieve[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: Sieve[N_co, M_co, T_co]

    def __init__(self, target: Sieve[N_co, M_co, T_co]) -> None:
        self.target = target

    def __repr__(self) -> str:
        return f"Rotation270(target={self.target!r})"

    @property
    def nrows(self) -> M_co:
        return self.target.ncols

    @property
    def ncols(self) -> N_co:
        return self.target.nrows

    def matrix_collect(self, row_index: int, col_index: int) -> T_co:
        col_index = self.ncols - col_index - 1
        return self.target.matrix_collect(col_index, row_index)
