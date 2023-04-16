import itertools
from typing import TypeVar

from .abc import MatrixSieve, Sieve

__all__ = [
    "RowStack",
    "ColStack",
]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class RowStack(MatrixSieve[int, N_co, T_co]):

    __slots__ = ("target_head", "target_body")

    target_head: Sieve[int, N_co, T_co]
    target_body: tuple[Sieve[int, N_co, T_co], ...]

    def __init__(self, target_head: Sieve[int, N_co, T_co], target_body: tuple[Sieve[int, N_co, T_co], ...]) -> None:
        self.target_head = target_head
        self.target_body = target_body

    def __repr__(self) -> str:
        return f"RowStack(target_head={self.target_head!r}, target_body={self.target_body!r})"

    @property
    def nrows(self) -> int:
        return (
            self.target_head.nrows
            + sum(map(lambda target: target.nrows, self.target_body))
        )

    @property
    def ncols(self) -> N_co:
        return self.target_head.ncols

    def matrix_collect(self, row_index: int, col_index: int) -> T_co:
        for target in itertools.chain((self.target_head,), self.target_body):
            nrows = target.nrows
            if (row_index >= nrows):
                row_index -= nrows
            else:
                break
        return target.matrix_collect(row_index, col_index)


class ColStack(MatrixSieve[M_co, int, T_co]):

    __slots__ = ("target_head", "target_body")

    target_head: Sieve[M_co, int, T_co]
    target_body: tuple[Sieve[M_co, int, T_co], ...]

    def __init__(self, target_head: Sieve[M_co, int, T_co], target_body: tuple[Sieve[M_co, int, T_co], ...]) -> None:
        self.target_head = target_head
        self.target_body = target_body

    def __repr__(self) -> str:
        return f"ColStack(target_head={self.target_head!r}, target_body={self.target_body!r})"

    @property
    def nrows(self) -> M_co:
        return self.target_head.nrows

    @property
    def ncols(self) -> int:
        return (
            self.target_head.ncols
            + sum(map(lambda target: target.ncols, self.target_body))
        )

    def matrix_collect(self, row_index: int, col_index: int) -> T_co:
        for target in itertools.chain((self.target_head,), self.target_body):
            ncols = target.ncols
            if (col_index >= ncols):
                col_index -= ncols
            else:
                break
        return target.matrix_collect(row_index, col_index)
