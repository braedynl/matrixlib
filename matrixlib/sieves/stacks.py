from collections.abc import Iterator
from typing import TypeVar

from .abc import BaseAccessor, BaseMatrixAccessor

__all__ = [
    "RowStackAccessor",
    "ColStackAccessor",
]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class RowStackAccessor(BaseMatrixAccessor[int, N_co, T_co]):

    __slots__ = ("target_head", "target_body")

    target_head: BaseAccessor[int, N_co, T_co]
    target_body: tuple[BaseAccessor[int, N_co, T_co], ...]

    def __init__(self, target_head: BaseAccessor[int, N_co, T_co], target_body: tuple[BaseAccessor[int, N_co, T_co], ...]) -> None:
        self.target_head = target_head
        self.target_body = target_body

    def __repr__(self) -> str:
        return f"RowStackAccessor(target_head={self.target_head!r}, target_body={self.target_body!r})"

    def __hash__(self) -> int:
        return hash((self.target_head, self.target_body))

    @property
    def nrows(self) -> int:
        return (
            self.target_head.nrows
            + sum(map(lambda target: target.nrows, self.target_body))
        )

    @property
    def ncols(self) -> N_co:
        return self.target_head.ncols

    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        for target in self.targets():
            nrows = target.nrows
            if (row_index >= nrows):
                row_index -= nrows
            else:
                break
        return target.matrix_access(row_index, col_index)

    def targets(self) -> Iterator[BaseAccessor[int, N_co, T_co]]:
        yield self.target_head
        yield from self.target_body


class ColStackAccessor(BaseMatrixAccessor[M_co, int, T_co]):

    __slots__ = ("target_head", "target_body")

    target_head: BaseAccessor[M_co, int, T_co]
    target_body: tuple[BaseAccessor[M_co, int, T_co], ...]

    def __init__(self, target_head: BaseAccessor[M_co, int, T_co], target_body: tuple[BaseAccessor[M_co, int, T_co], ...]) -> None:
        self.target_head = target_head
        self.target_body = target_body

    def __repr__(self) -> str:
        return f"ColStackAccessor(target_head={self.target_head!r}, target_body={self.target_body!r})"

    def __hash__(self) -> int:
        return hash((self.target_head, self.target_body))

    @property
    def nrows(self) -> M_co:
        return self.target_head.nrows

    @property
    def ncols(self) -> int:
        return (
            self.target_head.ncols
            + sum(map(lambda target: target.ncols, self.target_body))
        )

    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        for target in self.targets():
            ncols = target.ncols
            if (col_index >= ncols):
                col_index -= ncols
            else:
                break
        return target.matrix_access(row_index, col_index)

    def targets(self) -> Iterator[BaseAccessor[M_co, int, T_co]]:
        yield self.target_head
        yield from self.target_body
