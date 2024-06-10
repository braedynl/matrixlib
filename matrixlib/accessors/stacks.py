from __future__ import annotations

__all__ = ["RowStackAccessor", "ColStackAccessor"]

from typing import TypeVar, cast, final

from typing_extensions import override

from .abstracts import AbstractAccessor, AbstractMatrixAccessor

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)

T_co = TypeVar("T_co", covariant=True)


@final
class RowStackAccessor(AbstractMatrixAccessor[M_co, N_co, T_co]):

    __slots__ = ("target_head", "target_tail")
    target_head: AbstractAccessor[int, N_co, T_co]
    target_tail: AbstractAccessor[int, N_co, T_co]

    def __init__(self, target_head: AbstractAccessor[int, N_co, T_co], target_tail: AbstractAccessor[int, N_co, T_co]) -> None:
        assert target_head.col_count == target_tail.col_count
        self.target_head = target_head
        self.target_tail = target_tail

    def __hash__(self) -> int:
        return hash((self.target_head, self.target_tail))

    @property
    @override
    def row_count(self) -> M_co:
        return cast(M_co, self.target_head.row_count + self.target_tail.row_count)

    @property
    @override
    def col_count(self) -> N_co:
        return self.target_head.col_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        row_count = self.target_head.row_count
        if row_index >= row_count:
            value = self.target_tail.matrix_access(
                row_index - row_count,
                col_index,
            )
        else:
            value = self.target_head.matrix_access(
                row_index,
                col_index,
            )
        return value


@final
class ColStackAccessor(AbstractMatrixAccessor[M_co, N_co, T_co]):

    __slots__ = ("target_head", "target_tail")
    target_head: AbstractAccessor[M_co, int, T_co]
    target_tail: AbstractAccessor[M_co, int, T_co]

    def __init__(self, target_head: AbstractAccessor[M_co, int, T_co], target_tail: AbstractAccessor[M_co, int, T_co]) -> None:
        assert target_head.row_count == target_tail.row_count
        self.target_head = target_head
        self.target_tail = target_tail

    def __hash__(self) -> int:
        return hash((self.target_head, self.target_tail))

    @property
    @override
    def row_count(self) -> M_co:
        return self.target_head.row_count

    @property
    @override
    def col_count(self) -> N_co:
        return cast(N_co, self.target_head.col_count + self.target_tail.col_count)

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        col_count = self.target_head.col_count
        if col_index >= col_count:
            value = self.target_tail.matrix_access(
                row_index,
                col_index - col_count,
            )
        else:
            value = self.target_head.matrix_access(
                row_index,
                col_index,
            )
        return value
