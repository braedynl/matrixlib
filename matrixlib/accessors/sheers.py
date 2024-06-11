from __future__ import annotations

__all__ = ["RowSheerAccessor", "ColSheerAccessor"]

from typing import Generic, Literal, final

from typing_extensions import TypeVar, override

from .abstracts import AbstractAccessor, AbstractMatrixAccessor

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)

T_co = TypeVar("T_co", covariant=True, default=object)


@final
class RowSheerAccessor(AbstractMatrixAccessor[Literal[1], N_co, T_co], Generic[N_co, T_co]):

    __slots__ = ("target", "row_index")
    target: AbstractAccessor[int, N_co, T_co]
    row_index: int
    row_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, target: AbstractAccessor[int, N_co, T_co], *, row_index: int) -> None:
        self.target = target
        self.row_index = row_index

    def __hash__(self) -> int:
        return hash((self.target, self.row_index))

    @property
    @override
    def col_count(self) -> N_co:
        return self.target.col_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(
            self.row_index + row_index,
            col_index,
        )


@final
class ColSheerAccessor(AbstractMatrixAccessor[M_co, Literal[1], T_co], Generic[M_co, T_co]):

    __slots__ = ("target", "col_index")
    target: AbstractAccessor[M_co, int, T_co]
    col_index: int
    col_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, target: AbstractAccessor[M_co, int, T_co], *, col_index: int) -> None:
        self.target = target
        self.col_index = col_index

    def __hash__(self) -> int:
        return hash((self.target, self.col_index))

    @property
    @override
    def row_count(self) -> M_co:
        return self.target.row_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(
            row_index,
            self.col_index + col_index,
        )
