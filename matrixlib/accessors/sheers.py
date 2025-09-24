from __future__ import annotations

__all__ = ["RowSheerAccessor", "ColSheerAccessor"]

from typing import Literal, final, override

from .abstracts import AbstractAccessor, AbstractMatrixAccessor


@final
class RowSheerAccessor[N: int = int, T: object = object](AbstractMatrixAccessor[Literal[1], N, T]):

    __slots__ = ("target", "row_index")
    target: AbstractAccessor[int, N, T]
    row_index: int
    row_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, target: AbstractAccessor[int, N, T], *, row_index: int) -> None:
        self.target = target
        self.row_index = row_index

    def __hash__(self) -> int:
        return hash((self.target, self.row_index))

    @property
    @override
    def col_count(self) -> N:
        return self.target.col_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T:
        return self.target.matrix_access(
            self.row_index + row_index,
            col_index,
        )


@final
class ColSheerAccessor[M: int = int, T: object = object](AbstractMatrixAccessor[M, Literal[1], T]):

    __slots__ = ("target", "col_index")
    target: AbstractAccessor[M, int, T]
    col_index: int
    col_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, target: AbstractAccessor[M, int, T], *, col_index: int) -> None:
        self.target = target
        self.col_index = col_index

    def __hash__(self) -> int:
        return hash((self.target, self.col_index))

    @property
    @override
    def row_count(self) -> M:
        return self.target.row_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T:
        return self.target.matrix_access(
            row_index,
            self.col_index + col_index,
        )
