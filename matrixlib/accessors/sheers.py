from __future__ import annotations

__all__ = [
    "RowSheerAccessor",
    "ColSheerAccessor",
]

from typing import Generic, Literal, TypeVar, final

from typing_extensions import override

from .abstracts import AbstractAccessor, AbstractMatrixAccessor

T_co = TypeVar("T_co", covariant=True)


@final
class RowSheerAccessor(AbstractMatrixAccessor[T_co], Generic[T_co]):

    __slots__ = ("target", "row_index")
    target: AbstractAccessor[T_co]
    row_index: int

    def __init__(self, target: AbstractAccessor[T_co], *, row_index: int) -> None:
        self.target = target
        self.row_index = row_index

    def __repr__(self) -> str:
        return f"RowSheerAccessor(target={self.target!r}, row_index={self.row_index!r})"

    def __hash__(self) -> int:
        return hash((self.target, self.row_index))

    @property
    @override
    def row_count(self) -> Literal[1]:
        return 1

    @property
    @override
    def col_count(self) -> int:
        return self.target.col_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(
            self.row_index + row_index,
            col_index,
        )


@final
class ColSheerAccessor(AbstractMatrixAccessor[T_co], Generic[T_co]):

    __slots__ = ("target", "col_index")
    target: AbstractAccessor[T_co]
    col_index: int

    def __init__(self, target: AbstractAccessor[T_co], *, col_index: int) -> None:
        self.target = target
        self.col_index = col_index

    def __repr__(self) -> str:
        return f"ColSheerAccessor(target={self.target!r}, col_index={self.col_index!r})"

    def __hash__(self) -> int:
        return hash((self.target, self.col_index))

    @property
    @override
    def row_count(self) -> int:
        return self.target.row_count

    @property
    @override
    def col_count(self) -> Literal[1]:
        return 1

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(
            row_index,
            self.col_index + col_index,
        )
