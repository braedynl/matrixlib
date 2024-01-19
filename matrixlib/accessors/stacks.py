from __future__ import annotations

__all__ = [
    "RowStackAccessor",
    "ColStackAccessor",
]

from typing import TypeVar

from typing_extensions import override

from .abc import BaseAccessor, BaseMatrixAccessor

T_co = TypeVar("T_co", covariant=True)


class RowStackAccessor(BaseMatrixAccessor[T_co]):

    __slots__ = ("target_head", "target_tail")
    target_head: BaseAccessor[T_co]
    target_tail: BaseAccessor[T_co]

    def __init__(self, target_head: BaseAccessor[T_co], target_tail: BaseAccessor[T_co]) -> None:
        self.target_head = target_head
        self.target_tail = target_tail

    def __repr__(self) -> str:
        return f"RowStackAccessor(target_head={self.target_head!r}, target_tail={self.target_tail!r})"

    def __hash__(self) -> int:
        return hash((self.target_head, self.target_tail))

    @property
    @override
    def row_count(self) -> int:
        return self.target_head.row_count + self.target_tail.row_count

    @property
    @override
    def col_count(self) -> int:
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


class ColStackAccessor(BaseMatrixAccessor[T_co]):

    __slots__ = ("target_head", "target_tail")
    target_head: BaseAccessor[T_co]
    target_tail: BaseAccessor[T_co]

    def __init__(self, target_head: BaseAccessor[T_co], target_tail: BaseAccessor[T_co]) -> None:
        self.target_head = target_head
        self.target_tail = target_tail

    def __repr__(self) -> str:
        return f"ColStackAccessor(target_head={self.target_head!r}, target_tail={self.target_tail!r})"

    def __hash__(self) -> int:
        return hash((self.target_head, self.target_tail))

    @property
    @override
    def row_count(self) -> int:
        return self.target_head.row_count

    @property
    @override
    def col_count(self) -> int:
        return self.target_head.col_count + self.target_tail.col_count

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
