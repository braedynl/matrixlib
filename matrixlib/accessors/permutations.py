from __future__ import annotations

__all__ = [
    "TransposeAccessor",
    "RowFlipAccessor",
    "ColFlipAccessor",
    "Rotate090Accessor",
    "Rotate180Accessor",
    "Rotate270Accessor",
    "ReverseAccessor",
]

from abc import ABCMeta, abstractmethod
from typing import final, override

from .abstracts import (AbstractAccessor, AbstractMatrixAccessor,
                        AbstractVectorAccessor)


class AbstractPermutationAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractAccessor[M, N, T], metaclass=ABCMeta):

    __slots__ = ()

    def __hash__(self) -> int:
        return hash(self.target)

    @property
    @abstractmethod
    def target(self) -> AbstractAccessor[M, N, T]:
        """The permuted accessor"""
        raise NotImplementedError


@final
class TransposeAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractMatrixAccessor[M, N, T], AbstractPermutationAccessor[M, N, T]):

    __slots__ = ("target",)
    target: AbstractAccessor[N, M, T]  # NOTE: Reversed dimensions!

    def __init__(self, target: AbstractAccessor[N, M, T]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @override
    def row_count(self) -> M:
        return self.target.col_count

    @property
    @override
    def col_count(self) -> N:
        return self.target.row_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T:
        return self.target.matrix_access(col_index, row_index)


@final
class RowFlipAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractMatrixAccessor[M, N, T], AbstractPermutationAccessor[M, N, T]):

    __slots__ = ("target",)
    target: AbstractAccessor[M, N, T]

    def __init__(self, target: AbstractAccessor[M, N, T]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @override
    def row_count(self) -> M:
        return self.target.row_count

    @property
    @override
    def col_count(self) -> N:
        return self.target.col_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T:
        return self.target.matrix_access(
            self.row_count - row_index - 1,
            col_index,
        )


@final
class ColFlipAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractMatrixAccessor[M, N, T], AbstractPermutationAccessor[M, N, T]):

    __slots__ = ("target",)
    target: AbstractAccessor[M, N, T]

    def __init__(self, target: AbstractAccessor[M, N, T]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @override
    def row_count(self) -> M:
        return self.target.row_count

    @property
    @override
    def col_count(self) -> N:
        return self.target.col_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T:
        return self.target.matrix_access(
            row_index,
            self.col_count - col_index - 1,
        )


@final
class Rotate090Accessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractMatrixAccessor[M, N, T], AbstractPermutationAccessor[M, N, T]):

    __slots__ = ("target",)
    target: AbstractAccessor[N, M, T]  # NOTE: Reversed dimensions!

    def __init__(self, target: AbstractAccessor[N, M, T]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @override
    def row_count(self) -> M:
        return self.target.col_count

    @property
    @override
    def col_count(self) -> N:
        return self.target.row_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T:
        return self.target.matrix_access(
            col_index,
            self.row_count - row_index - 1,
        )


@final
class Rotate180Accessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractVectorAccessor[M, N, T], AbstractPermutationAccessor[M, N, T]):

    __slots__ = ("target",)
    target: AbstractAccessor[M, N, T]

    def __init__(self, target: AbstractAccessor[M, N, T]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @override
    def row_count(self) -> M:
        return self.target.row_count

    @property
    @override
    def col_count(self) -> N:
        return self.target.col_count

    @override
    def vector_access(self, index: int) -> T:
        return self.target.vector_access(len(self) - index - 1)


@final
class Rotate270Accessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractMatrixAccessor[M, N, T], AbstractPermutationAccessor[M, N, T]):

    __slots__ = ("target",)
    target: AbstractAccessor[N, M, T]  # NOTE: Reversed dimensions!

    def __init__(self, target: AbstractAccessor[N, M, T]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @override
    def row_count(self) -> M:
        return self.target.col_count

    @property
    @override
    def col_count(self) -> N:
        return self.target.row_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T:
        return self.target.matrix_access(
            self.col_count - col_index - 1,
            row_index,
        )


ReverseAccessor = Rotate180Accessor  #: Alias of ``Rotate180Accessor``.
