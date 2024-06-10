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
from typing import Generic, TypeVar, final

from typing_extensions import TypeAlias, override

from .abstracts import (AbstractAccessor, AbstractMatrixAccessor,
                        AbstractVectorAccessor)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)

T_co = TypeVar("T_co", covariant=True)


class AbstractPermutationAccessor(AbstractAccessor[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __hash__(self) -> int:
        return hash(self.target)

    @property
    @abstractmethod
    def target(self) -> AbstractAccessor[M_co, N_co, T_co]:
        """The permuted accessor"""
        raise NotImplementedError


@final
class TransposeAccessor(
    AbstractMatrixAccessor[M_co, N_co, T_co],
    AbstractPermutationAccessor[M_co, N_co, T_co],
    Generic[M_co, N_co, T_co],
):

    __slots__ = ("target")
    target: AbstractAccessor[N_co, M_co, T_co]  # NOTE: Reversed dimensions!

    def __init__(self, target: AbstractAccessor[N_co, M_co, T_co]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @override
    def row_count(self) -> M_co:
        return self.target.col_count

    @property
    @override
    def col_count(self) -> N_co:
        return self.target.row_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(col_index, row_index)


@final
class RowFlipAccessor(
    AbstractMatrixAccessor[M_co, N_co, T_co],
    AbstractPermutationAccessor[M_co, N_co, T_co],
    Generic[M_co, N_co, T_co],
):

    __slots__ = ("target")
    target: AbstractAccessor[M_co, N_co, T_co]

    def __init__(self, target: AbstractAccessor[M_co, N_co, T_co]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @override
    def row_count(self) -> M_co:
        return self.target.row_count

    @property
    @override
    def col_count(self) -> N_co:
        return self.target.col_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(
            self.row_count - row_index - 1,
            col_index,
        )


@final
class ColFlipAccessor(
    AbstractMatrixAccessor[M_co, N_co, T_co],
    AbstractPermutationAccessor[M_co, N_co, T_co],
    Generic[M_co, N_co, T_co],
):

    __slots__ = ("target")
    target: AbstractAccessor[M_co, N_co, T_co]

    def __init__(self, target: AbstractAccessor[M_co, N_co, T_co]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @override
    def row_count(self) -> M_co:
        return self.target.row_count

    @property
    @override
    def col_count(self) -> N_co:
        return self.target.col_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(
            row_index,
            self.col_count - col_index - 1,
        )


@final
class Rotate090Accessor(
    AbstractMatrixAccessor[M_co, N_co, T_co],
    AbstractPermutationAccessor[M_co, N_co, T_co],
    Generic[M_co, N_co, T_co],
):

    __slots__ = ("target")
    target: AbstractAccessor[N_co, M_co, T_co]  # NOTE: Reversed dimensions!

    def __init__(self, target: AbstractAccessor[N_co, M_co, T_co]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @override
    def row_count(self) -> M_co:
        return self.target.col_count

    @property
    @override
    def col_count(self) -> N_co:
        return self.target.row_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(
            col_index,
            self.row_count - row_index - 1,
        )


@final
class Rotate180Accessor(
    AbstractVectorAccessor[M_co, N_co, T_co],
    AbstractPermutationAccessor[M_co, N_co, T_co],
    Generic[M_co, N_co, T_co],
):

    __slots__ = ("target")
    target: AbstractAccessor[M_co, N_co, T_co]

    def __init__(self, target: AbstractAccessor[M_co, N_co, T_co]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @override
    def row_count(self) -> M_co:
        return self.target.row_count

    @property
    @override
    def col_count(self) -> N_co:
        return self.target.col_count

    @override
    def vector_access(self, index: int) -> T_co:
        return self.target.vector_access(len(self) - index - 1)


@final
class Rotate270Accessor(
    AbstractMatrixAccessor[M_co, N_co, T_co],
    AbstractPermutationAccessor[M_co, N_co, T_co],
    Generic[M_co, N_co, T_co],
):

    __slots__ = ("target")
    target: AbstractAccessor[N_co, M_co, T_co]  # NOTE: Reversed dimensions!

    def __init__(self, target: AbstractAccessor[N_co, M_co, T_co]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @override
    def row_count(self) -> M_co:
        return self.target.col_count

    @property
    @override
    def col_count(self) -> N_co:
        return self.target.row_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(
            self.col_count - col_index - 1,
            row_index,
        )


ReverseAccessor: TypeAlias = Rotate180Accessor[M_co, N_co, T_co]
