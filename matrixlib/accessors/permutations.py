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

T_co = TypeVar("T_co", covariant=True)


class PermutationAccessor(AbstractAccessor[T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __hash__(self) -> int:
        return hash(self.target)

    @property
    @abstractmethod
    def target(self) -> AbstractAccessor[T_co]:
        """The permuted accessor"""
        raise NotImplementedError


@final
class TransposeAccessor(
    AbstractMatrixAccessor[T_co],
    PermutationAccessor[T_co],
    Generic[T_co],
):

    __slots__ = ("target")
    target: AbstractAccessor[T_co]

    def __init__(self, target: AbstractAccessor[T_co]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"TransposeAccessor(target={self.target!r})"

    @property
    @override
    def row_count(self) -> int:
        return self.target.col_count

    @property
    @override
    def col_count(self) -> int:
        return self.target.row_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(col_index, row_index)


@final
class RowFlipAccessor(
    AbstractMatrixAccessor[T_co],
    PermutationAccessor[T_co],
    Generic[T_co],
):

    __slots__ = ("target")
    target: AbstractAccessor[T_co]

    def __init__(self, target: AbstractAccessor[T_co]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"RowFlipAccessor(target={self.target!r})"

    @property
    @override
    def row_count(self) -> int:
        return self.target.row_count

    @property
    @override
    def col_count(self) -> int:
        return self.target.col_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(
            self.row_count - row_index - 1,
            col_index,
        )


@final
class ColFlipAccessor(
    AbstractMatrixAccessor[T_co],
    PermutationAccessor[T_co],
    Generic[T_co],
):

    __slots__ = ("target")
    target: AbstractAccessor[T_co]

    def __init__(self, target: AbstractAccessor[T_co]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"ColFlipAccessor(target={self.target!r})"

    @property
    @override
    def row_count(self) -> int:
        return self.target.row_count

    @property
    @override
    def col_count(self) -> int:
        return self.target.col_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(
            row_index,
            self.col_count - col_index - 1,
        )


@final
class Rotate090Accessor(
    AbstractMatrixAccessor[T_co],
    PermutationAccessor[T_co],
    Generic[T_co],
):

    __slots__ = ("target")
    target: AbstractAccessor[T_co]

    def __init__(self, target: AbstractAccessor[T_co]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"Rotate090Accessor(target={self.target!r})"

    @property
    @override
    def row_count(self) -> int:
        return self.target.col_count

    @property
    @override
    def col_count(self) -> int:
        return self.target.row_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(
            col_index,
            self.row_count - row_index - 1,
        )


@final
class Rotate180Accessor(
    AbstractVectorAccessor[T_co],
    PermutationAccessor[T_co],
    Generic[T_co],
):

    __slots__ = ("target")
    target: AbstractAccessor[T_co]

    def __init__(self, target: AbstractAccessor[T_co]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"Rotate180Accessor(target={self.target!r})"

    @property
    @override
    def row_count(self) -> int:
        return self.target.row_count

    @property
    @override
    def col_count(self) -> int:
        return self.target.col_count

    @override
    def vector_access(self, index: int) -> T_co:
        return self.target.vector_access(len(self) - index - 1)


@final
class Rotate270Accessor(
    AbstractMatrixAccessor[T_co],
    PermutationAccessor[T_co],
    Generic[T_co],
):

    __slots__ = ("target")
    target: AbstractAccessor[T_co]

    def __init__(self, target: AbstractAccessor[T_co]) -> None:
        self.target = target  # pyright: ignore[reportIncompatibleMethodOverride]

    def __repr__(self) -> str:
        return f"Rotate270Accessor(target={self.target!r})"

    @property
    @override
    def row_count(self) -> int:
        return self.target.col_count

    @property
    @override
    def col_count(self) -> int:
        return self.target.row_count

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(
            self.col_count - col_index - 1,
            row_index,
        )


ReverseAccessor: TypeAlias = Rotate180Accessor[T_co]
