from __future__ import annotations

__all__ = [
    "MatrixAccessor",
    "RowVectorAccessor",
    "ColVectorAccessor",
    "ValueAccessor",
    "IdentityAccessor",
]

import itertools
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Literal, Self, cast, final, override

from .abstracts import (AbstractAccessor, AbstractMatrixAccessor,
                        AbstractVectorAccessor)


class AbstractDefaultAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractVectorAccessor[M, N, T], metaclass=ABCMeta):

    __slots__ = ()

    def __hash__(self) -> int:
        return hash((self.array, self.shape))

    @override
    def __len__(self) -> int:
        return len(self.array)

    @override
    def __iter__(self) -> Iterator[T]:
        return iter(self.array)

    @override
    def __reversed__(self) -> Iterator[T]:
        return reversed(self.array)

    @override
    def __contains__(self, value: object) -> bool:
        return value in self.array

    @property
    @abstractmethod
    def array(self) -> tuple[T, ...]:
        raise NotImplementedError

    @override
    def to_tuple(self) -> tuple[T, ...]:
        return self.array

    @override
    def vector_access(self, index: int) -> T:
        return self.array[index]


@final
class MatrixAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractDefaultAccessor[M, N, T]):

    __slots__ = (
        "array",  # pyright: ignore[reportIncompatibleMethodOverride]
        "shape",  # pyright: ignore[reportIncompatibleMethodOverride]
    )
    array: tuple[T, ...]
    shape: tuple[M, N]

    def __new__(cls, array: tuple[T, ...], shape: tuple[M, N]) -> Self:
        self = super(MatrixAccessor, cls).__new__(cls)
        self.array = array
        self.shape = shape
        return self

    def __reduce__(self) -> tuple[object, ...]:
        return (self.__class__, (self.array, self.shape))

    @property
    @override
    def row_count(self) -> M:
        return self.shape[0]

    @property
    @override
    def col_count(self) -> N:
        return self.shape[1]


@final
class RowVectorAccessor[
    N: int = int,
    T: object = object,
](AbstractDefaultAccessor[Literal[1], N, T]):

    __slots__ = (
        "array",  # pyright: ignore[reportIncompatibleMethodOverride]
    )
    array: tuple[T, ...]
    row_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __new__(cls, array: tuple[T, ...]) -> Self:
        self = super(RowVectorAccessor, cls).__new__(cls)
        self.array = array
        return self

    def __reduce__(self) -> tuple[object, ...]:
        return (self.__class__, (self.array,))

    @property
    @override
    def col_count(self) -> N:
        return cast(N, len(self.array))


@final
class ColVectorAccessor[
    M: int = int,
    T: object = object,
](AbstractDefaultAccessor[M, Literal[1], T]):

    __slots__ = (
        "array",  # pyright: ignore[reportIncompatibleMethodOverride]
    )
    array: tuple[T, ...]
    col_count: Literal[1] = 1  # pyright: ignore[reportIncompatibleMethodOverride]

    def __new__(cls, array: tuple[T, ...]) -> Self:
        self = super(ColVectorAccessor, cls).__new__(cls)
        self.array = array
        return self

    def __reduce__(self) -> tuple[object, ...]:
        return (self.__class__, (self.array,))

    @property
    @override
    def row_count(self) -> M:
        return cast(M, len(self.array))


@final
class ValueAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractAccessor[M, N, T]):

    __slots__ = (
        "value",
        "shape",  # pyright: ignore[reportIncompatibleMethodOverride]
    )
    value: T
    shape: tuple[M, N]

    def __new__(cls, value: T, shape: tuple[M, N]) -> Self:
        self = super(ValueAccessor, cls).__new__(cls)
        self.value = value
        self.shape = shape
        return self

    def __hash__(self) -> int:
        return hash((self.value, self.shape))

    def __reduce__(self) -> tuple[object, ...]:
        return (self.__class__, (self.value, self.shape))

    @override
    def __iter__(self) -> Iterator[T]:
        return itertools.repeat(self.value, times=len(self))

    __reversed__ = override(__iter__)

    @override
    def __contains__(self, value: object) -> bool:
        return value is self.value or value == self.value

    @property
    @override
    def row_count(self) -> M:
        return self.shape[0]

    @property
    @override
    def col_count(self) -> N:
        return self.shape[1]

    @override
    def vector_access(self, index: int) -> T:
        return self.value

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T:
        return self.value


@final
class IdentityAccessor[
    M: int = int,
    N: int = int,
    T: object = object,
](AbstractMatrixAccessor[M, N, T]):

    __slots__ = (
        "non_zero_value",
        "zero_value",
        "shape",  # pyright: ignore[reportIncompatibleMethodOverride]
    )
    non_zero_value: T
    zero_value: T
    shape: tuple[M, N]

    def __new__(cls, non_zero_value: T, shape: tuple[M, N], zero_value: T = 0) -> Self:
        self = super(IdentityAccessor, cls).__new__(cls)
        self.non_zero_value = non_zero_value
        self.shape = shape
        self.zero_value = zero_value
        return self

    def __hash__(self) -> int:
        return hash((self.non_zero_value, self.shape, self.zero_value))

    def __reduce__(self) -> tuple[object, ...]:
        return (self.__class__, (self.non_zero_value, self.shape, self.zero_value))

    @property
    @override
    def row_count(self) -> M:
        return self.shape[0]

    @property
    @override
    def col_count(self) -> N:
        return self.shape[1]

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T:
        return self.non_zero_value if row_index == col_index else self.zero_value
