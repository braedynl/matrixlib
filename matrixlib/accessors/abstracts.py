from __future__ import annotations

__all__ = [
    "AbstractAccessor",
    "AbstractVectorAccessor",
    "AbstractMatrixAccessor",
]

import operator
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Generic, SupportsIndex, TypeVar

from typing_extensions import override

from .rule import Rule

T_co = TypeVar("T_co", covariant=True)


class AbstractAccessor(Generic[T_co], metaclass=ABCMeta):

    __slots__ = ()

    @override
    def __eq__(self, other: object) -> bool:
        """Return true if the two accessors are equivalent, otherwise false"""
        if self is other:
            return True
        if isinstance(other, AbstractAccessor):
            if self.shape != other.shape:
                return False
            for x, y in zip(self, other):
                if x is y or x == y:
                    continue
                return False
            return True
        return NotImplemented

    def __len__(self) -> int:
        """Return the shape's product"""
        return self.row_count * self.col_count

    @abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        """Return an iterator over the accessor's values in row-major order"""
        raise NotImplementedError

    @abstractmethod
    def __reversed__(self) -> Iterator[T_co]:
        """Return an iterator over the accessor's values in reverse row-major
        order
        """
        raise NotImplementedError

    def __contains__(self, value: object) -> bool:
        """Return true if the accessor contains ``value``, otherwise false"""
        for x in self:
            if x is value or x == value:
                return True
        return False

    @property
    def shape(self) -> tuple[int, int]:
        """The number of rows and columns as a ``tuple``"""
        return (self.row_count, self.col_count)

    @property
    @abstractmethod
    def row_count(self) -> int:
        """The number of rows"""
        raise NotImplementedError

    @property
    @abstractmethod
    def col_count(self) -> int:
        """The number of columns"""
        raise NotImplementedError

    def collect(self) -> tuple[T_co, ...]:
        """Gather and return all accessor values as a ``tuple``, aligned in
        row-major order
        """
        return tuple(self)

    @abstractmethod
    def vector_access(self, index: int) -> T_co:
        """Return the value at ``index``"""
        raise NotImplementedError

    @abstractmethod
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        """Return the value at ``row_index``, ``col_index``"""
        raise NotImplementedError

    def resolve_vector_index(self, key: SupportsIndex) -> int:
        """Return ``key`` resolved with respect to the accessor's size"""
        bound = len(self)
        try:
            index = resolve_index(key, bound)
        except IndexError:
            raise IndexError(f"there are {bound} values, but index is {key}") from None
        else:
            return index

    def resolve_vector_slice(self, key: slice) -> range:
        """Return ``key`` resolved with respect to the accessor's size"""
        bound = len(self)
        return resolve_slice(key, bound)

    def resolve_matrix_index(self, key: SupportsIndex, *, by: Rule) -> int:
        """Return ``key`` resolved with respect to the given rule"""
        bound = self.shape[by]
        try:
            index = resolve_index(key, bound)
        except IndexError:
            raise IndexError(f"there are {bound} {by.handle}s, but index is {key}") from None
        else:
            return index

    def resolve_matrix_slice(self, key: slice, *, by: Rule) -> range:
        """Return ``key`` resolved with respect to the given rule"""
        bound = self.shape[by]
        return resolve_slice(key, bound)


class AbstractVectorAccessor(AbstractAccessor[T_co], metaclass=ABCMeta):
    """Sub-class of ``AbstractAccessor`` with preference for vector access"""

    __slots__ = ()

    @override
    def __iter__(self) -> Iterator[T_co]:
        indices = range(len(self))
        for index in indices:
            yield self.vector_access(index)

    @override
    def __reversed__(self) -> Iterator[T_co]:
        indices = range(len(self) - 1, -1, -1)
        for index in indices:
            yield self.vector_access(index)

    @override
    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        index = row_index * self.col_count + col_index
        return self.vector_access(index)


class AbstractMatrixAccessor(AbstractAccessor[T_co], metaclass=ABCMeta):
    """Sub-class of ``AbstractAccessor`` with preference for matrix access"""

    __slots__ = ()

    @override
    def __iter__(self) -> Iterator[T_co]:
        row_indices = range(self.row_count)
        col_indices = range(self.col_count)
        for row_index in row_indices:
            for col_index in col_indices:
                yield self.matrix_access(row_index, col_index)

    @override
    def __reversed__(self) -> Iterator[T_co]:
        row_indices = range(self.row_count - 1, -1, -1)
        col_indices = range(self.col_count - 1, -1, -1)
        for row_index in row_indices:
            for col_index in col_indices:
                yield self.matrix_access(row_index, col_index)

    @override
    def vector_access(self, index: int) -> T_co:
        row_index, col_index = divmod(index, self.col_count)
        return self.matrix_access(row_index, col_index)


def resolve_index(key: SupportsIndex, bound: int) -> int:
    """Return ``key`` as an index with respect to ``bound``

    Raises an empty ``IndexError`` if ``key`` is out of range.
    """
    index = operator.index(key)
    if index < 0:
        index += bound
        if index < 0:
            raise IndexError
        return index
    if index >= bound:
        raise IndexError
    return index


def resolve_slice(key: slice, bound: int) -> range:
    """Return ``key`` as a range of indices with respect to ``bound``"""
    return range(*key.indices(bound))
