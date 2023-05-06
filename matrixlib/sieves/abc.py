from __future__ import annotations

import operator
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Generic, Literal, SupportsIndex, TypeVar

from mypy_extensions import mypyc_attr

__all__ = [
    "BaseAccessor",
    "BaseVectorAccessor",
    "BaseMatrixAccessor",
]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


@mypyc_attr(allow_interpreted_subclasses=True)
class BaseAccessor(Generic[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        """Return true if the two accessors are equivalent, otherwise false"""
        if self is other:
            return True
        if isinstance(other, BaseAccessor):
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
        return self.nrows * self.ncols

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
        for val in self:
            if val is value or val == value:
                return True
        return False

    @property
    def shape(self) -> tuple[M_co, N_co]:
        """The number of rows and columns as a ``tuple``"""
        return (self.nrows, self.ncols)

    @property
    @abstractmethod
    def nrows(self) -> M_co:
        """The number of rows"""
        raise NotImplementedError

    @property
    @abstractmethod
    def ncols(self) -> N_co:
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

    def resolve_matrix_index(self, key: SupportsIndex, axis: Literal[0, 1]) -> int:
        """Return ``key`` resolved with respect to the ``axis`` dimension"""
        bound = self.shape[axis]
        try:
            index = resolve_index(key, bound)
        except IndexError:
            handle = ("row", "column")[axis]
            raise IndexError(f"there are {bound} {handle}s, but index is {key}") from None
        else:
            return index

    def resolve_matrix_slice(self, key: slice, axis: Literal[0, 1]) -> range:
        """Return ``key`` resolved with respect to the ``axis`` dimension"""
        bound = self.shape[axis]
        return resolve_slice(key, bound)


@mypyc_attr(allow_interpreted_subclasses=True)
class BaseVectorAccessor(BaseAccessor[M_co, N_co, T_co], metaclass=ABCMeta):
    """Sub-class of ``BaseAccessor`` that pipes calls from ``matrix_access()``
    to ``vector_access()``
    """

    __slots__ = ()

    def __iter__(self) -> Iterator[T_co]:
        indices = range(len(self))
        for index in indices:
            yield self.vector_access(index)

    def __reversed__(self) -> Iterator[T_co]:
        indices = range(len(self) - 1, -1, -1)
        for index in indices:
            yield self.vector_access(index)

    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        index = row_index * self.ncols + col_index
        return self.vector_access(index)


@mypyc_attr(allow_interpreted_subclasses=True)
class BaseMatrixAccessor(BaseAccessor[M_co, N_co, T_co], metaclass=ABCMeta):
    """Sub-class of ``BaseAccessor`` that pipes calls from ``vector_access()``
    to ``matrix_access()``
    """

    __slots__ = ()

    def __iter__(self) -> Iterator[T_co]:
        row_indices = range(self.nrows)
        col_indices = range(self.ncols)
        for row_index in row_indices:
            for col_index in col_indices:
                yield self.matrix_access(row_index, col_index)

    def __reversed__(self) -> Iterator[T_co]:
        row_indices = range(self.nrows - 1, -1, -1)
        col_indices = range(self.ncols - 1, -1, -1)
        for row_index in row_indices:
            for col_index in col_indices:
                yield self.matrix_access(row_index, col_index)

    def vector_access(self, index: int) -> T_co:
        row_index, col_index = divmod(index, self.ncols)
        return self.matrix_access(row_index, col_index)


def resolve_index(key: SupportsIndex, bound: int) -> int:
    """Return ``key`` as an index with respect to ``bound``

    Raises an empty ``IndexError`` if ``key`` is out of range.
    """
    index = operator.index(key)
    if index < 0:
        index += bound
    if index < 0 or index >= bound:
        raise IndexError
    return index


def resolve_slice(key: slice, bound: int) -> range:
    """Return ``key`` as a range of indices with respect to ``bound``"""
    return range(*key.indices(bound))
