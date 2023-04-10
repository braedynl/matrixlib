from __future__ import annotations

import operator
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Generic, Literal, SupportsIndex, TypeVar

__all__ = [
    "Sieve",
    "VectorSieve",
    "MatrixSieve",
]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class Sieve(Generic[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        """Return true if two sieves are equivalent, otherwise false"""
        if self is other:
            return True
        if isinstance(other, Sieve):
            return (
                self.shape == other.shape
                and
                all(map(operator.eq, self, other))
            )
        return NotImplemented

    def __len__(self) -> int:
        """Return the shape's product"""
        return self.nrows * self.ncols

    @abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        """Return an iterator over the sieve's values in row-major order"""
        raise NotImplementedError

    @abstractmethod
    def __reversed__(self) -> Iterator[T_co]:
        """Return an iterator over the sieve's values in reverse row-major
        order
        """
        raise NotImplementedError

    def __contains__(self, value: object) -> bool:
        """Return true if the sieve contains ``value``, otherwise false"""
        for val in self:
            if val == value:
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

    def sieve(self) -> tuple[T_co, ...]:
        """Collect and return all sieve values as a ``tuple``, aligned in
        row-major order
        """
        return tuple(self)

    @abstractmethod
    def vector_sieve(self, index: int) -> T_co:
        """Return the value at ``index``"""
        raise NotImplementedError

    @abstractmethod
    def matrix_sieve(self, row_index: int, col_index: int) -> T_co:
        """Return the value at ``row_index``, ``col_index``"""
        raise NotImplementedError

    def resolve_vector_index(self, key: SupportsIndex) -> int:
        """Return ``key`` resolved with respect to the sieve's size"""
        bound = len(self)
        try:
            index = resolve_index(key, bound)
        except IndexError:
            raise IndexError(f"there are {bound} values, but index is {key}") from None
        else:
            return index

    def resolve_vector_slice(self, key: slice) -> range:
        """Return ``key`` resolved with respect to the sieve's size"""
        bound = len(self)
        return resolve_slice(key, bound)

    def resolve_matrix_index(self, key: SupportsIndex, *, axis: Literal[0, 1]) -> int:
        """Return ``key`` resolved with respect to the ``axis`` dimension"""
        bound = self.shape[axis]
        try:
            index = resolve_index(key, bound)
        except IndexError:
            handle = ("row", "column")[axis]
            raise IndexError(f"there are {bound} {handle}s, but index is {key}") from None
        else:
            return index

    def resolve_matrix_slice(self, key: slice, *, axis: Literal[0, 1]) -> range:
        """Return ``key`` resolved with respect to the ``axis`` dimension"""
        bound = self.shape[axis]
        return resolve_slice(key, bound)


class VectorSieve(Sieve[M_co, N_co, T_co], metaclass=ABCMeta):
    """Sub-class of ``Sieve`` that pipes calls from ``matrix_sieve()`` to
    ``vector_sieve()``
    """

    __slots__ = ()

    def __iter__(self) -> Iterator[T_co]:
        indices = range(len(self))
        for index in indices:
            yield self.vector_sieve(index)

    def __reversed__(self) -> Iterator[T_co]:
        indices = range(len(self) - 1, -1, -1)
        for index in indices:
            yield self.vector_sieve(index)

    def matrix_sieve(self, row_index: int, col_index: int) -> T_co:
        index = row_index * self.ncols + col_index
        return self.vector_sieve(index)


class MatrixSieve(Sieve[M_co, N_co, T_co], metaclass=ABCMeta):
    """Sub-class of ``Sieve`` that pipes calls from ``vector_sieve()`` to
    ``matrix_sieve()``
    """

    __slots__ = ()

    def __iter__(self) -> Iterator[T_co]:
        row_indices = range(self.nrows)
        col_indices = range(self.ncols)
        for row_index in row_indices:
            for col_index in col_indices:
                yield self.matrix_sieve(row_index, col_index)

    def __reversed__(self) -> Iterator[T_co]:
        row_indices = range(self.nrows - 1, -1, -1)
        col_indices = range(self.ncols - 1, -1, -1)
        for row_index in row_indices:
            for col_index in col_indices:
                yield self.matrix_sieve(row_index, col_index)

    def vector_sieve(self, index: int) -> T_co:
        row_index, col_index = divmod(index, self.ncols)
        return self.matrix_sieve(row_index, col_index)


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
    start, stop, step = key.indices(bound)
    return range(start, stop, step)
