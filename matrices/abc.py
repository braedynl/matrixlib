import operator
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from .rule import Rule

__all__ = ["MatrixLike"]

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class MatrixLike(Sequence[T_co], Generic[T_co, M_co, N_co], metaclass=ABCMeta):
    """Abstract base class for matrix-like objects

    A kind of "hybrid" generic sequence type that interfaces both one and two
    dimensional methods, alongside a variety of vectorized operations.
    """

    __slots__ = ()
    __match_args__ = ("array", "shape")

    def __len__(self):
        """Return the matrix's size"""
        return self.size

    @abstractmethod
    def __getitem__(self, key):
        """Return the element or sub-matrix corresponding to ``key``"""
        pass

    def __iter__(self):
        """Return an iterator over the values of the matrix in row-major order"""
        yield from self.values()

    def __reversed__(self):
        """Return an iterator over the values of the matrix in reverse
        row-major order
        """
        yield from self.values(reverse=True)

    def __contains__(self, value):
        """Return true if the matrix contains ``value``, otherwise false"""
        return any(map(lambda x: x is value or x == value, self.values()))

    @property
    @abstractmethod
    def array(self):
        """A sequence of the matrix's elements"""
        pass

    @property
    @abstractmethod
    def shape(self):
        """A sequence of the matrix's dimensions"""
        pass

    @property
    def nrows(self):
        """The number of rows"""
        return self.shape[0]

    @property
    def ncols(self):
        """The number of columns"""
        return self.shape[1]

    @property
    def size(self):
        """The product of the matrix's shape"""
        shape = self.shape
        return shape[0] * shape[1]

    @abstractmethod
    def equal(self, other):
        """Return element-wise ``a == b``"""
        pass

    @abstractmethod
    def not_equal(self, other):
        """Return element-wise ``a != b``"""
        pass

    @abstractmethod
    def logical_and(self, other):
        """Return element-wise ``logical_and(a, b)``"""
        pass

    @abstractmethod
    def logical_or(self, other):
        """Return element-wise ``logical_or(a, b)``"""
        pass

    @abstractmethod
    def logical_not(self):
        """Return element-wise ``logical_not(a)``"""
        pass

    @abstractmethod
    def transpose(self):
        """Return the matrix transpose"""
        pass

    @abstractmethod
    def flip(self, *, by=Rule.ROW):
        """Return the matrix flipped across the rows or columns"""
        pass

    @abstractmethod
    def reverse(self):
        """Return the matrix reversed"""
        pass

    def n(self, by):
        """Return the dimension corresponding to the given rule

        At the base level, this method is equivalent to `self.shape[by.value]`.
        For some matrix implementations, however, retrieving a dimension from
        this method may be faster than going through the `shape` property.

        This is the recommended method to use for all rule-based dimension
        retrievals.
        """
        return self.shape[by.value]

    def values(self, *, by=Rule.ROW, reverse=False):
        """Return an iterator that yields the matrix's items in row or
        column-major order
        """
        values = reversed if reverse else iter
        row_indices = range(self.nrows)
        col_indices = range(self.ncols)
        if by is Rule.ROW:
            for row_index in values(row_indices):
                for col_index in values(col_indices):
                    yield self[row_index, col_index]
        else:
            for col_index in values(col_indices):
                for row_index in values(row_indices):
                    yield self[row_index, col_index]

    def slices(self, *, by=Rule.ROW, reverse=False):
        """Return an iterator that yields shallow copies of each row or column"""
        values = reversed if reverse else iter
        if by is Rule.ROW:
            row_indices = range(self.nrows)
            for row_index in values(row_indices):
                yield self[row_index, :]
        else:
            col_indices = range(self.ncols)
            for col_index in values(col_indices):
                yield self[:, col_index]

    def _resolve_vector_index(self, key):
        index = operator.index(key)
        bound = self.size
        index += bound * (index < 0)
        if index < 0 or index >= bound:
            raise IndexError(f"there are {bound} items but index is {key}")
        return index

    def _resolve_matrix_index(self, key, *, by=Rule.ROW):
        index = operator.index(key)
        bound = self.n(by)
        index += bound * (index < 0)
        if index < 0 or index >= bound:
            raise IndexError(f"there are {bound} {by.handle}s but index is {key}")
        return index

    def _resolve_vector_slice(self, key):
        bound = self.size
        return range(*key.indices(bound))

    def _resolve_matrix_slice(self, key, *, by=Rule.ROW):
        bound = self.n(by)
        return range(*key.indices(bound))
