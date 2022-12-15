from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from .rule import Rule
from .utilities import Ordering, apply, order

__all__ = ["MatrixLike"]

DTypeT_co = TypeVar("DTypeT_co", covariant=True)
NRowsT_co = TypeVar("NRowsT_co", covariant=True, bound=int)
NColsT_co = TypeVar("NColsT_co", covariant=True, bound=int)


class MatrixLike(Sequence[DTypeT_co], Generic[DTypeT_co, NRowsT_co, NColsT_co], metaclass=ABCMeta):

    def __eq__(self, other):
        """Return true if element-wise `a is b or a == b` is true for all
        element pairs, otherwise false
        """
        if isinstance(other, MatrixLike):
            return all(apply(lambda x, y: x is y or x == y, self, other))
        return NotImplemented

    def __lt__(self, other):
        """Return true if lexicographic `a < b`, otherwise false"""
        return order(self, other) is Ordering.LESSER

    def __le__(self, other):
        """Return true if lexicographic `a <= b`, otherwise false"""
        return order(self, other) in Ordering.LESSER | Ordering.EQUAL

    def __gt__(self, other):
        """Return true if lexicographic `a > b`, otherwise false"""
        return order(self, other) is Ordering.GREATER

    def __ge__(self, other):
        """Return true if lexicographic `a >= b`, otherwise false"""
        return order(self, other) in Ordering.GREATER | Ordering.EQUAL

    def __len__(self):
        """Return the matrix's size"""
        return self.size

    @abstractmethod
    def __getitem__(self, key):
        """Return the element or sub-matrix corresponding to `key`"""
        pass

    def __iter__(self):
        """Return an iterator over the values of the matrix in row-major order"""
        keys = range(self.size)
        yield from map(self.__getitem__, keys)

    def __reversed__(self):
        """Return an iterator over the values of the matrix in reverse
        row-major order
        """
        keys = range(self.size)
        yield from map(self.__getitem__, reversed(keys))

    def __contains__(self, value):
        """Return true if the matrix contains `value`, otherwise false"""
        return any(map(lambda x: x is value or x == value, self))

    @abstractmethod
    def __add__(self, other):
        """Return element-wise `a + b`"""
        pass

    @abstractmethod
    def __sub__(self, other):
        """Return element-wise `a - b`"""
        pass

    @abstractmethod
    def __mul__(self, other):
        """Return element-wise `a * b`"""
        pass

    @abstractmethod
    def __truediv__(self, other):
        """Return element-wise `a / b`"""
        pass

    @abstractmethod
    def __floordiv__(self, other):
        """Return element-wise `a // b`"""
        pass

    @abstractmethod
    def __mod__(self, other):
        """Return element-wise `a % b`"""
        pass

    @abstractmethod
    def __pow__(self, other):
        """Return element-wise `a ** b`"""
        pass

    @abstractmethod
    def __and__(self, other):
        """Return element-wise `logical_and(a, b)`"""
        pass

    @abstractmethod
    def __or__(self, other):
        """Return element-wise `logical_or(a, b)`"""
        pass

    @abstractmethod
    def __xor__(self, other):
        """Return element-wise `logical_xor(a, b)`"""
        pass

    @abstractmethod
    def __neg__(self):
        """Return element-wise `-a`"""
        pass

    @abstractmethod
    def __pos__(self):
        """Return element-wise `+a`"""
        pass

    @abstractmethod
    def __abs__(self):
        """Return element-wise `abs(a)`"""
        pass

    @abstractmethod
    def __invert__(self):
        """Return element-wise `logical_not(a)`"""
        pass

    @property
    @abstractmethod
    def shape(self):
        """A collection of the matrix's dimensions"""
        pass

    @property
    def nrows(self):
        """The matrix's number of rows"""
        return self.shape.nrows

    @property
    def ncols(self):
        """The matrix's number of columns"""
        return self.shape.ncols

    @property
    def size(self):
        """The product of the matrix's number of rows and columns"""
        nrows, ncols = self.shape
        return nrows * ncols

    @abstractmethod
    def eq(self, other):
        """Return element-wise `a == b`"""
        pass

    @abstractmethod
    def ne(self, other):
        """Return element-wise `a != b`"""
        pass

    @abstractmethod
    def lt(self, other):
        """Return element-wise `a < b`"""
        pass

    @abstractmethod
    def le(self, other):
        """Return element-wise `a <= b`"""
        pass

    @abstractmethod
    def gt(self, other):
        """Return element-wise `a > b`"""
        pass

    @abstractmethod
    def ge(self, other):
        """Return element-wise `a >= b`"""
        pass

    @abstractmethod
    def conjugate(self):
        """Return element-wise `conjugate(a)`"""
        pass

    @abstractmethod
    def slices(self, *, by=Rule.ROW):
        """Return an iterator that yields shallow copies of each row or column"""
        pass
