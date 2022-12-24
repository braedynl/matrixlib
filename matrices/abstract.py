from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from .utilities import Rule, checked_map, compare

__all__ = ["MatrixLike"]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class MatrixLike(Sequence[T_co], Generic[T_co, M_co, N_co], metaclass=ABCMeta):

    def __eq__(self, other):
        """Return true if element-wise `a is b or a == b` is true for all
        element pairs, otherwise false
        """
        if self is other:
            return True
        if isinstance(other, MatrixLike):
            return all(checked_map(lambda x, y: x is y or x == y, self, other))
        return NotImplemented

    def __lt__(self, other):
        """Return true if lexicographic `a < b`, otherwise false"""
        return compare(self, other) < 0

    def __le__(self, other):
        """Return true if lexicographic `a <= b`, otherwise false"""
        return compare(self, other) <= 0

    def __gt__(self, other):
        """Return true if lexicographic `a > b`, otherwise false"""
        return compare(self, other) > 0

    def __ge__(self, other):
        """Return true if lexicographic `a >= b`, otherwise false"""
        return compare(self, other) >= 0

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
    def __divmod__(self, other):
        """Return element-wise `divmod(a, b)`"""
        pass

    @abstractmethod
    def __pow__(self, other):
        """Return element-wise `a ** b`"""
        pass

    @abstractmethod
    def __lshift__(self, other):
        """Return element-wise `a << b`"""
        pass

    @abstractmethod
    def __rshift__(self, other):
        """Return element-wise `a >> b`"""
        pass

    @abstractmethod
    def __and__(self, other):
        """Return element-wise `a & b`"""
        pass

    @abstractmethod
    def __xor__(self, other):
        """Return element-wise `a ^ b`"""
        pass

    @abstractmethod
    def __or__(self, other):
        """Return element-wise `a | b`"""
        pass

    @abstractmethod
    def __matmul__(self, other):
        """Return the matrix product"""
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
        """Return element-wise `~a`"""
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
    def equal(self, other):
        """Return element-wise `a == b`"""
        pass

    @abstractmethod
    def not_equal(self, other):
        """Return element-wise `a != b`"""
        pass

    @abstractmethod
    def lesser(self, other):
        """Return element-wise `a < b`"""
        pass

    @abstractmethod
    def lesser_equal(self, other):
        """Return element-wise `a <= b`"""
        pass

    @abstractmethod
    def greater(self, other):
        """Return element-wise `a > b`"""
        pass

    @abstractmethod
    def greater_equal(self, other):
        """Return element-wise `a >= b`"""
        pass

    @abstractmethod
    def logical_and(self, other):
        """Return element-wise `logical_and(a, b)`"""
        pass

    @abstractmethod
    def logical_or(self, other):
        """Return element-wise `logical_or(a, b)`"""
        pass

    @abstractmethod
    def logical_not(self):
        """Return element-wise `logical_not(a)`"""
        pass

    @abstractmethod
    def conjugate(self):
        """Return element-wise `conjugate(a)`"""
        pass

    def slices(self, *, by=Rule.ROW):
        """Return an iterator that yields shallow copies of each row or column"""
        if by is Rule.ROW:
            for i in range(self.nrows):
                yield self[i, :]
        else:
            for j in range(self.ncols):
                yield self[:, j]

    @abstractmethod
    def transpose(self):
        """Return the transpose of the matrix"""
        pass
