import enum
from abc import ABCMeta, abstractmethod
from collections.abc import Collection, Sequence
from enum import Flag
from typing import Generic, TypeVar

from .rule import Rule

__all__ = ["ShapeLike", "MatrixLike"]


NRows_co = TypeVar("NRows_co", bound=int, covariant=True)
NCols_co = TypeVar("NCols_co", bound=int, covariant=True)

class ShapeLike(Collection[NRows_co | NCols_co], Generic[NRows_co, NCols_co], metaclass=ABCMeta):

    __match_args__ = ("nrows", "ncols")

    def __eq__(self, other):
        """Return true if the two shapes are equal, otherwise false"""
        if isinstance(other, ShapeLike):
            return (
                self[0] == other[0]
                and
                self[1] == other[1]
            )
        return NotImplemented

    def __len__(self):
        """Return literal 2"""
        return 2

    @abstractmethod
    def __getitem__(self, key):
        """Return the dimension corresponding to `key`"""
        pass

    def __iter__(self):
        """Return an iterator over the dimensions of the shape"""
        yield self[0]
        yield self[1]

    def __reversed__(self):
        """Return a reversed iterator over the dimensions of the shape"""
        yield self[1]
        yield self[0]

    def __contains__(self, value):
        """Return true if the shape contains `value`, otherwise false"""
        return self[0] == value or self[1] == value

    @property
    def nrows(self):
        """The first dimension of the shape"""
        return self[0]

    @property
    def ncols(self):
        """The second dimension of the shape"""
        return self[1]


class Ordering(Flag):
    LESSER  = enum.auto()
    EQUAL   = enum.auto()
    GREATER = enum.auto()


def matrix_ordering(a, b, /):
    if (u := a.shape) != (v := b.shape):
        raise ValueError(f"shape {u} is incompatible with operand shape {v}")
    for x, y in zip(a, b):
        if x < y:
            return Ordering.LESSER
        if x > y:
            return Ordering.GREATER
    return Ordering.EQUAL


def matrix_map(func, a, b, /):
    if (u := a.shape) != (v := b.shape):
        raise ValueError(f"shape {u} is incompatible with operand shape {v}")
    return map(func, a, b)


T_co = TypeVar("T_co", covariant=True)

class MatrixLike(Sequence[T_co], Generic[T_co, NRows_co, NCols_co], metaclass=ABCMeta):

    def __eq__(self, other):
        """Return true if element-wise `a is b or a == b` is true for all
        element pairs, otherwise false
        """
        return all(matrix_map(lambda x, y: x is y or x == y, self, other))

    def __lt__(self, other):
        """Return true if lexicographic `a < b`, otherwise false"""
        return matrix_ordering(self, other) is Ordering.LESSER

    def __le__(self, other):
        """Return true if lexicographic `a <= b`, otherwise false"""
        return matrix_ordering(self, other) in Ordering.LESSER | Ordering.EQUAL

    def __gt__(self, other):
        """Return true if lexicographic `a > b`, otherwise false"""
        return matrix_ordering(self, other) is Ordering.GREATER

    def __ge__(self, other):
        """Return true if lexicographic `a >= b`, otherwise false"""
        return matrix_ordering(self, other) in Ordering.GREATER | Ordering.EQUAL

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
        keys = reversed(range(self.size))
        yield from map(self.__getitem__, keys)

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
