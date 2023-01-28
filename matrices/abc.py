import operator
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from .utilities import Rule

__all__ = ["MatrixLike"]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


def lexicographic_compare(a, b):
    if a is b:
        return 0
    if (u := a.shape) != (v := b.shape):
        raise ValueError(f"incompatible shapes {u}, {v}")
    for x, y in zip(a, b):
        if x < y:
            return -1
        if x > y:
            return 1
    return 0


class MatrixLike(Sequence[T_co], Generic[T_co, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()

    def __lt__(self, other):
        """Return true if lexicographic `a < b`, otherwise false"""
        if isinstance(other, MatrixLike):
            return lexicographic_compare(self, other) < 0
        return NotImplemented

    def __le__(self, other):
        """Return true if lexicographic `a <= b`, otherwise false"""
        if isinstance(other, MatrixLike):
            return lexicographic_compare(self, other) <= 0
        return NotImplemented

    def __eq__(self, other):
        """Return true if lexicographic `a == b`, otherwise false"""
        if isinstance(other, MatrixLike):
            return lexicographic_compare(self, other) == 0
        return NotImplemented

    def __ne__(self, other):
        """Return true if lexicographic `a != b`, otherwise false"""
        if isinstance(other, MatrixLike):
            return lexicographic_compare(self, other) != 0
        return NotImplemented

    def __gt__(self, other):
        """Return true if lexicographic `a > b`, otherwise false"""
        if isinstance(other, MatrixLike):
            return lexicographic_compare(self, other) > 0
        return NotImplemented

    def __ge__(self, other):
        """Return true if lexicographic `a >= b`, otherwise false"""
        if isinstance(other, MatrixLike):
            return lexicographic_compare(self, other) >= 0
        return NotImplemented

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
        return self.shape[0]

    @property
    def ncols(self):
        """The matrix's number of columns"""
        return self.shape[1]

    @property
    def size(self):
        """The product of the matrix's number of rows and columns"""
        return self.nrows * self.ncols

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
        """Return element-wise `bool(a and b)`"""
        pass

    @abstractmethod
    def logical_or(self, other):
        """Return element-wise `bool(a or b)`"""
        pass

    @abstractmethod
    def logical_not(self):
        """Return element-wise `not a`"""
        pass

    @abstractmethod
    def conjugate(self):
        """Return element-wise `a.conjugate()`"""
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

    def _resolve_index(self, key, *, by=None):
        """Validate, sanitize, and return an index `key` as a built-in `int`
        with respect to the matrix's number of items, rows, or columns

        Raises `IndexError` if the `key` is out of range.

        This method uses the extended `Rule` convention, where `by=None` (the
        default) corresponds to a "flattened" interpretation of the method.
        """
        bound = self.size if by is None else self.shape[by.value]
        index = operator.index(key)
        index += bound * (index < 0)
        if index < 0 or index >= bound:
            handle = "item" if by is None else by.handle
            raise IndexError(f"there are {bound} {handle}s but index is {key}")
        return index

    def _resolve_slice(self, key, *, by=None):
        """Validate, sanitize, and return a slice `key` as an iterable of
        built-in `int`s with respect to the matrix's number of items, rows, or
        columns

        This method uses the extended `Rule` convention, where `by=None` (the
        default) corresponds to a "flattened" interpretation of the method.
        """
        bound = self.size if by is None else self.shape[by.value]
        return range(*key.indices(bound))

    def _permute_index(self, index):
        """Return a singular or paired `index` as its one-dimensional
        equivalent

        Typically, this method is called post-resolution, and
        pre-`array.__getitem__()`, where `array` is the one-dimensional
        sequence wrapped by the matrix implementation. If your implementation
        does not wrap a one-dimensional sequence, this method may not be
        useful.
        """
        if isinstance(index, tuple):
            return index[0] * self.ncols + index[1]
        return index
