import operator
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from .utilities import Rule

__all__ = ["MatrixLike"]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class MatrixLike(Sequence[T_co], Generic[T_co, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()

    def __lt__(self, other):
        """Return true if lexicographic `a < b`, otherwise false"""
        if isinstance(other, MatrixLike):
            return self.compare(other) < 0
        return NotImplemented

    def __le__(self, other):
        """Return true if lexicographic `a <= b`, otherwise false"""
        if isinstance(other, MatrixLike):
            return self.compare(other) <= 0
        return NotImplemented

    def __eq__(self, other):
        """Return true if lexicographic `a == b`, otherwise false"""
        if isinstance(other, MatrixLike):
            return self.compare(other) == 0
        return NotImplemented

    def __ne__(self, other):
        """Return true if lexicographic `a != b`, otherwise false"""
        if isinstance(other, MatrixLike):
            return self.compare(other) != 0
        return NotImplemented

    def __gt__(self, other):
        """Return true if lexicographic `a > b`, otherwise false"""
        if isinstance(other, MatrixLike):
            return self.compare(other) > 0
        return NotImplemented

    def __ge__(self, other):
        """Return true if lexicographic `a >= b`, otherwise false"""
        if isinstance(other, MatrixLike):
            return self.compare(other) >= 0
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
        yield from self.items()

    def __reversed__(self):
        """Return an iterator over the values of the matrix in reverse
        row-major order
        """
        yield from self.items(reverse=True)

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

    @abstractmethod
    def transpose(self):
        """Return the transpose of the matrix"""
        pass

    def compare(self, other):
        """Return literal -1, 0, or 1 if lexicographic `a < b`, `a == b`, or
        `a > b`, respectively
        """
        if self is other:
            return 0
        for x, y in zip(self, other):
            if x == y:
                continue
            if x < y:
                return -1
            if x > y:
                return 1
            raise RuntimeError  # Unreachable
        return self.shape.compare(other.shape)

    def items(self, *, by=Rule.ROW, reverse=False):
        """Return an iterator that yields the matrix's items in row or
        column-major order
        """
        it = reversed if reverse else iter
        ix = range(self.nrows)
        jx = range(self.ncols)
        if by is Rule.ROW:
            for i in it(ix):
                for j in it(jx):
                    yield self[i, j]
        else:
            for j in it(jx):
                for i in it(ix):
                    yield self[i, j]

    def slices(self, *, by=Rule.ROW, reverse=False):
        """Return an iterator that yields shallow copies of each row or column"""
        it = reversed if reverse else iter
        if by is Rule.ROW:
            for i in it(range(self.nrows)):
                yield self[i, :]
        else:
            for j in it(range(self.ncols)):
                yield self[:, j]

    def _resolve_index(self, key, *, by=None):
        """Validate, sanitize, and return an index `key` as a built-in `int`
        with respect to the matrix's number of items, rows, or columns

        This method uses the extended `Rule` convention, where `by=None`
        corresponds to a "flattened" interpretation of the method.

        Raises `IndexError` if the `key` is out of range.
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

        This method uses the extended `Rule` convention, where `by=None`
        corresponds to a "flattened" interpretation of the method.
        """
        bound = self.size if by is None else self.shape[by.value]
        return range(*key.indices(bound))

    def _permute_index_single(self, val_index):
        """Permute and return the given singular index as its "canonical" form

        This method is typically called post-resolution, and pre-retrieval (the
        return value of this function being the final index used in the
        retrieval process). At the base level, this function simply returns
        `val_index`.
        """
        return val_index

    def _permute_index_double(self, row_index, col_index):
        """Permute and return the given paired index as its "canonical" form

        This method is typically called post-resolution, and pre-retrieval (the
        return value of this function being the final index used in the
        retrieval process). At the base level, this function simply returns
        `row_index * self.ncols + col_index`.
        """
        return row_index * self.ncols + col_index
