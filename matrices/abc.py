import operator
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Sequence, Sized
from typing import Protocol, TypeVar, runtime_checkable

from .rule import Rule

__all__ = [
    "Shaped",
    "ShapedIterable",
    "MatrixLike",
    "ComplexMatrixLike",
    "RealMatrixLike",
    "IntegralMatrixLike",
]

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)

ComplexT_co = TypeVar("ComplexT_co", covariant=True, bound=complex)
RealT_co = TypeVar("RealT_co", covariant=True, bound=float)
IntegralT_co = TypeVar("IntegralT_co", covariant=True, bound=int)


@runtime_checkable
class Shaped(Sized, Protocol[M_co, N_co]):
    """Protocol for classes that support ``shape`` and ``__len__()``"""

    def __len__(self):
        """Return the product of the shape"""
        shape = self.shape
        return shape[0] * shape[1]

    @property
    @abstractmethod
    def shape(self):
        """The number of rows and columns as a ``tuple``"""
        pass


@runtime_checkable
class ShapedIterable(Shaped[M_co, N_co], Iterable[T_co], Protocol[T_co, M_co, N_co]):
    """Protocol for classes that support ``shape``, ``__len__()``, and
    ``__iter__()``
    """
    pass


class MatrixLike(Sequence[T_co], ShapedIterable[T_co, M_co, N_co], metaclass=ABCMeta):
    """Abstract base class for matrix-like objects

    A kind of "hybrid" generic sequence type that interfaces both one and two
    dimensional methods, alongside a variety of vectorized operations.
    """

    __slots__ = ()
    __match_args__ = ("array", "shape")

    def __eq__(self, other):
        """Return true if the two matrices are equal, otherwise false"""
        if isinstance(other, MatrixLike):
            return (
                self.shape == other.shape
                and
                all(map(lambda x, y: (x is y) or (x == y), self, other))
            )
        return NotImplemented

    def __ne__(self, other):
        """Return true if the two matrices are not equal, otherwise false"""
        if isinstance(other, MatrixLike):
            return (
                self.shape != other.shape
                or
                any(map(lambda x, y: (x is not y) and (x != y), self, other))
            )
        return NotImplemented

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
    def nrows(self):
        """The number of rows"""
        return self.shape[0]

    @property
    def ncols(self):
        """The number of columns"""
        return self.shape[1]

    @property
    def size(self):
        """The product of the shape"""
        return len(self)

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


class ComplexMatrixLike(MatrixLike[ComplexT_co, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()

    @abstractmethod
    def __add__(self, other):
        """Return element-wise ``a + b``"""
        pass

    @abstractmethod
    def __sub__(self, other):
        """Return element-wise ``a - b``"""
        pass

    @abstractmethod
    def __mul__(self, other):
        """Return element-wise ``a * b``"""
        pass

    @abstractmethod
    def __matmul__(self, other):
        """Return the matrix product"""
        pass

    @abstractmethod
    def __truediv__(self, other):
        """Return element-wise ``a / b``"""
        pass

    @abstractmethod
    def __radd__(self, other):
        """Return element-wise ``b + a``"""
        pass

    @abstractmethod
    def __rsub__(self, other):
        """Return element-wise ``b - a``"""
        pass

    @abstractmethod
    def __rmul__(self, other):
        """Return element-wise ``b * a``"""
        pass

    @abstractmethod
    def __rmatmul__(self, other):
        """Return the reverse matrix product"""
        pass

    @abstractmethod
    def __rtruediv__(self, other):
        """Return element-wise ``b / a``"""
        pass

    @abstractmethod
    def __neg__(self):
        """Return element-wise ``-a``"""
        pass

    @abstractmethod
    def __pos__(self):
        """Return element-wise ``+a``"""
        pass

    @abstractmethod
    def __abs__(self):
        """Return element-wise ``abs(a)``"""
        pass

    @abstractmethod
    def conjugate(self):
        """Return element-wise ``a.conjugate()``"""
        pass


class RealMatrixLike(MatrixLike[RealT_co, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()

    def __lt__(self, other):
        """Return true if lexicographic ``a < b``, otherwise false"""
        if isinstance(other, self.COMPARABLE_TYPES):
            return matrix_compare(self, other) < 0
        return NotImplemented

    def __le__(self, other):
        """Return true if lexicographic ``a <= b``, otherwise false"""
        if isinstance(other, self.COMPARABLE_TYPES):
            return matrix_compare(self, other) <= 0
        return NotImplemented

    def __gt__(self, other):
        """Return true if lexicographic ``a > b``, otherwise false"""
        if isinstance(other, self.COMPARABLE_TYPES):
            return matrix_compare(self, other) > 0
        return NotImplemented

    def __ge__(self, other):
        """Return true if lexicographic ``a >= b``, otherwise false"""
        if isinstance(other, self.COMPARABLE_TYPES):
            return matrix_compare(self, other) >= 0
        return NotImplemented

    @abstractmethod
    def __add__(self, other):
        """Return element-wise ``a + b``"""
        pass

    @abstractmethod
    def __sub__(self, other):
        """Return element-wise ``a - b``"""
        pass

    @abstractmethod
    def __mul__(self, other):
        """Return element-wise ``a * b``"""
        pass

    @abstractmethod
    def __matmul__(self, other):
        """Return the matrix product"""
        pass

    @abstractmethod
    def __truediv__(self, other):
        """Return element-wise ``a / b``"""
        pass

    @abstractmethod
    def __floordiv__(self, other):
        """Return element-wise ``a // b``"""
        pass

    @abstractmethod
    def __mod__(self, other):
        """Return element-wise ``a % b``"""
        pass

    @abstractmethod
    def __divmod__(self, other):
        """Return element-wise ``divmod(a, b)``"""
        pass

    @abstractmethod
    def __radd__(self, other):
        """Return element-wise ``b + a``"""
        pass

    @abstractmethod
    def __rsub__(self, other):
        """Return element-wise ``b - a``"""
        pass

    @abstractmethod
    def __rmul__(self, other):
        """Return element-wise ``b * a``"""
        pass

    @abstractmethod
    def __rmatmul__(self, other):
        """Return the reverse matrix product"""
        pass

    @abstractmethod
    def __rtruediv__(self, other):
        """Return element-wise ``b / a``"""
        pass

    @abstractmethod
    def __rfloordiv__(self, other):
        """Return element-wise ``b // a``"""
        pass

    @abstractmethod
    def __rmod__(self, other):
        """Return element-wise ``b % a``"""
        pass

    @abstractmethod
    def __rdivmod__(self, other):
        """Return element-wise ``divmod(b, a)``"""
        pass

    @abstractmethod
    def __neg__(self):
        """Return element-wise ``-a``"""
        pass

    @abstractmethod
    def __pos__(self):
        """Return element-wise ``+a``"""
        pass

    @abstractmethod
    def __abs__(self):
        """Return element-wise ``abs(a)``"""
        pass

    @abstractmethod
    def lesser(self, other):
        """Return element-wise ``a < b``"""
        pass

    @abstractmethod
    def lesser_equal(self, other):
        """Return element-wise ``a <= b``"""
        pass

    @abstractmethod
    def greater(self, other):
        """Return element-wise ``a > b``"""
        pass

    @abstractmethod
    def greater_equal(self, other):
        """Return element-wise ``a >= b``"""
        pass

    @abstractmethod
    def conjugate(self):
        """Return element-wise ``a.conjugate()``"""
        pass


class IntegralMatrixLike(MatrixLike[IntegralT_co, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()

    def __lt__(self, other):
        """Return true if lexicographic ``a < b``, otherwise false"""
        if isinstance(other, self.COMPARABLE_TYPES):
            return matrix_compare(self, other) < 0
        return NotImplemented

    def __le__(self, other):
        """Return true if lexicographic ``a <= b``, otherwise false"""
        if isinstance(other, self.COMPARABLE_TYPES):
            return matrix_compare(self, other) <= 0
        return NotImplemented

    def __gt__(self, other):
        """Return true if lexicographic ``a > b``, otherwise false"""
        if isinstance(other, self.COMPARABLE_TYPES):
            return matrix_compare(self, other) > 0
        return NotImplemented

    def __ge__(self, other):
        """Return true if lexicographic ``a >= b``, otherwise false"""
        if isinstance(other, self.COMPARABLE_TYPES):
            return matrix_compare(self, other) >= 0
        return NotImplemented

    @abstractmethod
    def __add__(self, other):
        """Return element-wise ``a + b``"""
        pass

    @abstractmethod
    def __sub__(self, other):
        """Return element-wise ``a - b``"""
        pass

    @abstractmethod
    def __mul__(self, other):
        """Return element-wise ``a * b``"""
        pass

    @abstractmethod
    def __matmul__(self, other):
        """Return the matrix product"""
        pass

    @abstractmethod
    def __truediv__(self, other):
        """Return element-wise ``a / b``"""
        pass

    @abstractmethod
    def __floordiv__(self, other):
        """Return element-wise ``a // b``"""
        pass

    @abstractmethod
    def __mod__(self, other):
        """Return element-wise ``a % b``"""
        pass

    @abstractmethod
    def __divmod__(self, other):
        """Return element-wise ``divmod(a, b)``"""
        pass

    @abstractmethod
    def __lshift__(self, other):
        """Return element-wise ``a << b``"""
        pass

    @abstractmethod
    def __rshift__(self, other):
        """Return element-wise ``a >> b``"""
        pass

    @abstractmethod
    def __and__(self, other):
        """Return element-wise ``a & b``"""
        pass

    @abstractmethod
    def __xor__(self, other):
        """Return element-wise ``a ^ b``"""
        pass

    @abstractmethod
    def __or__(self, other):
        """Return element-wise ``a | b``"""
        pass

    @abstractmethod
    def __radd__(self, other):
        """Return element-wise ``b + a``"""
        pass

    @abstractmethod
    def __rsub__(self, other):
        """Return element-wise ``b - a``"""
        pass

    @abstractmethod
    def __rmul__(self, other):
        """Return element-wise ``b * a``"""
        pass

    @abstractmethod
    def __rmatmul__(self, other):
        """Return the reverse matrix product"""
        pass

    @abstractmethod
    def __rtruediv__(self, other):
        """Return element-wise ``b / a``"""
        pass

    @abstractmethod
    def __rfloordiv__(self, other):
        """Return element-wise ``b // a``"""
        pass

    @abstractmethod
    def __rmod__(self, other):
        """Return element-wise ``b % a``"""
        pass

    @abstractmethod
    def __rdivmod__(self, other):
        """Return element-wise ``divmod(b, a)``"""
        pass

    @abstractmethod
    def __rlshift__(self, other):
        """Return element-wise ``b << a``"""
        pass

    @abstractmethod
    def __rrshift__(self, other):
        """Return element-wise ``b >> a``"""
        pass

    @abstractmethod
    def __rand__(self, other):
        """Return element-wise ``b & a``"""
        pass

    @abstractmethod
    def __rxor__(self, other):
        """Return element-wise ``b ^ a``"""
        pass

    @abstractmethod
    def __ror__(self, other):
        """Return element-wise ``b | a``"""
        pass

    @abstractmethod
    def __neg__(self):
        """Return element-wise ``-a``"""
        pass

    @abstractmethod
    def __pos__(self):
        """Return element-wise ``+a``"""
        pass

    @abstractmethod
    def __abs__(self):
        """Return element-wise ``abs(a)``"""
        pass

    @abstractmethod
    def __invert__(self):
        """Return element-wise ``~a``"""
        pass

    @abstractmethod
    def lesser(self, other):
        """Return element-wise ``a < b``"""
        pass

    @abstractmethod
    def lesser_equal(self, other):
        """Return element-wise ``a <= b``"""
        pass

    @abstractmethod
    def greater(self, other):
        """Return element-wise ``a > b``"""
        pass

    @abstractmethod
    def greater_equal(self, other):
        """Return element-wise ``a >= b``"""
        pass

    @abstractmethod
    def conjugate(self):
        """Return element-wise ``a.conjugate()``"""
        pass


ComplexMatrixLike.COMPARABLE_TYPES = (ComplexMatrixLike, RealMatrixLike, IntegralMatrixLike)
RealMatrixLike.COMPARABLE_TYPES = (RealMatrixLike, IntegralMatrixLike)
IntegralMatrixLike.COMPARABLE_TYPES = (IntegralMatrixLike,)


def matrix_compare(a, b):
    if a is b:
        return 0
    for x, y in zip(a, b):
        if x == y:
            continue
        if x < y:
            return -1
        if x > y:
            return 1
        raise RuntimeError
    u = a.shape
    v = b.shape
    if u == v:
        return 0
    if u < v:
        return -1
    if u > v:
        return 1
    raise RuntimeError
