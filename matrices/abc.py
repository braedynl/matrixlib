from __future__ import annotations

import functools
import operator
from abc import ABCMeta, abstractmethod
from collections.abc import Collection, Iterable, Iterator, Sequence, Sized
from typing import (Any, Literal, Protocol, SupportsIndex, TypeVar, Union,
                    overload, runtime_checkable)

from .rule import Rule

__all__ = [
    "Indexable",
    "Shaped",
    "ShapedIndexable",
    "ShapedIterable",
    "ShapedCollection",
    "ShapedSequence",
    "MatrixLike",
    "check_friendly",
    "ComplexMatrixLike",
    "RealMatrixLike",
    "IntegralMatrixLike",
]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)
P_co = TypeVar("P_co", covariant=True, bound=int)

ComplexT_co = TypeVar("ComplexT_co", covariant=True, bound=complex)
RealT_co = TypeVar("RealT_co", covariant=True, bound=float)
IntegralT_co = TypeVar("IntegralT_co", covariant=True, bound=int)


@runtime_checkable
class Indexable(Protocol[T_co]):
    """Protocol for classes that support vector and matrix-like
    ``__getitem__()`` access

    Note that slicing is not a requirement by this protocol.
    """

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...

    @abstractmethod
    def __getitem__(self, key):
        """Return the value corresponding to ``key``"""
        raise NotImplementedError


@runtime_checkable
class Shaped(Sized, Protocol[M_co, N_co]):
    """Protocol for classes that support ``shape`` and ``__len__()``"""

    def __len__(self) -> int:
        """Return the product of the shape"""
        shape = self.shape
        return shape[0] * shape[1]

    @property
    @abstractmethod
    def shape(self) -> tuple[M_co, N_co]:
        """The number of rows and columns as a ``tuple``"""
        raise NotImplementedError

    @property
    def nrows(self) -> M_co:
        """The number of rows"""
        return self.shape[0]

    @property
    def ncols(self) -> N_co:
        """The number of columns"""
        return self.shape[1]

    @property
    def size(self) -> int:
        """The product of the shape"""
        return len(self)


@runtime_checkable
class ShapedIndexable(Shaped[M_co, N_co], Indexable[T_co], Protocol[T_co, M_co, N_co]):
    """Protocol for classes that support ``shape``, and index-based
    ``__getitem__()``
    """


@runtime_checkable
class ShapedIterable(Shaped[M_co, N_co], Iterable[T_co], Protocol[T_co, M_co, N_co]):
    """Protocol for classes that support ``shape``, ``__len__()``, and
    ``__iter__()``
    """


@runtime_checkable
class ShapedCollection(ShapedIterable[T_co, M_co, N_co], Collection[T_co], Protocol[T_co, M_co, N_co]):
    """Protocol for classes that support ``shape``, ``__len__()``,
    ``__iter__()``, and ``__contains__()``
    """

    def __contains__(self, value: Any) -> bool:
        """Return true if the collection contains ``value``, otherwise false"""
        for x in self:
            if x is value or x == value:
                return True
        return False


class ShapedSequence(ShapedCollection[T_co, M_co, N_co], Sequence[T_co], metaclass=ABCMeta):
    """Abstract base class for shaped sequence types"""

    __slots__ = ()

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> ShapedSequence[T_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> ShapedSequence[T_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> ShapedSequence[T_co, Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> ShapedSequence[T_co, Any, Any]: ...

    @abstractmethod
    def __getitem__(self, key):
        """Return the value or sub-sequence corresponding to ``key``"""
        raise NotImplementedError


class MatrixLike(ShapedSequence[T_co, M_co, N_co], metaclass=ABCMeta):
    """Abstract base class for matrix-like objects

    A kind of "hybrid" generic sequence type that interfaces both one and two
    dimensional methods, alongside a variety of vectorized operations.
    """

    __slots__ = ()
    __match_args__ = ("array", "shape")

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> MatrixLike[T_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> MatrixLike[T_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> MatrixLike[T_co, Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> MatrixLike[T_co, Any, Any]: ...

    @abstractmethod
    def __getitem__(self, key):
        """Return the value or sub-matrix corresponding to ``key``"""
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        """Return true if the two matrices are equal, otherwise false"""
        if isinstance(other, MatrixLike):
            return (
                self.shape == other.shape
                and
                all((x is y or x == y) for x, y in zip(self, other))
            )
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        """Return true if the two matrices are not equal, otherwise false"""
        if isinstance(other, MatrixLike):
            return (
                self.shape != other.shape
                or
                any(not (x is y or x == y) for x, y in zip(self, other))
            )
        return NotImplemented

    def __iter__(self) -> Iterator[T_co]:
        """Return an iterator over the values of the matrix in row-major order"""
        yield from self.values()

    def __reversed__(self) -> Iterator[T_co]:
        """Return an iterator over the values of the matrix in reverse
        row-major order
        """
        yield from self.values(reverse=True)

    @property
    @abstractmethod
    def array(self) -> Sequence[T_co]:
        """A sequence of the matrix's elements"""
        raise NotImplementedError

    @abstractmethod
    def equal(self, other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]:
        """Return element-wise ``a == b``"""
        raise NotImplementedError

    @abstractmethod
    def not_equal(self, other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]:
        """Return element-wise ``a != b``"""
        raise NotImplementedError

    @abstractmethod
    def transpose(self) -> MatrixLike[T_co, N_co, M_co]:
        """Return the matrix transpose"""
        raise NotImplementedError

    @abstractmethod
    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T_co, M_co, N_co]:
        """Return the matrix flipped across the rows or columns"""
        raise NotImplementedError

    @abstractmethod
    def reverse(self) -> MatrixLike[T_co, M_co, N_co]:
        """Return the matrix reversed"""
        raise NotImplementedError

    @overload
    def n(self, by: Literal[Rule.ROW]) -> M_co: ...
    @overload
    def n(self, by: Literal[Rule.COL]) -> N_co: ...
    @overload
    def n(self, by: Rule) -> Union[M_co, N_co]: ...

    def n(self, by):
        """Return the dimension corresponding to the given rule

        At the base level, this method is equivalent to `self.shape[by.value]`.
        For some matrix implementations, however, retrieving a dimension from
        this method may be faster than going through the `shape` property.

        This is the recommended method to use for all rule-based dimension
        retrievals.
        """
        return self.shape[by.value]

    def values(self, *, by: Rule = Rule.ROW, reverse: bool = False) -> Iterator[T_co]:
        """Return an iterator that yields the matrix's items in row or
        column-major order
        """
        values: Any = reversed if reverse else iter  # MyPy doesn't handle this expression well
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

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[MatrixLike[T_co, Literal[1], N_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[MatrixLike[T_co, M_co, Literal[1]]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[MatrixLike[T_co, Any, Any]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[MatrixLike[T_co, Literal[1], N_co]]: ...

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

    def _resolve_vector_index(self, key: SupportsIndex) -> int:
        index = operator.index(key)
        bound = self.size
        index += bound * (index < 0)
        if index < 0 or index >= bound:
            raise IndexError(f"there are {bound} items but index is {key}")
        return index

    def _resolve_matrix_index(self, key: SupportsIndex, *, by: Rule = Rule.ROW) -> int:
        index = operator.index(key)
        bound = self.n(by)
        index += bound * (index < 0)
        if index < 0 or index >= bound:
            raise IndexError(f"there are {bound} {by.handle}s but index is {key}")
        return index

    def _resolve_vector_slice(self, key: slice) -> Iterable[int]:
        bound = self.size
        return range(*key.indices(bound))

    def _resolve_matrix_slice(self, key: slice, *, by: Rule = Rule.ROW) -> Iterable[int]:
        bound = self.n(by)
        return range(*key.indices(bound))


def check_friendly(method, /):
    """Return ``NotImplemented`` for "un-friendly" argument types passed to
    special binary methods

    This utility is primarily intended for implementations of
    ``ComplexMatrixLike``, ``RealMatrixLike``, and ``IntegralMatrixLike``, who
    define a ``FRIENDLY_TYPES`` class attribute for interoperability with one
    another.
    """

    @functools.wraps(method)
    def check_friendly_wrapper(self, other):
        if isinstance(other, self.FRIENDLY_TYPES):
            return method(self, other)
        return NotImplemented

    return check_friendly_wrapper


def compare(a, b, /):
    """Return literal -1, 0, or 1 if two matrices lexicographically compare as
    ``a < b``, ``a = b``, or ``a > b``, respectively

    This function is used to implement the base comparison operators for
    ``RealMatrixLike`` and ``IntegralMatrixLike``.
    """
    if a is b:
        return 0
    for x, y in zip(a, b):
        if x is y or x == y:
            continue
        if x < y:
            return -1
        if x > y:
            return 1
        raise RuntimeError
    u = a.shape
    v = b.shape
    if u is v:
        return 0
    for m, n in zip(u, v):
        if m is n or m == n:
            continue
        if m < n:
            return -1
        if m > n:
            return 1
    return 0


class ComplexMatrixLike(MatrixLike[ComplexT_co, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> ComplexT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> ComplexMatrixLike[ComplexT_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> ComplexT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> ComplexMatrixLike[ComplexT_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> ComplexMatrixLike[ComplexT_co, Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> ComplexMatrixLike[ComplexT_co, Any, Any]: ...

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    @overload
    @abstractmethod
    def __add__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __add__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __add__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...

    @abstractmethod
    def __add__(self, other):
        """Return element-wise ``a + b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __sub__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __sub__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __sub__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...

    @abstractmethod
    def __sub__(self, other):
        """Return element-wise ``a - b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __mul__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mul__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mul__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...

    @abstractmethod
    def __mul__(self, other):
        """Return element-wise ``a * b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __matmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, N_co, P_co]) -> ComplexMatrixLike[complex, M_co, P_co]: ...
    @overload
    @abstractmethod
    def __matmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, N_co, P_co]) -> ComplexMatrixLike[complex, M_co, P_co]: ...
    @overload
    @abstractmethod
    def __matmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, N_co, P_co]) -> ComplexMatrixLike[complex, M_co, P_co]: ...

    @abstractmethod
    def __matmul__(self, other):
        """Return the matrix product"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __truediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...

    @abstractmethod
    def __truediv__(self, other):
        """Return element-wise ``a / b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __radd__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __radd__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __radd__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...

    @abstractmethod
    def __radd__(self, other):
        """Return element-wise ``b + a``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rsub__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rsub__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rsub__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...

    @abstractmethod
    def __rsub__(self, other):
        """Return element-wise ``b - a``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...

    @abstractmethod
    def __rmul__(self, other):
        """Return element-wise ``b * a``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rmatmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, P_co, M_co]) -> ComplexMatrixLike[complex, P_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmatmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, P_co, M_co]) -> ComplexMatrixLike[complex, P_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmatmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, P_co, M_co]) -> ComplexMatrixLike[complex, P_co, N_co]: ...

    @abstractmethod
    def __rmatmul__(self, other):
        """Return the reverse matrix product"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rtruediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rtruediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rtruediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...

    @abstractmethod
    def __rtruediv__(self, other):
        """Return element-wise ``b / a``"""
        raise NotImplementedError

    @abstractmethod
    def __neg__(self: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]:
        """Return element-wise ``-a``"""
        raise NotImplementedError

    @abstractmethod
    def __abs__(self: ComplexMatrixLike[complex, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]:
        """Return element-wise ``abs(a)``"""
        raise NotImplementedError

    def __pos__(self: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]:
        """Return element-wise ``+a``"""
        return self

    @abstractmethod
    def transpose(self) -> ComplexMatrixLike[ComplexT_co, N_co, M_co]:
        raise NotImplementedError

    @abstractmethod
    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[ComplexT_co, M_co, N_co]:
        raise NotImplementedError

    @abstractmethod
    def reverse(self) -> ComplexMatrixLike[ComplexT_co, M_co, N_co]:
        raise NotImplementedError

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[ComplexMatrixLike[ComplexT_co, Literal[1], N_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[ComplexMatrixLike[ComplexT_co, M_co, Literal[1]]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[ComplexMatrixLike[ComplexT_co, Any, Any]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[ComplexMatrixLike[ComplexT_co, Literal[1], N_co]]: ...

    def slices(self, *, by=Rule.ROW, reverse=False):
        yield from super().slices(by=by, reverse=reverse)

    @abstractmethod
    def conjugate(self) -> ComplexMatrixLike[ComplexT_co, M_co, N_co]:
        """Return element-wise ``a.conjugate()``"""
        raise NotImplementedError

    def transjugate(self) -> ComplexMatrixLike[ComplexT_co, N_co, M_co]:
        """Return the conjugate transpose"""
        return self.transpose().conjugate()


class RealMatrixLike(MatrixLike[RealT_co, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> RealT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> RealMatrixLike[RealT_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> RealT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> RealMatrixLike[RealT_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> RealMatrixLike[RealT_co, Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> RealMatrixLike[RealT_co, Any, Any]: ...

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    @overload
    def __lt__(self, other: IntegralMatrixLike) -> bool: ...  # type: ignore[has-type]
    @overload
    def __lt__(self, other: RealMatrixLike) -> bool: ...

    @check_friendly
    def __lt__(self, other):
        """Return true if lexicographic ``a < b``, otherwise false"""
        return compare(self, other) < 0

    @overload
    def __le__(self, other: IntegralMatrixLike) -> bool: ...  # type: ignore[has-type]
    @overload
    def __le__(self, other: RealMatrixLike) -> bool: ...

    @check_friendly
    def __le__(self, other):
        """Return true if lexicographic ``a <= b``, otherwise false"""
        return compare(self, other) <= 0

    @overload
    def __gt__(self, other: IntegralMatrixLike) -> bool: ...  # type: ignore[has-type]
    @overload
    def __gt__(self, other: RealMatrixLike) -> bool: ...

    @check_friendly
    def __gt__(self, other):
        """Return true if lexicographic ``a > b``, otherwise false"""
        return compare(self, other) > 0

    @overload
    def __ge__(self, other: IntegralMatrixLike) -> bool: ...  # type: ignore[has-type]
    @overload
    def __ge__(self, other: RealMatrixLike) -> bool: ...

    @check_friendly
    def __ge__(self, other):
        """Return true if lexicographic ``a >= b``, otherwise false"""
        return compare(self, other) >= 0

    @overload
    @abstractmethod
    def __add__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __add__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    @abstractmethod
    def __add__(self, other):
        """Return element-wise ``a + b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __sub__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __sub__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    @abstractmethod
    def __sub__(self, other):
        """Return element-wise ``a - b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __mul__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mul__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    @abstractmethod
    def __mul__(self, other):
        """Return element-wise ``a * b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __matmul__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, N_co, P_co]) -> RealMatrixLike[float, M_co, P_co]: ...
    @overload
    @abstractmethod
    def __matmul__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, N_co, P_co]) -> RealMatrixLike[float, M_co, P_co]: ...

    @abstractmethod
    def __matmul__(self, other):
        """Return the matrix product"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __truediv__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    @abstractmethod
    def __truediv__(self, other):
        """Return element-wise ``a / b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __floordiv__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __floordiv__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    @abstractmethod
    def __floordiv__(self, other):
        """Return element-wise ``a // b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __mod__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mod__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    @abstractmethod
    def __mod__(self, other):
        """Return element-wise ``a % b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __divmod__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> MatrixLike[tuple[float, float], M_co, N_co]: ...
    @overload
    @abstractmethod
    def __divmod__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> MatrixLike[tuple[float, float], M_co, N_co]: ...

    @abstractmethod
    def __divmod__(self, other):
        """Return element-wise ``divmod(a, b)``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __radd__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __radd__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    @abstractmethod
    def __radd__(self, other):
        """Return element-wise ``b + a``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rsub__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rsub__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    @abstractmethod
    def __rsub__(self, other):
        """Return element-wise ``b - a``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rmul__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmul__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    @abstractmethod
    def __rmul__(self, other):
        """Return element-wise ``b * a``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rmatmul__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, P_co, M_co]) -> RealMatrixLike[float, P_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmatmul__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, P_co, M_co]) -> RealMatrixLike[float, P_co, N_co]: ...

    @abstractmethod
    def __rmatmul__(self, other):
        """Return the reverse matrix product"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rtruediv__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rtruediv__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    @abstractmethod
    def __rtruediv__(self, other):
        """Return element-wise ``b / a``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rfloordiv__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rfloordiv__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    @abstractmethod
    def __rfloordiv__(self, other):
        """Return element-wise ``b // a``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rmod__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmod__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    @abstractmethod
    def __rmod__(self, other):
        """Return element-wise ``b % a``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rdivmod__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> MatrixLike[tuple[float, float], M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rdivmod__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> MatrixLike[tuple[float, float], M_co, N_co]: ...

    @abstractmethod
    def __rdivmod__(self, other):
        """Return element-wise ``divmod(b, a)``"""
        raise NotImplementedError

    @abstractmethod
    def __neg__(self: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]:
        """Return element-wise ``-a``"""
        raise NotImplementedError

    @abstractmethod
    def __abs__(self: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]:
        """Return element-wise ``abs(a)``"""
        raise NotImplementedError

    def __pos__(self: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]:
        """Return element-wise ``+a``"""
        return self

    @abstractmethod
    def transpose(self) -> RealMatrixLike[RealT_co, N_co, M_co]:
        raise NotImplementedError

    @abstractmethod
    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[RealT_co, M_co, N_co]:
        raise NotImplementedError

    @abstractmethod
    def reverse(self) -> RealMatrixLike[RealT_co, M_co, N_co]:
        raise NotImplementedError

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[RealMatrixLike[RealT_co, Literal[1], N_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[RealMatrixLike[RealT_co, M_co, Literal[1]]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[RealMatrixLike[RealT_co, Any, Any]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[RealMatrixLike[RealT_co, Literal[1], N_co]]: ...

    def slices(self, *, by=Rule.ROW, reverse=False):
        yield from super().slices(by=by, reverse=reverse)

    @overload
    @abstractmethod
    def lesser(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def lesser(self, other: RealMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...

    @abstractmethod
    def lesser(self, other):
        """Return element-wise ``a < b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def lesser_equal(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def lesser_equal(self, other: RealMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...

    @abstractmethod
    def lesser_equal(self, other):
        """Return element-wise ``a <= b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def greater(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def greater(self, other: RealMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...

    @abstractmethod
    def greater(self, other):
        """Return element-wise ``a > b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def greater_equal(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def greater_equal(self, other: RealMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...

    @abstractmethod
    def greater_equal(self, other):
        """Return element-wise ``a >= b``"""
        raise NotImplementedError

    def conjugate(self) -> RealMatrixLike[RealT_co, M_co, N_co]:
        """Return element-wise ``a.conjugate()``"""
        return self

    def transjugate(self) -> RealMatrixLike[RealT_co, N_co, M_co]:
        """Return the conjugate transpose"""
        return self.transpose()


class IntegralMatrixLike(MatrixLike[IntegralT_co, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> IntegralT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> IntegralMatrixLike[IntegralT_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> IntegralT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> IntegralMatrixLike[IntegralT_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> IntegralMatrixLike[IntegralT_co, Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> IntegralMatrixLike[IntegralT_co, Any, Any]: ...

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    @check_friendly
    def __lt__(self, other: IntegralMatrixLike) -> bool:  # type: ignore[has-type]
        """Return true if lexicographic ``a < b``, otherwise false"""
        return compare(self, other) < 0

    @check_friendly
    def __le__(self, other: IntegralMatrixLike) -> bool:  # type: ignore[has-type]
        """Return true if lexicographic ``a <= b``, otherwise false"""
        return compare(self, other) <= 0

    @check_friendly
    def __gt__(self, other: IntegralMatrixLike) -> bool:
        """Return true if lexicographic ``a > b``, otherwise false"""
        return compare(self, other) > 0

    @check_friendly
    def __ge__(self, other: IntegralMatrixLike) -> bool:
        """Return true if lexicographic ``a >= b``, otherwise false"""
        return compare(self, other) >= 0

    @abstractmethod
    def __add__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``a + b``"""
        raise NotImplementedError

    @abstractmethod
    def __sub__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``a - b``"""
        raise NotImplementedError

    @abstractmethod
    def __mul__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``a * b``"""
        raise NotImplementedError

    @abstractmethod
    def __matmul__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, N_co, P_co]) -> IntegralMatrixLike[int, M_co, P_co]:
        """Return the matrix product"""
        raise NotImplementedError

    @abstractmethod
    def __truediv__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]:
        """Return element-wise ``a / b``"""
        raise NotImplementedError

    @abstractmethod
    def __floordiv__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``a // b``"""
        raise NotImplementedError

    @abstractmethod
    def __mod__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``a % b``"""
        raise NotImplementedError

    @abstractmethod
    def __divmod__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> MatrixLike[tuple[int, int], M_co, N_co]:
        """Return element-wise ``divmod(a, b)``"""
        raise NotImplementedError

    @abstractmethod
    def __lshift__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``a << b``"""
        raise NotImplementedError

    @abstractmethod
    def __rshift__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``a >> b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __and__(self: IntegralMatrixLike[bool, M_co, N_co], other: IntegralMatrixLike[bool, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __and__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...

    @abstractmethod
    def __and__(self, other):
        """Return element-wise ``a & b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __xor__(self: IntegralMatrixLike[bool, M_co, N_co], other: IntegralMatrixLike[bool, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __xor__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...

    @abstractmethod
    def __xor__(self, other):
        """Return element-wise ``a ^ b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __or__(self: IntegralMatrixLike[bool, M_co, N_co], other: IntegralMatrixLike[bool, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __or__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...

    @abstractmethod
    def __or__(self, other):
        """Return element-wise ``a | b``"""
        raise NotImplementedError

    @abstractmethod
    def __radd__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``b + a``"""
        raise NotImplementedError

    @abstractmethod
    def __rsub__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``b - a``"""
        raise NotImplementedError

    @abstractmethod
    def __rmul__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``b * a``"""
        raise NotImplementedError

    @abstractmethod
    def __rmatmul__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, P_co, M_co]) -> IntegralMatrixLike[int, P_co, N_co]:
        """Return the reverse matrix product"""
        raise NotImplementedError

    @abstractmethod
    def __rtruediv__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]:
        """Return element-wise ``b / a``"""
        raise NotImplementedError

    @abstractmethod
    def __rfloordiv__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``b // a``"""
        raise NotImplementedError

    @abstractmethod
    def __rmod__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``b % a``"""
        raise NotImplementedError

    @abstractmethod
    def __rdivmod__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> MatrixLike[tuple[int, int], M_co, N_co]:
        """Return element-wise ``divmod(b, a)``"""
        raise NotImplementedError

    @abstractmethod
    def __rlshift__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``b << a``"""
        raise NotImplementedError

    @abstractmethod
    def __rrshift__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``b >> a``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rand__(self: IntegralMatrixLike[bool, M_co, N_co], other: IntegralMatrixLike[bool, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rand__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...

    @abstractmethod
    def __rand__(self, other):
        """Return element-wise ``b & a``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rxor__(self: IntegralMatrixLike[bool, M_co, N_co], other: IntegralMatrixLike[bool, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rxor__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...

    @abstractmethod
    def __rxor__(self, other):
        """Return element-wise ``b ^ a``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __ror__(self: IntegralMatrixLike[bool, M_co, N_co], other: IntegralMatrixLike[bool, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __ror__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...

    @abstractmethod
    def __ror__(self, other):
        """Return element-wise ``b | a``"""
        raise NotImplementedError

    @abstractmethod
    def __neg__(self: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``-a``"""
        raise NotImplementedError

    @abstractmethod
    def __abs__(self: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``abs(a)``"""
        raise NotImplementedError

    @abstractmethod
    def __invert__(self: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``~a``"""
        raise NotImplementedError

    def __pos__(self: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]:
        """Return element-wise ``+a``"""
        return self

    @abstractmethod
    def transpose(self) -> IntegralMatrixLike[IntegralT_co, N_co, M_co]:
        raise NotImplementedError

    @abstractmethod
    def flip(self, *, by: Rule = Rule.ROW) -> IntegralMatrixLike[IntegralT_co, M_co, N_co]:
        raise NotImplementedError

    @abstractmethod
    def reverse(self) -> IntegralMatrixLike[IntegralT_co, M_co, N_co]:
        raise NotImplementedError

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[IntegralMatrixLike[IntegralT_co, Literal[1], N_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[IntegralMatrixLike[IntegralT_co, M_co, Literal[1]]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[IntegralMatrixLike[IntegralT_co, Any, Any]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[IntegralMatrixLike[IntegralT_co, Literal[1], N_co]]: ...

    def slices(self, *, by=Rule.ROW, reverse=False):
        yield from super().slices(by=by, reverse=reverse)

    @abstractmethod
    def lesser(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]:
        """Return element-wise ``a < b``"""
        raise NotImplementedError

    @abstractmethod
    def lesser_equal(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]:
        """Return element-wise ``a <= b``"""
        raise NotImplementedError

    @abstractmethod
    def greater(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]:
        """Return element-wise ``a > b``"""
        raise NotImplementedError

    @abstractmethod
    def greater_equal(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]:
        """Return element-wise ``a >= b``"""
        raise NotImplementedError

    def conjugate(self) -> IntegralMatrixLike[IntegralT_co, M_co, N_co]:
        """Return element-wise ``a.conjugate()``"""
        return self

    def transjugate(self) -> IntegralMatrixLike[IntegralT_co, N_co, M_co]:
        """Return the conjugate transpose"""
        return self.transpose()


ComplexMatrixLike.FRIENDLY_TYPES = (ComplexMatrixLike, RealMatrixLike, IntegralMatrixLike)  # type: ignore
RealMatrixLike.FRIENDLY_TYPES = (RealMatrixLike, IntegralMatrixLike)  # type: ignore
IntegralMatrixLike.FRIENDLY_TYPES = IntegralMatrixLike  # type: ignore
