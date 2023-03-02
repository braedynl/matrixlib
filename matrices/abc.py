from __future__ import annotations

import copy
import operator
from abc import ABCMeta, abstractmethod
from collections.abc import Collection, Iterable, Iterator, Sequence, Sized
from datetime import datetime, timedelta
from typing import (Any, Literal, Optional, Protocol, SupportsIndex, TypeVar,
                    Union, overload, runtime_checkable)

from typing_extensions import Self

from .rule import Rule

__all__ = [
    "Indexable",
    "Shaped",
    "ShapedIndexable",
    "ShapedIterable",
    "ShapedCollection",
    "ShapedSequence",
    "MatrixLike",
    "StringMatrixLike",
    "ComplexMatrixLike",
    "RealMatrixLike",
    "IntegralMatrixLike",
    "TimedeltaMatrixLike",
    "DatetimeMatrixLike",
]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)
P_co = TypeVar("P_co", covariant=True, bound=int)


@runtime_checkable
class Indexable(Protocol[T_co]):
    """Protocol for classes that support vector ``__getitem__()`` access"""

    @abstractmethod
    def __getitem__(self, key: int) -> T_co:
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

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError


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

    def __deepcopy__(self, memo: Optional[dict[int, Any]] = None) -> Self:
        """Return the matrix"""
        return self

    __copy__ = __deepcopy__

    @property
    def array(self) -> Sequence[T_co]:
        """A sequence of the matrix's elements

        This property serves two purposes, in allowing for matrix data to be
        matched via a ``match``-``case`` statement (for Python 3.10 or later),
        and in allowing for matrix permutations to have faster access times.

        If the matrix implementation composes a ``Sequence`` type that can be
        safely exposed, it should be returned by this property. Otherwise, the
        matrix itself should be returned (the default).
        """
        return self

    @abstractmethod
    def equal(self, other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a == b``"""
        raise NotImplementedError

    @abstractmethod
    def not_equal(self, other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
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
        bound = len(self)
        index += bound * (index < 0)
        if index < 0 or index >= bound:
            raise IndexError(f"there are {bound} items but index is {key}")
        return index

    def _resolve_matrix_index(self, key: SupportsIndex, *, by: Rule = Rule.ROW) -> int:
        index = operator.index(key)
        bound = self.n(by)
        index += bound * (index < 0)
        if index < 0 or index >= bound:
            handle = by.handle
            raise IndexError(f"there are {bound} {handle}s but index is {key}")
        return index

    def _resolve_vector_slice(self, key: slice) -> Collection[int]:
        bound = len(self)
        return range(*key.indices(bound))

    def _resolve_matrix_slice(self, key: slice, *, by: Rule = Rule.ROW) -> Collection[int]:
        bound = self.n(by)
        return range(*key.indices(bound))


class StringMatrixLike(MatrixLike[str, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> str: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> StringMatrixLike[Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> str: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> StringMatrixLike[Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> StringMatrixLike[Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> StringMatrixLike[Any, Any]: ...

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    def __lt__(self, other: StringMatrixLike) -> bool:
        """Return true if lexicographic ``a < b``, otherwise false"""
        if isinstance(other, StringMatrixLike):
            return lexicographic_compare(self, other) < 0
        return NotImplemented

    def __le__(self, other: StringMatrixLike) -> bool:
        """Return true if lexicographic ``a <= b``, otherwise false"""
        if isinstance(other, StringMatrixLike):
            return lexicographic_compare(self, other) <= 0
        return NotImplemented

    def __gt__(self, other: StringMatrixLike) -> bool:
        """Return true if lexicographic ``a > b``, otherwise false"""
        if isinstance(other, StringMatrixLike):
            return lexicographic_compare(self, other) > 0
        return NotImplemented

    def __ge__(self, other: StringMatrixLike) -> bool:
        """Return true if lexicographic ``a >= b``, otherwise false"""
        if isinstance(other, StringMatrixLike):
            return lexicographic_compare(self, other) >= 0
        return NotImplemented

    @abstractmethod
    def __add__(self, other: StringMatrixLike[M_co, N_co]) -> StringMatrixLike[M_co, N_co]:
        """Return element-wise ``a + b``"""
        raise NotImplementedError

    @abstractmethod
    def __mul__(self, other: IntegralMatrixLike[M_co, N_co]) -> StringMatrixLike[M_co, N_co]:
        """Return element-wise ``a * b``"""
        raise NotImplementedError

    @abstractmethod
    def __rmul__(self, other: IntegralMatrixLike[M_co, N_co]) -> StringMatrixLike[M_co, N_co]:
        """Return element-wise ``b * a``"""
        raise NotImplementedError

    @abstractmethod
    def lesser(self, other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a < b``"""
        raise NotImplementedError

    @abstractmethod
    def lesser_equal(self, other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a <= b``"""
        raise NotImplementedError

    @abstractmethod
    def greater(self, other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a > b``"""
        raise NotImplementedError

    @abstractmethod
    def greater_equal(self, other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a >= b``"""
        raise NotImplementedError


class ComplexMatrixLike(MatrixLike[complex, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> complex: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> ComplexMatrixLike[Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> complex: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> ComplexMatrixLike[Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> ComplexMatrixLike[Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> ComplexMatrixLike[Any, Any]: ...

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    @overload
    @abstractmethod
    def __add__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __add__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __add__(self, other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __add__(self, other):
        """Return element-wise ``a + b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __sub__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __sub__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __sub__(self, other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __sub__(self, other):
        """Return element-wise ``a - b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __mul__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mul__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mul__(self, other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __mul__(self, other):
        """Return element-wise ``a * b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __matmul__(self, other: IntegralMatrixLike[N_co, P_co]) -> ComplexMatrixLike[M_co, P_co]: ...
    @overload
    @abstractmethod
    def __matmul__(self, other: RealMatrixLike[N_co, P_co]) -> ComplexMatrixLike[M_co, P_co]: ...
    @overload
    @abstractmethod
    def __matmul__(self, other: ComplexMatrixLike[N_co, P_co]) -> ComplexMatrixLike[M_co, P_co]: ...

    @abstractmethod
    def __matmul__(self, other):
        """Return the matrix product"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __truediv__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self, other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __truediv__(self, other):
        """Return element-wise ``a / b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __radd__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __radd__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __radd__(self, other):
        """Return element-wise ``b + a``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rsub__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rsub__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __rsub__(self, other):
        """Return element-wise ``b - a``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rmul__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmul__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __rmul__(self, other):
        """Return element-wise ``b * a``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rmatmul__(self, other: IntegralMatrixLike[P_co, M_co]) -> ComplexMatrixLike[P_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmatmul__(self, other: RealMatrixLike[P_co, M_co]) -> ComplexMatrixLike[P_co, N_co]: ...

    @abstractmethod
    def __rmatmul__(self, other):
        """Return the reverse matrix product"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rtruediv__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rtruediv__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __rtruediv__(self, other):
        """Return element-wise ``b / a``"""
        raise NotImplementedError

    @abstractmethod
    def __neg__(self) -> ComplexMatrixLike[M_co, N_co]:
        """Return element-wise ``-a``"""
        raise NotImplementedError

    @abstractmethod
    def __abs__(self) -> RealMatrixLike[M_co, N_co]:
        """Return element-wise ``abs(a)``"""
        raise NotImplementedError

    def __pos__(self) -> ComplexMatrixLike[M_co, N_co]:
        """Return element-wise ``+a``"""
        return copy.copy(self)

    @abstractmethod
    def transpose(self) -> ComplexMatrixLike[N_co, M_co]:
        raise NotImplementedError

    @abstractmethod
    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[M_co, N_co]:
        raise NotImplementedError

    @abstractmethod
    def reverse(self) -> ComplexMatrixLike[M_co, N_co]:
        raise NotImplementedError

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[ComplexMatrixLike[Literal[1], N_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[ComplexMatrixLike[M_co, Literal[1]]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[ComplexMatrixLike[Any, Any]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[ComplexMatrixLike[Literal[1], N_co]]: ...

    def slices(self, *, by=Rule.ROW, reverse=False):
        yield from super().slices(by=by, reverse=reverse)

    @abstractmethod
    def conjugate(self) -> ComplexMatrixLike[M_co, N_co]:
        """Return element-wise ``a.conjugate()``"""
        raise NotImplementedError

    def transjugate(self) -> ComplexMatrixLike[N_co, M_co]:
        """Return the conjugate transpose"""
        return self.transpose().conjugate()


class RealMatrixLike(MatrixLike[float, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> float: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> RealMatrixLike[Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> float: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> RealMatrixLike[Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> RealMatrixLike[Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> RealMatrixLike[Any, Any]: ...

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    @overload
    def __lt__(self, other: IntegralMatrixLike) -> bool: ...
    @overload
    def __lt__(self, other: RealMatrixLike) -> bool: ...

    def __lt__(self, other):
        """Return true if lexicographic ``a < b``, otherwise false"""
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return lexicographic_compare(self, other) < 0
        return NotImplemented

    @overload
    def __le__(self, other: IntegralMatrixLike) -> bool: ...
    @overload
    def __le__(self, other: RealMatrixLike) -> bool: ...

    def __le__(self, other):
        """Return true if lexicographic ``a <= b``, otherwise false"""
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return lexicographic_compare(self, other) <= 0
        return NotImplemented

    @overload
    def __gt__(self, other: IntegralMatrixLike) -> bool: ...
    @overload
    def __gt__(self, other: RealMatrixLike) -> bool: ...

    def __gt__(self, other):
        """Return true if lexicographic ``a > b``, otherwise false"""
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return lexicographic_compare(self, other) > 0
        return NotImplemented

    @overload
    def __ge__(self, other: IntegralMatrixLike) -> bool: ...
    @overload
    def __ge__(self, other: RealMatrixLike) -> bool: ...

    def __ge__(self, other):
        """Return true if lexicographic ``a >= b``, otherwise false"""
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return lexicographic_compare(self, other) >= 0
        return NotImplemented

    @overload
    @abstractmethod
    def __add__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __add__(self, other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __add__(self, other):
        """Return element-wise ``a + b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __sub__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __sub__(self, other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __sub__(self, other):
        """Return element-wise ``a - b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __mul__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mul__(self, other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __mul__(self, other):
        """Return element-wise ``a * b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __matmul__(self, other: IntegralMatrixLike[N_co, P_co]) -> RealMatrixLike[M_co, P_co]: ...
    @overload
    @abstractmethod
    def __matmul__(self, other: RealMatrixLike[N_co, P_co]) -> RealMatrixLike[M_co, P_co]: ...

    @abstractmethod
    def __matmul__(self, other):
        """Return the matrix product"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __truediv__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self, other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __truediv__(self, other):
        """Return element-wise ``a / b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __floordiv__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __floordiv__(self, other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __floordiv__(self, other):
        """Return element-wise ``a // b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __mod__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mod__(self, other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __mod__(self, other):
        """Return element-wise ``a % b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __divmod__(self, other: IntegralMatrixLike[M_co, N_co]) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]: ...
    @overload
    @abstractmethod
    def __divmod__(self, other: RealMatrixLike[M_co, N_co]) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]: ...

    @abstractmethod
    def __divmod__(self, other):
        """Return element-wise ``divmod(a, b)``"""
        raise NotImplementedError

    @abstractmethod
    def __radd__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        """Return element-wise ``b + a``"""
        raise NotImplementedError

    @abstractmethod
    def __rsub__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        """Return element-wise ``b - a``"""
        raise NotImplementedError

    @abstractmethod
    def __rmul__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        """Return element-wise ``b * a``"""
        raise NotImplementedError

    @abstractmethod
    def __rmatmul__(self, other: IntegralMatrixLike[P_co, M_co]) -> RealMatrixLike[P_co, N_co]:
        """Return the reverse matrix product"""
        raise NotImplementedError

    @abstractmethod
    def __rtruediv__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        """Return element-wise ``b / a``"""
        raise NotImplementedError

    @abstractmethod
    def __rfloordiv__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        """Return element-wise ``b // a``"""
        raise NotImplementedError

    @abstractmethod
    def __rmod__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        """Return element-wise ``b % a``"""
        raise NotImplementedError

    @abstractmethod
    def __rdivmod__(self, other: IntegralMatrixLike[M_co, N_co]) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]:
        """Return element-wise ``divmod(b, a)``"""
        raise NotImplementedError

    @abstractmethod
    def __neg__(self) -> RealMatrixLike[M_co, N_co]:
        """Return element-wise ``-a``"""
        raise NotImplementedError

    @abstractmethod
    def __abs__(self) -> RealMatrixLike[M_co, N_co]:
        """Return element-wise ``abs(a)``"""
        raise NotImplementedError

    def __pos__(self) -> RealMatrixLike[M_co, N_co]:
        """Return element-wise ``+a``"""
        return copy.copy(self)

    @abstractmethod
    def transpose(self) -> RealMatrixLike[N_co, M_co]:
        raise NotImplementedError

    @abstractmethod
    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[M_co, N_co]:
        raise NotImplementedError

    @abstractmethod
    def reverse(self) -> RealMatrixLike[M_co, N_co]:
        raise NotImplementedError

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[RealMatrixLike[Literal[1], N_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[RealMatrixLike[M_co, Literal[1]]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[RealMatrixLike[Any, Any]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[RealMatrixLike[Literal[1], N_co]]: ...

    def slices(self, *, by=Rule.ROW, reverse=False):
        yield from super().slices(by=by, reverse=reverse)

    @overload
    @abstractmethod
    def lesser(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def lesser(self, other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def lesser(self, other):
        """Return element-wise ``a < b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def lesser_equal(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def lesser_equal(self, other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def lesser_equal(self, other):
        """Return element-wise ``a <= b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def greater(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def greater(self, other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def greater(self, other):
        """Return element-wise ``a > b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def greater_equal(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def greater_equal(self, other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def greater_equal(self, other):
        """Return element-wise ``a >= b``"""
        raise NotImplementedError

    def conjugate(self) -> RealMatrixLike[M_co, N_co]:
        """Return element-wise ``a.conjugate()``"""
        return copy.copy(self)

    def transjugate(self) -> RealMatrixLike[N_co, M_co]:
        """Return the conjugate transpose"""
        return self.transpose()


class IntegralMatrixLike(MatrixLike[int, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> int: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> IntegralMatrixLike[Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> int: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> IntegralMatrixLike[Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> IntegralMatrixLike[Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> IntegralMatrixLike[Any, Any]: ...

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    def __lt__(self, other: IntegralMatrixLike) -> bool:
        """Return true if lexicographic ``a < b``, otherwise false"""
        if isinstance(other, IntegralMatrixLike):
            return lexicographic_compare(self, other) < 0
        return NotImplemented

    def __le__(self, other: IntegralMatrixLike) -> bool:
        """Return true if lexicographic ``a <= b``, otherwise false"""
        if isinstance(other, IntegralMatrixLike):
            return lexicographic_compare(self, other) <= 0
        return NotImplemented

    def __gt__(self, other: IntegralMatrixLike) -> bool:
        """Return true if lexicographic ``a > b``, otherwise false"""
        if isinstance(other, IntegralMatrixLike):
            return lexicographic_compare(self, other) > 0
        return NotImplemented

    def __ge__(self, other: IntegralMatrixLike) -> bool:
        """Return true if lexicographic ``a >= b``, otherwise false"""
        if isinstance(other, IntegralMatrixLike):
            return lexicographic_compare(self, other) >= 0
        return NotImplemented

    @abstractmethod
    def __add__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a + b``"""
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a - b``"""
        raise NotImplementedError

    @abstractmethod
    def __mul__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a * b``"""
        raise NotImplementedError

    @abstractmethod
    def __matmul__(self, other: IntegralMatrixLike[N_co, P_co]) -> IntegralMatrixLike[M_co, P_co]:
        """Return the matrix product"""
        raise NotImplementedError

    @abstractmethod
    def __truediv__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        """Return element-wise ``a / b``"""
        raise NotImplementedError

    @abstractmethod
    def __floordiv__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a // b``"""
        raise NotImplementedError

    @abstractmethod
    def __mod__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a % b``"""
        raise NotImplementedError

    @abstractmethod
    def __divmod__(self, other: IntegralMatrixLike[M_co, N_co]) -> tuple[IntegralMatrixLike[M_co, N_co], IntegralMatrixLike[M_co, N_co]]:
        """Return element-wise ``divmod(a, b)``"""
        raise NotImplementedError

    @abstractmethod
    def __lshift__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a << b``"""
        raise NotImplementedError

    @abstractmethod
    def __rshift__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a >> b``"""
        raise NotImplementedError

    @abstractmethod
    def __and__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a & b``"""
        raise NotImplementedError

    @abstractmethod
    def __xor__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a ^ b``"""
        raise NotImplementedError

    @abstractmethod
    def __or__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a | b``"""
        raise NotImplementedError

    @abstractmethod
    def __neg__(self) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``-a``"""
        raise NotImplementedError

    @abstractmethod
    def __abs__(self) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``abs(a)``"""
        raise NotImplementedError

    @abstractmethod
    def __invert__(self) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``~a``"""
        raise NotImplementedError

    def __pos__(self) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``+a``"""
        return copy.copy(self)

    @abstractmethod
    def transpose(self) -> IntegralMatrixLike[N_co, M_co]:
        raise NotImplementedError

    @abstractmethod
    def flip(self, *, by: Rule = Rule.ROW) -> IntegralMatrixLike[M_co, N_co]:
        raise NotImplementedError

    @abstractmethod
    def reverse(self) -> IntegralMatrixLike[M_co, N_co]:
        raise NotImplementedError

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[IntegralMatrixLike[Literal[1], N_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[IntegralMatrixLike[M_co, Literal[1]]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[IntegralMatrixLike[Any, Any]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[IntegralMatrixLike[Literal[1], N_co]]: ...

    def slices(self, *, by=Rule.ROW, reverse=False):
        yield from super().slices(by=by, reverse=reverse)

    @abstractmethod
    def lesser(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a < b``"""
        raise NotImplementedError

    @abstractmethod
    def lesser_equal(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a <= b``"""
        raise NotImplementedError

    @abstractmethod
    def greater(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a > b``"""
        raise NotImplementedError

    @abstractmethod
    def greater_equal(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a >= b``"""
        raise NotImplementedError

    def conjugate(self) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a.conjugate()``"""
        return copy.copy(self)

    def transjugate(self) -> IntegralMatrixLike[N_co, M_co]:
        """Return the conjugate transpose"""
        return self.transpose()


class TimedeltaMatrixLike(MatrixLike[timedelta, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> timedelta: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> TimedeltaMatrixLike[Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> timedelta: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> TimedeltaMatrixLike[Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> TimedeltaMatrixLike[Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> TimedeltaMatrixLike[Any, Any]: ...

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    def __lt__(self, other: TimedeltaMatrixLike) -> bool:
        """Return true if lexicographic ``a < b``, otherwise false"""
        if isinstance(other, TimedeltaMatrixLike):
            return lexicographic_compare(self, other) < 0
        return NotImplemented

    def __le__(self, other: TimedeltaMatrixLike) -> bool:
        """Return true if lexicographic ``a <= b``, otherwise false"""
        if isinstance(other, TimedeltaMatrixLike):
            return lexicographic_compare(self, other) <= 0
        return NotImplemented

    def __gt__(self, other: TimedeltaMatrixLike) -> bool:
        """Return true if lexicographic ``a > b``, otherwise false"""
        if isinstance(other, TimedeltaMatrixLike):
            return lexicographic_compare(self, other) > 0
        return NotImplemented

    def __ge__(self, other: TimedeltaMatrixLike) -> bool:
        """Return true if lexicographic ``a >= b``, otherwise false"""
        if isinstance(other, TimedeltaMatrixLike):
            return lexicographic_compare(self, other) >= 0
        return NotImplemented

    @abstractmethod
    def __add__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]:
        """Return element-wise ``a + b``"""
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]:
        """Return element-wise ``a - b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __mul__(self, other: IntegralMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mul__(self, other: RealMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __mul__(self, other):
        """Return element-wise ``a * b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __truediv__(self, other: IntegralMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self, other: RealMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __truediv__(self, other):
        """Return element-wise ``a / b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __floordiv__(self, other: IntegralMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __floordiv__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __floordiv__(self, other):
        """Return element-wise ``a // b``"""
        raise NotImplementedError

    @abstractmethod
    def __mod__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]:
        """Return element-wise ``a % b``"""
        raise NotImplementedError

    @abstractmethod
    def __divmod__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> tuple[IntegralMatrixLike[M_co, N_co], TimedeltaMatrixLike[M_co, N_co]]:
        """Return element-wise ``divmod(a, b)``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __rmul__(self, other: IntegralMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmul__(self, other: RealMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __rmul__(self, other):
        """Return element-wise ``b * a``"""
        raise NotImplementedError

    @abstractmethod
    def __neg__(self) -> TimedeltaMatrixLike[M_co, N_co]:
        """Return element-wise ``-a``"""
        raise NotImplementedError

    @abstractmethod
    def __abs__(self) -> TimedeltaMatrixLike[M_co, N_co]:
        """Return element-wise ``abs(a)``"""
        raise NotImplementedError

    def __pos__(self) -> TimedeltaMatrixLike[M_co, N_co]:
        """Return element-wise ``+a``"""
        return copy.copy(self)

    @abstractmethod
    def lesser(self, other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a < b``"""
        raise NotImplementedError

    @abstractmethod
    def lesser_equal(self, other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a <= b``"""
        raise NotImplementedError

    @abstractmethod
    def greater(self, other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a > b``"""
        raise NotImplementedError

    @abstractmethod
    def greater_equal(self, other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a >= b``"""
        raise NotImplementedError


class DatetimeMatrixLike(MatrixLike[datetime, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> datetime: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> DatetimeMatrixLike[Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> datetime: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> DatetimeMatrixLike[Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> DatetimeMatrixLike[Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> DatetimeMatrixLike[Any, Any]: ...

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    def __lt__(self, other: DatetimeMatrixLike) -> bool:
        """Return true if lexicographic ``a < b``, otherwise false"""
        if isinstance(other, DatetimeMatrixLike):
            return lexicographic_compare(self, other) < 0
        return NotImplemented

    def __le__(self, other: DatetimeMatrixLike) -> bool:
        """Return true if lexicographic ``a <= b``, otherwise false"""
        if isinstance(other, DatetimeMatrixLike):
            return lexicographic_compare(self, other) <= 0
        return NotImplemented

    def __gt__(self, other: DatetimeMatrixLike) -> bool:
        """Return true if lexicographic ``a > b``, otherwise false"""
        if isinstance(other, DatetimeMatrixLike):
            return lexicographic_compare(self, other) > 0
        return NotImplemented

    def __ge__(self, other: DatetimeMatrixLike) -> bool:
        """Return true if lexicographic ``a >= b``, otherwise false"""
        if isinstance(other, DatetimeMatrixLike):
            return lexicographic_compare(self, other) >= 0
        return NotImplemented

    @abstractmethod
    def __add__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]:
        """Return element-wise ``a + b``"""
        raise NotImplementedError

    @overload
    @abstractmethod
    def __sub__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    @abstractmethod
    def __sub__(self, other: DatetimeMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...

    @abstractmethod
    def __sub__(self, other):
        """Return element-wise ``a - b``"""
        raise NotImplementedError

    @abstractmethod
    def __radd__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]:
        """Return element-wise ``b + a``"""
        raise NotImplementedError

    @abstractmethod
    def __rsub__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> DatetimeMatrixLike[M_co, N_co]:
        """Return element-wise ``b - a``"""
        raise NotImplementedError

    @abstractmethod
    def lesser(self, other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a < b``"""
        raise NotImplementedError

    @abstractmethod
    def lesser_equal(self, other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a <= b``"""
        raise NotImplementedError

    @abstractmethod
    def greater(self, other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a > b``"""
        raise NotImplementedError

    @abstractmethod
    def greater_equal(self, other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        """Return element-wise ``a >= b``"""
        raise NotImplementedError


class SupportsLT(Protocol[T_contra]):
    def __lt__(self, other: T_contra) -> Any: ...
class SupportsGT(Protocol[T_contra]):
    def __gt__(self, other: T_contra) -> Any: ...


@overload
def lexicographic_compare(
    a: ShapedIterable[SupportsLT[T], Any, Any],
    b: ShapedIterable[T, Any, Any],
    /,
) -> Literal[-1, 0, 1]: ...
@overload
def lexicographic_compare(
    a: ShapedIterable[T, Any, Any],
    b: ShapedIterable[SupportsGT[T], Any, Any],
    /,
) -> Literal[-1, 0, 1]: ...

def lexicographic_compare(a, b, /):
    """Return literal -1, 0, or 1 if two matrices lexicographically compare as
    ``a < b``, ``a = b``, or ``a > b``, respectively

    This function is used to implement the base comparison operators for
    some matrix types.
    """
    if a is b:
        return 0
    for x, y in zip(a, b):
        if x is y or x == y:
            continue
        return -1 if x < y else 1
    u = a.shape
    v = b.shape
    if u is v:
        return 0
    for m, n in zip(u, v):
        if m is n or m == n:
            continue
        return -1 if m < n else 1
    return 0
