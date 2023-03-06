from __future__ import annotations

import itertools
import operator
from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timedelta
from typing import (Any, Generic, Literal, Optional, Protocol, SupportsIndex,
                    TypeVar, overload)

from typing_extensions import Self

from .abc import (ComplexMatrixLike, DatetimeMatrixLike, IntegralMatrixLike,
                  MatrixLike, RealMatrixLike, Shaped, ShapedIterable,
                  StringMatrixLike, TimedeltaMatrixLike)
from .rule import COL, ROW, Rule

__all__ = [
    "MatrixOperatorsMixin",
    "Matrix",
    "ComplexMatrixOperatorsMixin",
    "ComplexMatrix",
    "RealMatrixOperatorsMixin",
    "RealMatrix",
    "IntegralMatrixOperatorsMixin",
    "IntegralMatrix",
    "TimedeltaMatrixOperatorsMixin",
    "TimedeltaMatrix",
    "DatetimeMatrixOperatorsMixin",
    "DatetimeMatrix",
]

T = TypeVar("T")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
P = TypeVar("P", bound=int)

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)
P_co = TypeVar("P_co", covariant=True, bound=int)


class SupportsMatrixParts(Shaped[M_co, N_co], Protocol[T_co, M_co, N_co]):

    @property
    @abstractmethod
    def array(self) -> Sequence[T_co]:
        raise NotImplementedError


class MatrixOperatorsMixin(Generic[T_co, M_co, N_co]):
    """Mixin class that defines the "operator methods" of ``MatrixLike`` using
    built-in matrix types
    """

    __slots__ = ()

    @overload
    def equal(self: SupportsMatrixParts[T_co, M_co, N_co], other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def equal(self: SupportsMatrixParts[T_co, M_co, N_co], other: Any) -> IntegralMatrixLike[M_co, N_co]: ...

    def equal(self, other):
        a = self.array
        if isinstance(other, MatrixLike):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__eq__, a, b),
            shape=shape,
        )

    @overload
    def not_equal(self: SupportsMatrixParts[T_co, M_co, N_co], other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def not_equal(self: SupportsMatrixParts[T_co, M_co, N_co], other: Any) -> IntegralMatrixLike[M_co, N_co]: ...

    def not_equal(self, other):
        a = self.array
        if isinstance(other, MatrixLike):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__ne__, a, b),
            shape=shape,
        )


class Matrix(MatrixOperatorsMixin[T_co, M_co, N_co], MatrixLike[T_co, M_co, N_co]):

    __slots__ = ("_array", "_shape")

    @overload
    def __new__(cls, array: ShapedIterable[T_co, M_co, N_co]) -> Self: ...
    @overload
    def __new__(cls, array: Iterable[T_co] = (), shape: Optional[tuple[M_co, N_co]] = None) -> Self: ...

    def __new__(cls, array=(), shape=None):
        """Create a matrix from the contents of ``array``, interpreting its
        dimensions as ``shape``

        If ``shape`` is unspecified or ``None``, the matrix will be a row
        vector with dimensions ``(1, N)``, where ``N`` is the size of
        ``array``. Note that this case will also apply when ``array`` is empty
        or unspecified (meaning that an empty construction call will result in
        a ``(1, 0)`` matrix being created).

        Raises ``ValueError`` if any of the given dimensions are negative, or
        if their product does not equal the size of ``array``.

        This constructor can be used as a means to cast between matrix types
        quickly. Casting between sub-classes of ``Matrix``, specifically, is an
        O(1) operation due to immutable storage optimizations.
        """
        if type(array) is cls:
            return array

        self = super().__new__(cls)

        if isinstance(array, Matrix):
            self._array = array._array
            self._shape = array._shape
            return self

        self._array = tuple(array)
        try:
            self._shape = array.shape
        except AttributeError:
            pass
        else:
            return self

        size = len(self._array)

        if shape is None:
            self._shape = (1, size)
            return self

        nrows, ncols = shape

        if nrows < 0 or ncols < 0:
            raise ValueError("shape must contain non-negative values")
        if size != nrows * ncols:
            raise ValueError(f"cannot interpret iterable of size {size} as shape {shape}")

        self._shape = shape

        return self

    def __repr__(self) -> str:
        """Return a canonical representation of the matrix"""
        return f"{self.__class__.__name__}(array={self.array!r}, shape={self.shape!r})"

    @overload
    def __getitem__(self, key: SupportsIndex) -> T_co: ...
    @overload
    def __getitem__(self, key: slice) -> MatrixLike[T_co, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, SupportsIndex]) -> T_co: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, slice]) -> MatrixLike[T_co, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, SupportsIndex]) -> MatrixLike[T_co, Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> MatrixLike[T_co, Any, Any]: ...

    def __getitem__(self, key):
        array = self.array

        if isinstance(key, (tuple, list)):
            row_key, col_key = key
            ncols = self.ncols

            if isinstance(row_key, slice):
                row_indices = self._resolve_matrix_slice(row_key, by=ROW)

                if isinstance(col_key, slice):
                    col_indices = self._resolve_matrix_slice(col_key, by=COL)
                    return self.__class__(
                        array=(
                            array[row_index * ncols + col_index]
                            for row_index in row_indices
                            for col_index in col_indices
                        ),
                        shape=(len(row_indices), len(col_indices)),
                    )

                col_index = self._resolve_matrix_index(col_key, by=COL)
                return self.__class__(
                    array=(
                        array[row_index * ncols + col_index]
                        for row_index in row_indices
                    ),
                    shape=(len(row_indices), 1),
                )

            row_index = self._resolve_matrix_index(row_key, by=ROW)

            if isinstance(col_key, slice):
                col_indices = self._resolve_matrix_slice(col_key, by=COL)
                return self.__class__(
                    array=(
                        array[row_index * ncols + col_index]
                        for col_index in col_indices
                    ),
                    shape=(1, len(col_indices)),
                )

            col_index = self._resolve_matrix_index(col_key, by=COL)
            return array[row_index * ncols + col_index]

        if isinstance(key, slice):
            val_indices = self._resolve_vector_slice(key)
            return self.__class__(
                array=(
                    array[val_index]
                    for val_index in val_indices
                ),
                shape=(1, len(val_indices)),
            )

        val_index = self._resolve_vector_index(key)
        return array[val_index]

    def __hash__(self) -> int:
        """Return a hash of the matrix

        Instances of ``Matrix`` are only hashable if their elements are
        hashable.
        """
        return hash((self.array, self.shape))

    @classmethod
    def from_nesting(cls, nesting: Iterable[Iterable[T_co]]) -> Self:
        """Construct a matrix from a singly-nested iterable, using the
        shallowest iterable's length to deduce the number of rows, and the
        nested iterables' lengths to deduce the number of columns

        Raises ``ValueError`` if the length of the nested iterables is
        inconsistent.
        """
        array: list[T_co] = []

        nrows = 0
        ncols = 0

        rows = iter(nesting)
        try:
            row = next(rows)
        except StopIteration:
            return cls(array=array, shape=(nrows, ncols))
        else:
            array.extend(row)

        nrows += 1
        ncols += len(array)

        for row in rows:
            n = 0
            for val in row:
                array.append(val)
                n += 1
            if ncols != n:
                raise ValueError(f"row at index {nrows} has length {n}, but precedent rows have length {ncols}")
            nrows += 1

        return cls(array=array, shape=(nrows, ncols))

    @classmethod
    def from_indicator(cls, indicator: Callable[[int, int], T_co], shape: tuple[M_co, N_co]) -> Self:
        """Construct a matrix from an ``indicator`` function that maps row,
        column index pairings to a value of the matrix
        """
        nrows, ncols = shape
        ix = range(nrows)
        jx = range(ncols)
        return cls(
            array=tuple(
                indicator(i, j)
                for i in ix
                for j in jx
            ),
            shape=shape,
        )

    @property
    def array(self) -> tuple[T_co, ...]:
        return self._array  # type: ignore[attr-defined]

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self._shape  # type: ignore[attr-defined]

    def transpose(self) -> MatrixLike[T_co, N_co, M_co]:
        raise NotImplementedError

    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T_co, M_co, N_co]:
        raise NotImplementedError

    def reverse(self) -> MatrixLike[T_co, M_co, N_co]:
        raise NotImplementedError


class StringMatrixOperatorsMixin(Generic[M_co, N_co]):

    __slots__ = ()

    @overload
    def __add__(self: SupportsMatrixParts[str, M_co, N_co], other: StringMatrixLike[M_co, N_co]) -> StringMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: SupportsMatrixParts[str, M_co, N_co], other: str) -> StringMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        a = self.array
        if isinstance(other, StringMatrixLike):
            b = other.array
        elif isinstance(other, str):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return StringMatrix(
            array=map(operator.__add__, a, b),
            shape=shape,
        )

    @overload
    def __mul__(self: SupportsMatrixParts[str, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> StringMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[str, M_co, N_co], other: int) -> StringMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return StringMatrix(
            array=map(operator.__mul__, a, b),
            shape=shape,
        )

    def __radd__(self: SupportsMatrixParts[str, M_co, N_co], other: str) -> StringMatrixLike[M_co, N_co]:
        a = self.array
        if isinstance(other, str):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return StringMatrix(
            array=map(operator.__add__, b, a),
            shape=shape,
        )

    @overload
    def lesser(self: SupportsMatrixParts[str, M_co, N_co], other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: SupportsMatrixParts[str, M_co, N_co], other: str) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        a = self.array
        if isinstance(other, StringMatrixLike):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__lt__, a, b),
            shape=shape,
        )

    @overload
    def lesser_equal(self: SupportsMatrixParts[str, M_co, N_co], other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: SupportsMatrixParts[str, M_co, N_co], other: str) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        a = self.array
        if isinstance(other, StringMatrixLike):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__le__, a, b),
            shape=shape,
        )

    @overload
    def greater(self: SupportsMatrixParts[str, M_co, N_co], other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: SupportsMatrixParts[str, M_co, N_co], other: str) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        a = self.array
        if isinstance(other, StringMatrixLike):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__gt__, a, b),
            shape=shape,
        )

    @overload
    def greater_equal(self: SupportsMatrixParts[str, M_co, N_co], other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: SupportsMatrixParts[str, M_co, N_co], other: str) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        a = self.array
        if isinstance(other, StringMatrixLike):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__ge__, a, b),
            shape=shape,
        )


class StringMatrix(StringMatrixOperatorsMixin[M_co, N_co], StringMatrixLike[M_co, N_co], Matrix[str, M_co, N_co]):

    __slots__ = ()

    @overload
    def __getitem__(self, key: SupportsIndex) -> str: ...
    @overload
    def __getitem__(self, key: slice) -> StringMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, SupportsIndex]) -> str: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, slice]) -> StringMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, SupportsIndex]) -> StringMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> StringMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        return Matrix.__getitem__(self, key)


class ComplexMatrixOperatorsMixin(Generic[M_co, N_co]):
    """Mixin class that defines the "operator methods" of ``ComplexMatrixLike``
    using built-in matrix types
    """

    __slots__ = ()

    @overload
    def __add__(self: SupportsMatrixParts[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: SupportsMatrixParts[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: SupportsMatrixParts[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: SupportsMatrixParts[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: SupportsMatrixParts[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: SupportsMatrixParts[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, ComplexMatrixLike)):
            b = other.array
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__add__, a, b),
            shape=shape,
        )

    @overload
    def __sub__(self: SupportsMatrixParts[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, ComplexMatrixLike)):
            b = other.array
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__sub__, a, b),
            shape=shape,
        )

    @overload
    def __mul__(self: SupportsMatrixParts[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, ComplexMatrixLike)):
            b = other.array
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__mul__, a, b),
            shape=shape,
        )

    @overload
    def __matmul__(self: SupportsMatrixParts[complex, M_co, N_co], other: IntegralMatrixLike[N_co, P_co]) -> ComplexMatrixLike[M_co, P_co]: ...
    @overload
    def __matmul__(self: SupportsMatrixParts[complex, M_co, N_co], other: RealMatrixLike[N_co, P_co]) -> ComplexMatrixLike[M_co, P_co]: ...
    @overload
    def __matmul__(self: SupportsMatrixParts[complex, M_co, N_co], other: ComplexMatrixLike[N_co, P_co]) -> ComplexMatrixLike[M_co, P_co]: ...

    def __matmul__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, ComplexMatrixLike)):
            return ComplexMatrix(matrix_dot(self, other))
        return NotImplemented

    @overload
    def __truediv__(self: SupportsMatrixParts[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __truediv__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, ComplexMatrixLike)):
            b = other.array
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__truediv__, a, b),
            shape=shape,
        )

    @overload
    def __radd__(self: SupportsMatrixParts[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: SupportsMatrixParts[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: SupportsMatrixParts[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: SupportsMatrixParts[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: SupportsMatrixParts[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __radd__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__add__, b, a),
            shape=shape,
        )

    @overload
    def __rsub__(self: SupportsMatrixParts[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: SupportsMatrixParts[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: SupportsMatrixParts[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: SupportsMatrixParts[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: SupportsMatrixParts[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rsub__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__sub__, b, a),
            shape=shape,
        )

    @overload
    def __rmul__(self: SupportsMatrixParts[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: SupportsMatrixParts[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: SupportsMatrixParts[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: SupportsMatrixParts[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: SupportsMatrixParts[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rmul__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__mul__, b, a),
            shape=shape,
        )

    @overload
    def __rmatmul__(self: SupportsMatrixParts[complex, M_co, N_co], other: IntegralMatrixLike[P_co, M_co]) -> ComplexMatrixLike[P_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: SupportsMatrixParts[complex, M_co, N_co], other: RealMatrixLike[P_co, M_co]) -> ComplexMatrixLike[P_co, N_co]: ...  # type: ignore[misc]

    def __rmatmul__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return ComplexMatrix(matrix_dot(other, self))
        return NotImplemented

    @overload
    def __rtruediv__(self: SupportsMatrixParts[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: SupportsMatrixParts[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: SupportsMatrixParts[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: SupportsMatrixParts[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: SupportsMatrixParts[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rtruediv__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__truediv__, b, a),
            shape=shape,
        )

    def __neg__(self: SupportsMatrixParts[complex, M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]:
        return ComplexMatrix(
            array=map(operator.__neg__, self.array),  # type: ignore[arg-type]
            shape=self.shape,
        )

    def __abs__(self: SupportsMatrixParts[complex, M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        return RealMatrix(
            array=map(abs, self.array),  # type: ignore[arg-type]
            shape=self.shape,
        )

    def conjugate(self: SupportsMatrixParts[complex, M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]:
        return ComplexMatrix(
            array=map(lambda x: x.conjugate(), self.array),
            shape=self.shape,
        )


class ComplexMatrix(ComplexMatrixOperatorsMixin[M_co, N_co], ComplexMatrixLike[M_co, N_co], Matrix[complex, M_co, N_co]):

    __slots__ = ()

    @overload
    def __getitem__(self, key: SupportsIndex) -> complex: ...
    @overload
    def __getitem__(self, key: slice) -> ComplexMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, SupportsIndex]) -> complex: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, slice]) -> ComplexMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, SupportsIndex]) -> ComplexMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> ComplexMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        return Matrix.__getitem__(self, key)

    def transpose(self) -> ComplexMatrixLike[N_co, M_co]:
        raise NotImplementedError

    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[M_co, N_co]:
        raise NotImplementedError

    def reverse(self) -> ComplexMatrixLike[M_co, N_co]:
        raise NotImplementedError


class RealMatrixOperatorsMixin(Generic[M_co, N_co]):
    """Mixin class that defines the "operator methods" of ``RealMatrixLike``
    using built-in matrix types
    """

    __slots__ = ()

    @overload
    def __add__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: SupportsMatrixParts[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__add__, a, b),
            shape=shape,
        )

    @overload
    def __sub__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__sub__, a, b),
            shape=shape,
        )

    @overload
    def __mul__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__mul__, a, b),
            shape=shape,
        )

    @overload
    def __matmul__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[N_co, P_co]) -> RealMatrixLike[M_co, P_co]: ...
    @overload
    def __matmul__(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[N_co, P_co]) -> RealMatrixLike[M_co, P_co]: ...

    def __matmul__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return RealMatrix(matrix_dot(self, other))
        return NotImplemented

    @overload
    def __truediv__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __truediv__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__truediv__, a, b),
            shape=shape,
        )

    @overload
    def __floordiv__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __floordiv__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        elif isinstance(other, (int, float)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return RealMatrix(
            array=map(operator.__floordiv__, a, b),
            shape=shape,
        )

    @overload
    def __mod__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __mod__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        elif isinstance(other, (int, float)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return RealMatrix(
            array=map(operator.__mod__, a, b),
            shape=shape,
        )

    @overload
    def __divmod__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...
    @overload
    def __divmod__(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...
    @overload
    def __divmod__(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...
    @overload
    def __divmod__(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...

    def __divmod__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        elif isinstance(other, (int, float)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        c, d = itertools.tee(map(divmod, a, b))
        shape = self.shape
        return (
            RealMatrix(
                array=map(operator.itemgetter(0), c),
                shape=shape,
            ),
            RealMatrix(
                array=map(operator.itemgetter(1), d),
                shape=shape,
            ),
        )

    @overload
    def __radd__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: SupportsMatrixParts[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __radd__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__add__, b, a),
            shape=shape,
        )

    @overload
    def __rsub__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: SupportsMatrixParts[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rsub__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__sub__, b, a),
            shape=shape,
        )

    @overload
    def __rmul__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: SupportsMatrixParts[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rmul__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__mul__, b, a),
            shape=shape,
        )

    def __rmatmul__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[P_co, M_co]) -> RealMatrixLike[P_co, N_co]:  # type: ignore[misc]
        if isinstance(other, IntegralMatrixLike):
            return RealMatrix(matrix_dot(other, self))
        return NotImplemented

    @overload
    def __rtruediv__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: SupportsMatrixParts[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rtruediv__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__truediv__, b, a),
            shape=shape,
        )

    @overload
    def __rfloordiv__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rfloordiv__(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __rfloordiv__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
        elif isinstance(other, (int, float)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return RealMatrix(
            array=map(operator.__floordiv__, b, a),
            shape=shape,
        )

    @overload
    def __rmod__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmod__(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __rmod__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
        elif isinstance(other, (int, float)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return RealMatrix(
            array=map(operator.__mod__, b, a),
            shape=shape,
        )

    @overload
    def __rdivmod__(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...
    @overload
    def __rdivmod__(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...

    def __rdivmod__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
        elif isinstance(other, (int, float)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        c, d = itertools.tee(map(divmod, b, a))
        shape = self.shape
        return (
            RealMatrix(
                array=map(operator.itemgetter(0), c),
                shape=shape,
            ),
            RealMatrix(
                array=map(operator.itemgetter(1), d),
                shape=shape,
            ),
        )

    def __neg__(self: SupportsMatrixParts[float, M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        return RealMatrix(
            array=map(operator.__neg__, self.array),  # type: ignore[arg-type]
            shape=self.shape,
        )

    def __abs__(self: SupportsMatrixParts[float, M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        return RealMatrix(
            array=map(abs, self.array),  # type: ignore[arg-type]
            shape=self.shape,
        )

    @overload
    def lesser(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__lt__, a, b),
            shape=shape,
        )

    @overload
    def lesser_equal(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__le__, a, b),
            shape=shape,
        )

    @overload
    def greater(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__gt__, a, b),
            shape=shape,
        )

    @overload
    def greater_equal(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__ge__, a, b),
            shape=shape,
        )


class RealMatrix(RealMatrixOperatorsMixin[M_co, N_co], RealMatrixLike[M_co, N_co], Matrix[float, M_co, N_co]):

    __slots__ = ()

    @overload
    def __getitem__(self, key: SupportsIndex) -> float: ...
    @overload
    def __getitem__(self, key: slice) -> RealMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, SupportsIndex]) -> float: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, slice]) -> RealMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, SupportsIndex]) -> RealMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> RealMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        return Matrix.__getitem__(self, key)

    def transpose(self) -> RealMatrixLike[N_co, M_co]:
        raise NotImplementedError

    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[M_co, N_co]:
        raise NotImplementedError

    def reverse(self) -> RealMatrixLike[M_co, N_co]:
        raise NotImplementedError


class IntegralMatrixOperatorsMixin(Generic[M_co, N_co]):
    """Mixin class that defines the "operator methods" of
    ``IntegralMatrixLike`` using built-in matrix types
    """

    __slots__ = ()

    @overload
    def __add__(self: SupportsMatrixParts[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: SupportsMatrixParts[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: SupportsMatrixParts[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
            Matrix = IntegralMatrix
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            elif isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__add__, a, b),
            shape=shape,
        )

    @overload
    def __sub__(self: SupportsMatrixParts[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
            Matrix = IntegralMatrix
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            elif isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__sub__, a, b),
            shape=shape,
        )

    @overload
    def __mul__(self: SupportsMatrixParts[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
            Matrix = IntegralMatrix
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            elif isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__mul__, a, b),
            shape=shape,
        )

    def __matmul__(self: SupportsMatrixParts[int, M_co, N_co], other: IntegralMatrixLike[N_co, P_co]) -> IntegralMatrixLike[M_co, P_co]:
        if isinstance(other, IntegralMatrixLike):
            return IntegralMatrix(matrix_dot(self, other))
        return NotImplemented

    @overload
    def __truediv__(self: SupportsMatrixParts[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __truediv__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
            Matrix = IntegralMatrix
        elif isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__truediv__, a, b),
            shape=shape,
        )

    @overload
    def __floordiv__(self: SupportsMatrixParts[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: SupportsMatrixParts[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __floordiv__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
            Matrix = IntegralMatrix
        elif isinstance(other, (int, float)):
            b = itertools.repeat(other)
            if isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__floordiv__, a, b),
            shape=shape,
        )

    @overload
    def __mod__(self: SupportsMatrixParts[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self: SupportsMatrixParts[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __mod__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
            Matrix = IntegralMatrix
        elif isinstance(other, (int, float)):
            b = itertools.repeat(other)
            if isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__mod__, a, b),
            shape=shape,
        )

    @overload
    def __divmod__(self: SupportsMatrixParts[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> tuple[IntegralMatrixLike[M_co, N_co], IntegralMatrixLike[M_co, N_co]]: ...
    @overload
    def __divmod__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> tuple[IntegralMatrixLike[M_co, N_co], IntegralMatrixLike[M_co, N_co]]: ...
    @overload
    def __divmod__(self: SupportsMatrixParts[int, M_co, N_co], other: float) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]: ...

    def __divmod__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
            Matrix = IntegralMatrix
        elif isinstance(other, (int, float)):
            b = itertools.repeat(other)
            if isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        c, d = itertools.tee(map(divmod, a, b))
        shape = self.shape
        return (
            Matrix(
                array=map(operator.itemgetter(0), c),
                shape=shape,
            ),
            Matrix(
                array=map(operator.itemgetter(1), d),
                shape=shape,
            ),
        )

    @overload
    def __lshift__(self: SupportsMatrixParts[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __lshift__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...

    def __lshift__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__lshift__, a, b),
            shape=shape,
        )

    @overload
    def __rshift__(self: SupportsMatrixParts[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __rshift__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...

    def __rshift__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__rshift__, a, b),
            shape=shape,
        )

    @overload
    def __and__(self: SupportsMatrixParts[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __and__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...

    def __and__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__and__, a, b),
            shape=shape,
        )

    @overload
    def __xor__(self: SupportsMatrixParts[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __xor__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...

    def __xor__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__xor__, a, b),
            shape=shape,
        )

    @overload
    def __or__(self: SupportsMatrixParts[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __or__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...

    def __or__(self, other):
        a = self.array
        if isinstance(other, IntegralMatrixLike):
            b = other.array
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__or__, a, b),
            shape=shape,
        )

    @overload
    def __radd__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: SupportsMatrixParts[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: SupportsMatrixParts[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __radd__(self, other):
        a = self.array
        if isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            elif isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__add__, b, a),
            shape=shape,
        )

    @overload
    def __rsub__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: SupportsMatrixParts[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: SupportsMatrixParts[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rsub__(self, other):
        a = self.array
        if isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            elif isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__sub__, b, a),
            shape=shape,
        )

    @overload
    def __rmul__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: SupportsMatrixParts[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: SupportsMatrixParts[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rmul__(self, other):
        a = self.array
        if isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            elif isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__mul__, b, a),
            shape=shape,
        )

    @overload
    def __rtruediv__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: SupportsMatrixParts[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: SupportsMatrixParts[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rtruediv__(self, other):
        a = self.array
        if isinstance(other, (int, float, complex)):
            b = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__truediv__, b, a),
            shape=shape,
        )

    @overload
    def __rfloordiv__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __rfloordiv__(self: SupportsMatrixParts[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __rfloordiv__(self, other):
        a = self.array
        if isinstance(other, (int, float)):
            b = itertools.repeat(other)
            if isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__floordiv__, b, a),
            shape=shape,
        )

    @overload
    def __rmod__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __rmod__(self: SupportsMatrixParts[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __rmod__(self, other):
        a = self.array
        if isinstance(other, (int, float)):
            b = itertools.repeat(other)
            if isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__mod__, b, a),
            shape=shape,
        )

    @overload
    def __rdivmod__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> tuple[IntegralMatrixLike[M_co, N_co], IntegralMatrixLike[M_co, N_co]]: ...
    @overload
    def __rdivmod__(self: SupportsMatrixParts[int, M_co, N_co], other: float) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]: ...

    def __rdivmod__(self, other):
        a = self.array
        if isinstance(other, (int, float)):
            b = itertools.repeat(other)
            if isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        c, d = itertools.tee(map(divmod, b, a))
        shape = self.shape
        return (
            Matrix(
                array=map(operator.itemgetter(0), c),
                shape=shape,
            ),
            Matrix(
                array=map(operator.itemgetter(1), d),
                shape=shape,
            ),
        )

    def __rlshift__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]:
        a = self.array
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__lshift__, b, a),
            shape=shape,
        )

    def __rrshift__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]:
        a = self.array
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__rshift__, b, a),
            shape=shape,
        )

    def __rand__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]:
        a = self.array
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__and__, b, a),
            shape=shape,
        )

    def __rxor__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]:
        a = self.array
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__xor__, b, a),
            shape=shape,
        )

    def __ror__(self: SupportsMatrixParts[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]:
        a = self.array
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__or__, b, a),
            shape=shape,
        )

    def __neg__(self: SupportsMatrixParts[int, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(
            array=map(operator.__neg__, self.array),  # type: ignore[arg-type]
            shape=self.shape,
        )

    def __abs__(self: SupportsMatrixParts[int, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(
            array=map(abs, self.array),  # type: ignore[arg-type]
            shape=self.shape,
        )

    def __invert__(self: SupportsMatrixParts[int, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(
            array=map(operator.__invert__, self.array),  # type: ignore[arg-type]
            shape=self.shape,
        )

    @overload
    def lesser(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__lt__, a, b),
            shape=shape,
        )

    @overload
    def lesser_equal(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__le__, a, b),
            shape=shape,
        )

    @overload
    def greater(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__gt__, a, b),
            shape=shape,
        )

    @overload
    def greater_equal(self: SupportsMatrixParts[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: SupportsMatrixParts[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: SupportsMatrixParts[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: SupportsMatrixParts[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__ge__, a, b),
            shape=shape,
        )


class IntegralMatrix(IntegralMatrixOperatorsMixin[M_co, N_co], IntegralMatrixLike[M_co, N_co], Matrix[int, M_co, N_co]):

    __slots__ = ()

    @overload
    def __getitem__(self, key: SupportsIndex) -> int: ...
    @overload
    def __getitem__(self, key: slice) -> IntegralMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, SupportsIndex]) -> int: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, slice]) -> IntegralMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, SupportsIndex]) -> IntegralMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> IntegralMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        return Matrix.__getitem__(self, key)

    def transpose(self) -> IntegralMatrixLike[N_co, M_co]:
        raise NotImplementedError

    def flip(self, *, by: Rule = Rule.ROW) -> IntegralMatrixLike[M_co, N_co]:
        raise NotImplementedError

    def reverse(self) -> IntegralMatrixLike[M_co, N_co]:
        raise NotImplementedError


class TimedeltaMatrixOperatorsMixin(Generic[M_co, N_co]):

    __slots__ = ()

    @overload
    def __add__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: timedelta) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        a = self.array
        if isinstance(other, TimedeltaMatrixLike):
            b = other.array
        elif isinstance(other, timedelta):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return TimedeltaMatrix(
            array=map(operator.__add__, a, b),
            shape=shape,
        )

    @overload
    def __sub__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: timedelta) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        a = self.array
        if isinstance(other, TimedeltaMatrixLike):
            b = other.array
        elif isinstance(other, timedelta):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return TimedeltaMatrix(
            array=map(operator.__sub__, a, b),
            shape=shape,
        )

    @overload
    def __mul__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: int) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: float) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            b = other.array
        elif isinstance(other, (int, float)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return TimedeltaMatrix(
            array=map(operator.__mul__, a, b),
            shape=shape,
        )

    @overload
    def __truediv__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: int) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: float) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: timedelta) -> RealMatrixLike[M_co, N_co]: ...

    def __truediv__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, TimedeltaMatrixLike)):
            b = other.array
            if isinstance(other, TimedeltaMatrixLike):
                Matrix = RealMatrix
            else:
                Matrix = TimedeltaMatrix
        elif isinstance(other, (int, float, timedelta)):
            b = itertools.repeat(other)
            if isinstance(other, timedelta):
                Matrix = RealMatrix
            else:
                Matrix = TimedeltaMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__truediv__, a, b),
            shape=shape,
        )

    @overload
    def __floordiv__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: int) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: timedelta) -> IntegralMatrixLike[M_co, N_co]: ...

    def __floordiv__(self, other):
        a = self.array
        if isinstance(other, (IntegralMatrixLike, TimedeltaMatrixLike)):
            b = other.array
            if isinstance(other, TimedeltaMatrixLike):
                Matrix = IntegralMatrix
            else:
                Matrix = TimedeltaMatrix
        elif isinstance(other, (int, timedelta)):
            b = itertools.repeat(other)
            if isinstance(other, timedelta):
                Matrix = IntegralMatrix
            else:
                Matrix = TimedeltaMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__floordiv__, a, b),
            shape=shape,
        )

    @overload
    def __mod__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: timedelta) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __mod__(self, other):
        a = self.array
        if isinstance(other, TimedeltaMatrixLike):
            b = other.array
        elif isinstance(other, timedelta):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return TimedeltaMatrix(
            array=map(operator.__mod__, a, b),
            shape=shape,
        )

    @overload
    def __divmod__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> tuple[IntegralMatrixLike[M_co, N_co], TimedeltaMatrixLike[M_co, N_co]]: ...
    @overload
    def __divmod__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: timedelta) -> tuple[IntegralMatrixLike[M_co, N_co], TimedeltaMatrixLike[M_co, N_co]]: ...

    def __divmod__(self, other):
        a = self.array
        if isinstance(other, TimedeltaMatrixLike):
            b = other.array
        elif isinstance(other, timedelta):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        c, d = itertools.tee(map(divmod, a, b))
        shape = self.shape
        return (
            IntegralMatrix(
                array=map(operator.itemgetter(0), c),
                shape=shape,
            ),
            TimedeltaMatrix(
                array=map(operator.itemgetter(1), d),
                shape=shape,
            ),
        )

    def __radd__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: timedelta) -> TimedeltaMatrixLike[M_co, N_co]:
        a = self.array
        if isinstance(other, timedelta):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return TimedeltaMatrix(
            array=map(operator.__add__, b, a),
            shape=shape,
        )

    def __rsub__(self: SupportsMatrixParts[timedelta, M_co, N_co], other: timedelta) -> TimedeltaMatrixLike[M_co, N_co]:
        a = self.array
        if isinstance(other, timedelta):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return TimedeltaMatrix(
            array=map(operator.__sub__, b, a),
            shape=shape,
        )

    def __neg__(self: SupportsMatrixParts[timedelta, M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]:
        return TimedeltaMatrix(
            array=map(operator.__neg__, self.array),  # type: ignore[arg-type]
            shape=self.shape,
        )

    def __abs__(self: SupportsMatrixParts[timedelta, M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]:
        return TimedeltaMatrix(
            array=map(abs, self.array),  # type: ignore[arg-type]
            shape=self.shape,
        )

    @overload
    def lesser(self: SupportsMatrixParts[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: SupportsMatrixParts[timedelta, M_co, N_co], other: timedelta) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        a = self.array
        if isinstance(other, TimedeltaMatrixLike):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__lt__, a, b),
            shape=shape,
        )

    @overload
    def lesser_equal(self: SupportsMatrixParts[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: SupportsMatrixParts[timedelta, M_co, N_co], other: timedelta) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        a = self.array
        if isinstance(other, TimedeltaMatrixLike):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__le__, a, b),
            shape=shape,
        )

    @overload
    def greater(self: SupportsMatrixParts[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: SupportsMatrixParts[timedelta, M_co, N_co], other: timedelta) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        a = self.array
        if isinstance(other, TimedeltaMatrixLike):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__gt__, a, b),
            shape=shape,
        )

    @overload
    def greater_equal(self: SupportsMatrixParts[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: SupportsMatrixParts[timedelta, M_co, N_co], other: timedelta) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        a = self.array
        if isinstance(other, TimedeltaMatrixLike):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__ge__, a, b),
            shape=shape,
        )


class TimedeltaMatrix(TimedeltaMatrixOperatorsMixin[M_co, N_co], TimedeltaMatrixLike[M_co, N_co], Matrix[timedelta, M_co, N_co]):

    __slots__ = ()

    @overload
    def __getitem__(self, key: SupportsIndex) -> timedelta: ...
    @overload
    def __getitem__(self, key: slice) -> TimedeltaMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, SupportsIndex]) -> timedelta: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, slice]) -> TimedeltaMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, SupportsIndex]) -> TimedeltaMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> TimedeltaMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        return Matrix.__getitem__(self, key)


class DatetimeMatrixOperatorsMixin(Generic[M_co, N_co]):

    __slots__ = ()

    @overload
    def __add__(self: SupportsMatrixParts[datetime, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: SupportsMatrixParts[datetime, M_co, N_co], other: timedelta) -> DatetimeMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        a = self.array
        if isinstance(other, TimedeltaMatrixLike):
            b = other.array
        elif isinstance(other, timedelta):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return DatetimeMatrix(
            array=map(operator.__add__, a, b),
            shape=shape,
        )

    @overload
    def __sub__(self: SupportsMatrixParts[datetime, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[datetime, M_co, N_co], other: timedelta) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[datetime, M_co, N_co], other: DatetimeMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: SupportsMatrixParts[datetime, M_co, N_co], other: datetime) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        a = self.array
        if isinstance(other, (TimedeltaMatrixLike, DatetimeMatrixLike)):
            b = other.array
            if isinstance(other, DatetimeMatrixLike):
                Matrix = TimedeltaMatrix
            else:
                Matrix = DatetimeMatrix
        elif isinstance(other, (timedelta, datetime)):
            b = itertools.repeat(other)
            if isinstance(other, datetime):
                Matrix = TimedeltaMatrix
            else:
                Matrix = DatetimeMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__sub__, a, b),
            shape=shape,
        )

    @overload
    def __rsub__(self: SupportsMatrixParts[datetime, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: SupportsMatrixParts[datetime, M_co, N_co], other: timedelta) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: SupportsMatrixParts[datetime, M_co, N_co], other: datetime) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __rsub__(self, other):
        a = self.array
        if isinstance(other, TimedeltaMatrixLike):
            b = other.array
            Matrix = DatetimeMatrix
        elif isinstance(other, (timedelta, datetime)):
            b = itertools.repeat(other)
            if isinstance(other, datetime):
                Matrix = TimedeltaMatrix
            else:
                Matrix = DatetimeMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__sub__, b, a),
            shape=shape,
        )

    @overload
    def lesser(self: SupportsMatrixParts[datetime, M_co, N_co], other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: SupportsMatrixParts[datetime, M_co, N_co], other: datetime) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        a = self.array
        if isinstance(other, DatetimeMatrixLike):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__lt__, a, b),
            shape=shape,
        )

    @overload
    def lesser_equal(self: SupportsMatrixParts[datetime, M_co, N_co], other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: SupportsMatrixParts[datetime, M_co, N_co], other: datetime) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        a = self.array
        if isinstance(other, DatetimeMatrixLike):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__le__, a, b),
            shape=shape,
        )

    @overload
    def greater(self: SupportsMatrixParts[datetime, M_co, N_co], other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: SupportsMatrixParts[datetime, M_co, N_co], other: datetime) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        a = self.array
        if isinstance(other, DatetimeMatrixLike):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__gt__, a, b),
            shape=shape,
        )

    @overload
    def greater_equal(self: SupportsMatrixParts[datetime, M_co, N_co], other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: SupportsMatrixParts[datetime, M_co, N_co], other: datetime) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        a = self.array
        if isinstance(other, DatetimeMatrixLike):
            b = other.array
        else:
            b = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__ge__, a, b),
            shape=shape,
        )


class DatetimeMatrix(DatetimeMatrixOperatorsMixin[M_co, N_co], DatetimeMatrixLike[M_co, N_co], Matrix[datetime, M_co, N_co]):

    __slots__ = ()

    @overload
    def __getitem__(self, key: SupportsIndex) -> datetime: ...
    @overload
    def __getitem__(self, key: slice) -> DatetimeMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, SupportsIndex]) -> datetime: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, slice]) -> DatetimeMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, SupportsIndex]) -> DatetimeMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> DatetimeMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        return Matrix.__getitem__(self, key)


def vector_dot(a, b):
    """Return the vector dot product of ``a`` and ``b``

    This function is primarily intended as a helper for ``matrix_dot()``.
    """
    return sum(map(operator.mul, a, b))


def matrix_dot(a, b):
    """Return the matrix dot product of ``a`` and ``b`` (AKA the matrix
    product)

    Function used to implement the ``__matmul__()``/``__rmatmul__()`` operators
    of the integral, real, and complex matrix operator mixins.

    Performs a variation of the naive matrix multiplication algorithm. This
    function makes a mild effort to keep a majority of the execution in C code,
    but can still be a timesink with larger matrices.

    Raises ``ValueError`` if the inner dimensions of ``a`` and ``b`` are
    unequal. Returns a basic ``Matrix`` instance (that you'd usually want to
    cast to a different sub-class - casting between ``Matrix`` sub-classes is
    an O(1) operation due to immutability optimizations).
    """
    m, n = a.shape
    p, q = b.shape

    if n != p:
        raise ValueError(f"incompatible shapes, ({n = }) != ({p = })")

    if n:
        a = a.array
        b = b.array

        mn = m * n
        pq = p * q

        ix = range(0, mn, n)
        jx = range(0,  q, 1)

        array = tuple(
            vector_dot(
                itertools.islice(a, i, i +  n, 1),
                itertools.islice(b, j, j + pq, q),
            )
            for i in ix
            for j in jx
        )
    else:
        array = (0,) * (m * q)

    shape = (m, q)

    return Matrix(array, shape)
