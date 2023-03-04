from __future__ import annotations

import itertools
import operator
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import (Any, Generic, Literal, Optional, SupportsIndex, TypeVar,
                    overload)

from typing_extensions import Self

from .abc import (ComplexMatrixLike, DatetimeMatrixLike, IntegralMatrixLike,
                  MatrixLike, RealMatrixLike, ShapedIndexable, ShapedIterable,
                  StringMatrixLike, TimedeltaMatrixLike)
from .rule import COL, ROW, Rule
from .utilities.matrix_product import MatrixProduct

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


class MatrixOperatorsMixin(Generic[T_co, M_co, N_co]):
    """Mixin class that defines the "operator methods" of ``MatrixLike`` using
    built-in matrix types
    """

    __slots__ = ()

    @overload
    def equal(self: ShapedIterable[T_co, M_co, N_co], other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def equal(self: ShapedIterable[T_co, M_co, N_co], other: Any) -> IntegralMatrixLike[M_co, N_co]: ...

    def equal(self, other):
        if isinstance(other, MatrixLike):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__eq__, self, args),
            shape=shape,
        )

    @overload
    def not_equal(self: ShapedIterable[T_co, M_co, N_co], other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def not_equal(self: ShapedIterable[T_co, M_co, N_co], other: Any) -> IntegralMatrixLike[M_co, N_co]: ...

    def not_equal(self, other):
        if isinstance(other, MatrixLike):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__ne__, self, args),
            shape=shape,
        )


class Matrix(MatrixOperatorsMixin[T_co, M_co, N_co], MatrixLike[T_co, M_co, N_co]):

    __slots__ = ("_array", "_shape")

    @overload
    def __init__(self, array: ShapedIterable[T_co, M_co, N_co]) -> None: ...
    @overload
    def __init__(self, array: Iterable[T_co] = (), shape: tuple[Optional[M_co], Optional[N_co]] = (None, None)) -> None: ...

    def __init__(self, array=(), shape=(None, None)):
        if isinstance(array, Matrix):
            self._array = array._array
            self._shape = array._shape
            return

        self._array = tuple(array)
        try:
            shape = array.shape  # Avoids having to instance-check with ShapedIterable,
        except AttributeError:   # which is painfully slow
            pass
        else:
            self._shape = shape
            return

        nrows = shape[0]
        ncols = shape[1]
        nvals = len(self._array)

        # We remake the shape in 3/4 cases (where one or both dimensions are None). If
        # both dimensions are given, we maintain a reference for potential savings
        remake = True

        if nrows is None and ncols is None:
            nrows = 1
            ncols = nvals
        elif nrows is None:
            nrows, loss = divmod(nvals, ncols) if ncols else (0, nvals)
            if loss:
                raise ValueError(f"cannot interpret array of size {nvals} as shape M × {ncols}")
        elif ncols is None:
            ncols, loss = divmod(nvals, nrows) if nrows else (0, nvals)
            if loss:
                raise ValueError(f"cannot interpret array of size {nvals} as shape {nrows} × N")
        else:
            remake = False
            if nvals != nrows * ncols:
                raise ValueError(f"cannot interpret array of size {nvals} as shape {nrows} × {ncols}")

        if nrows < 0 or ncols < 0:
            raise ValueError("dimensions must be non-negative")

        self._shape = (nrows, ncols) if remake else shape

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

        if isinstance(key, tuple):
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

    @property
    def array(self) -> tuple[T_co, ...]:
        return self._array

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self._shape

    def transpose(self) -> MatrixLike[T_co, N_co, M_co]:
        raise NotImplementedError
        from .views.builtins import MatrixTranspose
        return MatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T_co, M_co, N_co]:
        raise NotImplementedError
        from .views.builtins import MatrixColFlip, MatrixRowFlip
        MatrixTransform = (MatrixRowFlip, MatrixColFlip)[by.value]
        return MatrixTransform(self)

    def reverse(self) -> MatrixLike[T_co, M_co, N_co]:
        raise NotImplementedError
        from .views.builtins import MatrixReverse
        return MatrixReverse(self)


class StringMatrixOperatorsMixin(Generic[M_co, N_co]):

    __slots__ = ()

    @overload
    def __add__(self: ShapedIterable[str, M_co, N_co], other: StringMatrixLike[M_co, N_co]) -> StringMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[str, M_co, N_co], other: str) -> StringMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        if isinstance(other, StringMatrixLike):
            args = iter(other)
        elif isinstance(other, str):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return StringMatrix(
            array=map(operator.__add__, self, args),
            shape=shape,
        )

    @overload
    def __mul__(self: ShapedIterable[str, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> StringMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[str, M_co, N_co], other: int) -> StringMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args = iter(other)
        elif isinstance(other, int):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return StringMatrix(
            array=map(operator.__mul__, self, args),
            shape=shape,
        )

    def __radd__(self: ShapedIterable[str, M_co, N_co], other: str) -> StringMatrixLike[M_co, N_co]:
        if isinstance(other, str):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return StringMatrix(
            array=map(operator.__add__, args, self),
            shape=shape,
        )

    @overload
    def lesser(self: ShapedIterable[str, M_co, N_co], other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: ShapedIterable[str, M_co, N_co], other: str) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        if isinstance(other, StringMatrixLike):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__lt__, self, args),
            shape=shape,
        )

    @overload
    def lesser_equal(self: ShapedIterable[str, M_co, N_co], other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: ShapedIterable[str, M_co, N_co], other: str) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        if isinstance(other, StringMatrixLike):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__le__, self, args),
            shape=shape,
        )

    @overload
    def greater(self: ShapedIterable[str, M_co, N_co], other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: ShapedIterable[str, M_co, N_co], other: str) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        if isinstance(other, StringMatrixLike):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__gt__, self, args),
            shape=shape,
        )

    @overload
    def greater_equal(self: ShapedIterable[str, M_co, N_co], other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: ShapedIterable[str, M_co, N_co], other: str) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        if isinstance(other, StringMatrixLike):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__ge__, self, args),
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
    def __add__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, ComplexMatrixLike)):
            args = iter(other)
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__add__, self, args),
            shape=shape,
        )

    @overload
    def __sub__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, ComplexMatrixLike)):
            args = iter(other)
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__sub__, self, args),
            shape=shape,
        )

    @overload
    def __mul__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, ComplexMatrixLike)):
            args = iter(other)
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__mul__, self, args),
            shape=shape,
        )

    @overload
    def __matmul__(self: ShapedIndexable[complex, M_co, N_co], other: IntegralMatrixLike[N_co, P_co]) -> ComplexMatrixLike[M_co, P_co]: ...
    @overload
    def __matmul__(self: ShapedIndexable[complex, M_co, N_co], other: RealMatrixLike[N_co, P_co]) -> ComplexMatrixLike[M_co, P_co]: ...
    @overload
    def __matmul__(self: ShapedIndexable[complex, M_co, N_co], other: ComplexMatrixLike[N_co, P_co]) -> ComplexMatrixLike[M_co, P_co]: ...

    def __matmul__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, ComplexMatrixLike)):
            return ComplexMatrix(MatrixProduct(self, other))
        return NotImplemented

    @overload
    def __truediv__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __truediv__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, ComplexMatrixLike)):
            args = iter(other)
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__truediv__, self, args),
            shape=shape,
        )

    @overload
    def __radd__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: ShapedIterable[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: ShapedIterable[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: ShapedIterable[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __radd__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__add__, args, self),
            shape=shape,
        )

    @overload
    def __rsub__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: ShapedIterable[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: ShapedIterable[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: ShapedIterable[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rsub__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__sub__, args, self),
            shape=shape,
        )

    @overload
    def __rmul__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: ShapedIterable[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: ShapedIterable[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: ShapedIterable[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rmul__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__mul__, args, self),
            shape=shape,
        )

    @overload
    def __rmatmul__(self: ShapedIndexable[complex, M_co, N_co], other: IntegralMatrixLike[P_co, M_co]) -> ComplexMatrixLike[P_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: ShapedIndexable[complex, M_co, N_co], other: RealMatrixLike[P_co, M_co]) -> ComplexMatrixLike[P_co, N_co]: ...  # type: ignore[misc]

    def __rmatmul__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return ComplexMatrix(MatrixProduct(other, self))
        return NotImplemented

    @overload
    def __rtruediv__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: ShapedIterable[complex, M_co, N_co], other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: ShapedIterable[complex, M_co, N_co], other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: ShapedIterable[complex, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rtruediv__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__truediv__, args, self),
            shape=shape,
        )

    def __neg__(self: ShapedIterable[complex, M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]:
        shape = self.shape
        return ComplexMatrix(
            array=map(operator.__neg__, self),  # type: ignore[arg-type]
            shape=shape,
        )

    def __abs__(self: ShapedIterable[complex, M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        shape = self.shape
        return RealMatrix(
            array=map(abs, self),  # type: ignore[arg-type]
            shape=shape,
        )

    def conjugate(self: ShapedIterable[complex, M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]:
        shape = self.shape
        return ComplexMatrix(
            array=map(lambda x: x.conjugate(), self),
            shape=shape,
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
        from .views.builtins import ComplexMatrixTranspose
        return ComplexMatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[M_co, N_co]:
        raise NotImplementedError
        from .views.builtins import ComplexMatrixColFlip, ComplexMatrixRowFlip
        ComplexMatrixTransform = (ComplexMatrixRowFlip, ComplexMatrixColFlip)[by.value]
        return ComplexMatrixTransform(self)

    def reverse(self) -> ComplexMatrixLike[M_co, N_co]:
        raise NotImplementedError
        from .views.builtins import ComplexMatrixReverse
        return ComplexMatrixReverse(self)


class RealMatrixOperatorsMixin(Generic[M_co, N_co]):
    """Mixin class that defines the "operator methods" of ``RealMatrixLike``
    using built-in matrix types
    """

    __slots__ = ()

    @overload
    def __add__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args   = iter(other)
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__add__, self, args),
            shape=shape,
        )

    @overload
    def __sub__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args   = iter(other)
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__sub__, self, args),
            shape=shape,
        )

    @overload
    def __mul__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args   = iter(other)
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__mul__, self, args),
            shape=shape,
        )

    @overload
    def __matmul__(self: ShapedIndexable[float, M_co, N_co], other: IntegralMatrixLike[N_co, P_co]) -> RealMatrixLike[M_co, P_co]: ...
    @overload
    def __matmul__(self: ShapedIndexable[float, M_co, N_co], other: RealMatrixLike[N_co, P_co]) -> RealMatrixLike[M_co, P_co]: ...

    def __matmul__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return RealMatrix(MatrixProduct(self, other))
        return NotImplemented

    @overload
    def __truediv__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __truediv__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args   = iter(other)
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__truediv__, self, args),
            shape=shape,
        )

    @overload
    def __floordiv__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: ShapedIterable[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: ShapedIterable[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __floordiv__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        elif isinstance(other, (int, float)):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return RealMatrix(
            array=map(operator.__floordiv__, self, args),
            shape=shape,
        )

    @overload
    def __mod__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self: ShapedIterable[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self: ShapedIterable[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __mod__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        elif isinstance(other, (int, float)):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return RealMatrix(
            array=map(operator.__mod__, self, args),
            shape=shape,
        )

    @overload
    def __divmod__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...
    @overload
    def __divmod__(self: ShapedIterable[float, M_co, N_co], other: int) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...
    @overload
    def __divmod__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...
    @overload
    def __divmod__(self: ShapedIterable[float, M_co, N_co], other: float) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...

    def __divmod__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        elif isinstance(other, (int, float)):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        args_x, args_y = itertools.tee(map(divmod, self, args))
        shape = self.shape
        return (
            RealMatrix(
                array=map(operator.itemgetter(0), args_x),
                shape=shape,
            ),
            RealMatrix(
                array=map(operator.itemgetter(1), args_y),
                shape=shape,
            ),
        )

    @overload
    def __radd__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: ShapedIterable[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: ShapedIterable[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: ShapedIterable[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __radd__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args   = iter(other)
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__add__, args, self),
            shape=shape,
        )

    @overload
    def __rsub__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: ShapedIterable[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: ShapedIterable[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: ShapedIterable[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rsub__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args   = iter(other)
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__sub__, args, self),
            shape=shape,
        )

    @overload
    def __rmul__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: ShapedIterable[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: ShapedIterable[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: ShapedIterable[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rmul__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args   = iter(other)
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__mul__, args, self),
            shape=shape,
        )

    def __rmatmul__(self: ShapedIndexable[float, M_co, N_co], other: IntegralMatrixLike[P_co, M_co]) -> RealMatrixLike[P_co, N_co]:  # type: ignore[misc]
        if isinstance(other, IntegralMatrixLike):
            return RealMatrix(MatrixProduct(other, self))
        return NotImplemented

    @overload
    def __rtruediv__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: ShapedIterable[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: ShapedIterable[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: ShapedIterable[float, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rtruediv__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args   = iter(other)
            Matrix = RealMatrix
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__truediv__, args, self),
            shape=shape,
        )

    @overload
    def __rfloordiv__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rfloordiv__(self: ShapedIterable[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rfloordiv__(self: ShapedIterable[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __rfloordiv__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args = iter(other)
        elif isinstance(other, (int, float)):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return RealMatrix(
            array=map(operator.__floordiv__, args, self),
            shape=shape,
        )

    @overload
    def __rmod__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmod__(self: ShapedIterable[float, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmod__(self: ShapedIterable[float, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __rmod__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args = iter(other)
        elif isinstance(other, (int, float)):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return RealMatrix(
            array=map(operator.__mod__, args, self),
            shape=shape,
        )

    @overload
    def __rdivmod__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...
    @overload
    def __rdivmod__(self: ShapedIterable[float, M_co, N_co], other: int) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...
    @overload
    def __rdivmod__(self: ShapedIterable[float, M_co, N_co], other: float) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...

    def __rdivmod__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args = iter(other)
        elif isinstance(other, (int, float)):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        args_x, args_y = itertools.tee(map(divmod, args, self))
        shape = self.shape
        return (
            RealMatrix(
                array=map(operator.itemgetter(0), args_x),
                shape=shape,
            ),
            RealMatrix(
                array=map(operator.itemgetter(1), args_y),
                shape=shape,
            ),
        )

    def __neg__(self: ShapedIterable[float, M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        shape = self.shape
        return RealMatrix(
            array=map(operator.__neg__, self),  # type: ignore[arg-type]
            shape=shape,
        )

    def __abs__(self: ShapedIterable[float, M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        shape = self.shape
        return RealMatrix(
            array=map(abs, self),  # type: ignore[arg-type]
            shape=shape,
        )

    @overload
    def lesser(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: ShapedIterable[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: ShapedIterable[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__lt__, self, args),
            shape=shape,
        )

    @overload
    def lesser_equal(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: ShapedIterable[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: ShapedIterable[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__le__, self, args),
            shape=shape,
        )

    @overload
    def greater(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: ShapedIterable[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: ShapedIterable[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__gt__, self, args),
            shape=shape,
        )

    @overload
    def greater_equal(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: ShapedIterable[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: ShapedIterable[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__ge__, self, args),
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
        from .views.builtins import RealMatrixTranspose
        return RealMatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[M_co, N_co]:
        raise NotImplementedError
        from .views.builtins import RealMatrixColFlip, RealMatrixRowFlip
        RealMatrixTransform = (RealMatrixRowFlip, RealMatrixColFlip)[by.value]
        return RealMatrixTransform(self)

    def reverse(self) -> RealMatrixLike[M_co, N_co]:
        raise NotImplementedError
        from .views.builtins import RealMatrixReverse
        return RealMatrixReverse(self)


class IntegralMatrixOperatorsMixin(Generic[M_co, N_co]):
    """Mixin class that defines the "operator methods" of
    ``IntegralMatrixLike`` using built-in matrix types
    """

    __slots__ = ()

    @overload
    def __add__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args   = iter(other)
            Matrix = IntegralMatrix
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
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
            array=map(operator.__add__, self, args),
            shape=shape,
        )

    @overload
    def __sub__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args   = iter(other)
            Matrix = IntegralMatrix
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
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
            array=map(operator.__sub__, self, args),
            shape=shape,
        )

    @overload
    def __mul__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args   = iter(other)
            Matrix = IntegralMatrix
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
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
            array=map(operator.__mul__, self, args),
            shape=shape,
        )

    def __matmul__(self: ShapedIndexable[int, M_co, N_co], other: IntegralMatrixLike[N_co, P_co]) -> IntegralMatrixLike[M_co, P_co]:
        if isinstance(other, IntegralMatrixLike):
            return IntegralMatrix(MatrixProduct(self, other))
        return NotImplemented

    @overload
    def __truediv__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[int, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __truediv__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args   = iter(other)
            Matrix = IntegralMatrix
        elif isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__truediv__, self, args),
            shape=shape,
        )

    @overload
    def __floordiv__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: ShapedIterable[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __floordiv__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args   = iter(other)
            Matrix = IntegralMatrix
        elif isinstance(other, (int, float)):
            args = itertools.repeat(other)
            if isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__floordiv__, self, args),
            shape=shape,
        )

    @overload
    def __mod__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self: ShapedIterable[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __mod__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args   = iter(other)
            Matrix = IntegralMatrix
        elif isinstance(other, (int, float)):
            args = itertools.repeat(other)
            if isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__mod__, self, args),
            shape=shape,
        )

    @overload
    def __divmod__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> tuple[IntegralMatrixLike[M_co, N_co], IntegralMatrixLike[M_co, N_co]]: ...
    @overload
    def __divmod__(self: ShapedIterable[int, M_co, N_co], other: int) -> tuple[IntegralMatrixLike[M_co, N_co], IntegralMatrixLike[M_co, N_co]]: ...
    @overload
    def __divmod__(self: ShapedIterable[int, M_co, N_co], other: float) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]: ...

    def __divmod__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args   = iter(other)
            Matrix = IntegralMatrix
        elif isinstance(other, (int, float)):
            args = itertools.repeat(other)
            if isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        args_x, args_y = itertools.tee(map(divmod, self, args))
        shape = self.shape
        return (
            Matrix(
                array=map(operator.itemgetter(0), args_x),
                shape=shape,
            ),
            Matrix(
                array=map(operator.itemgetter(1), args_y),
                shape=shape,
            ),
        )

    @overload
    def __lshift__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __lshift__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...

    def __lshift__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args = iter(other)
        elif isinstance(other, int):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__lshift__, self, args),
            shape=shape,
        )

    @overload
    def __rshift__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __rshift__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...

    def __rshift__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args = iter(other)
        elif isinstance(other, int):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__rshift__, self, args),
            shape=shape,
        )

    @overload
    def __and__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __and__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...

    def __and__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args = iter(other)
        elif isinstance(other, int):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__and__, self, args),
            shape=shape,
        )

    @overload
    def __xor__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __xor__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...

    def __xor__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args = iter(other)
        elif isinstance(other, int):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__xor__, self, args),
            shape=shape,
        )

    @overload
    def __or__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __or__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...

    def __or__(self, other):
        if isinstance(other, IntegralMatrixLike):
            args = iter(other)
        elif isinstance(other, int):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__or__, self, args),
            shape=shape,
        )

    @overload
    def __radd__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: ShapedIterable[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: ShapedIterable[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __radd__(self, other):
        if isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
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
            array=map(operator.__add__, args, self),
            shape=shape,
        )

    @overload
    def __rsub__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: ShapedIterable[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: ShapedIterable[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rsub__(self, other):
        if isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
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
            array=map(operator.__sub__, args, self),
            shape=shape,
        )

    @overload
    def __rmul__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: ShapedIterable[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: ShapedIterable[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
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
            array=map(operator.__mul__, args, self),
            shape=shape,
        )

    @overload
    def __rtruediv__(self: ShapedIterable[int, M_co, N_co], other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: ShapedIterable[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: ShapedIterable[int, M_co, N_co], other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, complex)):
            args = itertools.repeat(other)
            if isinstance(other, complex):
                Matrix = ComplexMatrix
            else:
                Matrix = RealMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__truediv__, args, self),
            shape=shape,
        )

    @overload
    def __rfloordiv__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __rfloordiv__(self: ShapedIterable[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __rfloordiv__(self, other):
        if isinstance(other, (int, float)):
            args = itertools.repeat(other)
            if isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__floordiv__, args, self),
            shape=shape,
        )

    @overload
    def __rmod__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __rmod__(self: ShapedIterable[int, M_co, N_co], other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __rmod__(self, other):
        if isinstance(other, (int, float)):
            args = itertools.repeat(other)
            if isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__mod__, args, self),
            shape=shape,
        )

    @overload
    def __rdivmod__(self: ShapedIterable[int, M_co, N_co], other: int) -> tuple[IntegralMatrixLike[M_co, N_co], IntegralMatrixLike[M_co, N_co]]: ...
    @overload
    def __rdivmod__(self: ShapedIterable[int, M_co, N_co], other: float) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]: ...

    def __rdivmod__(self, other):
        if isinstance(other, (int, float)):
            args = itertools.repeat(other)
            if isinstance(other, float):
                Matrix = RealMatrix
            else:
                Matrix = IntegralMatrix
        else:
            return NotImplemented
        args_x, args_y = itertools.tee(map(divmod, args, self))
        shape = self.shape
        return (
            Matrix(
                array=map(operator.itemgetter(0), args_x),
                shape=shape,
            ),
            Matrix(
                array=map(operator.itemgetter(1), args_y),
                shape=shape,
            ),
        )

    def __rlshift__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]:
        if isinstance(other, int):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__lshift__, args, self),
            shape=shape,
        )

    def __rrshift__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]:
        if isinstance(other, int):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__rshift__, args, self),
            shape=shape,
        )

    def __rand__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]:
        if isinstance(other, int):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__and__, args, self),
            shape=shape,
        )

    def __rxor__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]:
        if isinstance(other, int):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__xor__, args, self),
            shape=shape,
        )

    def __ror__(self: ShapedIterable[int, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]:
        if isinstance(other, int):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__or__, args, self),
            shape=shape,
        )

    def __neg__(self: ShapedIterable[int, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__neg__, self),  # type: ignore[arg-type]
            shape=shape,
        )

    def __abs__(self: ShapedIterable[int, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        shape = self.shape
        return IntegralMatrix(
            array=map(abs, self),  # type: ignore[arg-type]
            shape=shape,
        )

    def __invert__(self: ShapedIterable[int, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__invert__, self),  # type: ignore[arg-type]
            shape=shape,
        )

    @overload
    def lesser(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: ShapedIterable[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: ShapedIterable[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__lt__, self, args),
            shape=shape,
        )

    @overload
    def lesser_equal(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: ShapedIterable[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: ShapedIterable[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__le__, self, args),
            shape=shape,
        )

    @overload
    def greater(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: ShapedIterable[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: ShapedIterable[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__gt__, self, args),
            shape=shape,
        )

    @overload
    def greater_equal(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: ShapedIterable[float, M_co, N_co], other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: ShapedIterable[float, M_co, N_co], other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__ge__, self, args),
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
        from .views.builtins import IntegralMatrixTranspose
        return IntegralMatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> IntegralMatrixLike[M_co, N_co]:
        raise NotImplementedError
        from .views.builtins import (IntegralMatrixColFlip,
                                     IntegralMatrixRowFlip)
        IntegralMatrixTransform = (IntegralMatrixRowFlip, IntegralMatrixColFlip)[by.value]
        return IntegralMatrixTransform(self)

    def reverse(self) -> IntegralMatrixLike[M_co, N_co]:
        raise NotImplementedError
        from .views.builtins import IntegralMatrixReverse
        return IntegralMatrixReverse(self)


class TimedeltaMatrixOperatorsMixin(Generic[M_co, N_co]):

    __slots__ = ()

    @overload
    def __add__(self: ShapedIterable[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[timedelta, M_co, N_co], other: timedelta) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        if isinstance(other, TimedeltaMatrixLike):
            args = iter(other)
        elif isinstance(other, timedelta):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return TimedeltaMatrix(
            array=map(operator.__add__, self, args),
            shape=shape,
        )

    @overload
    def __sub__(self: ShapedIterable[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[timedelta, M_co, N_co], other: timedelta) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        if isinstance(other, TimedeltaMatrixLike):
            args = iter(other)
        elif isinstance(other, timedelta):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return TimedeltaMatrix(
            array=map(operator.__sub__, self, args),
            shape=shape,
        )

    @overload
    def __mul__(self: ShapedIterable[timedelta, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[timedelta, M_co, N_co], other: int) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[timedelta, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[timedelta, M_co, N_co], other: float) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            args = iter(other)
        elif isinstance(other, (int, float)):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return TimedeltaMatrix(
            array=map(operator.__mul__, self, args),
            shape=shape,
        )

    @overload
    def __truediv__(self: ShapedIterable[timedelta, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[timedelta, M_co, N_co], other: int) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[timedelta, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[timedelta, M_co, N_co], other: float) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[timedelta, M_co, N_co], other: timedelta) -> RealMatrixLike[M_co, N_co]: ...

    def __truediv__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, TimedeltaMatrixLike)):
            args = iter(other)
            if isinstance(other, TimedeltaMatrixLike):
                Matrix = RealMatrix
            else:
                Matrix = TimedeltaMatrix
        elif isinstance(other, (int, float, timedelta)):
            args = itertools.repeat(other)
            if isinstance(other, timedelta):
                Matrix = RealMatrix
            else:
                Matrix = TimedeltaMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__truediv__, self, args),
            shape=shape,
        )

    @overload
    def __floordiv__(self: ShapedIterable[timedelta, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: ShapedIterable[timedelta, M_co, N_co], other: int) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: ShapedIterable[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: ShapedIterable[timedelta, M_co, N_co], other: timedelta) -> IntegralMatrixLike[M_co, N_co]: ...

    def __floordiv__(self, other):
        if isinstance(other, (IntegralMatrixLike, TimedeltaMatrixLike)):
            args = iter(other)
            if isinstance(other, TimedeltaMatrixLike):
                Matrix = IntegralMatrix
            else:
                Matrix = TimedeltaMatrix
        elif isinstance(other, (int, timedelta)):
            args = itertools.repeat(other)
            if isinstance(other, timedelta):
                Matrix = IntegralMatrix
            else:
                Matrix = TimedeltaMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__floordiv__, self, args),
            shape=shape,
        )

    @overload
    def __mod__(self: ShapedIterable[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self: ShapedIterable[timedelta, M_co, N_co], other: timedelta) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __mod__(self, other):
        if isinstance(other, TimedeltaMatrixLike):
            args = iter(other)
        elif isinstance(other, timedelta):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return TimedeltaMatrix(
            array=map(operator.__mod__, self, args),
            shape=shape,
        )

    @overload
    def __divmod__(self: ShapedIterable[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> tuple[IntegralMatrixLike[M_co, N_co], TimedeltaMatrixLike[M_co, N_co]]: ...
    @overload
    def __divmod__(self: ShapedIterable[timedelta, M_co, N_co], other: timedelta) -> tuple[IntegralMatrixLike[M_co, N_co], TimedeltaMatrixLike[M_co, N_co]]: ...

    def __divmod__(self, other):
        if isinstance(other, TimedeltaMatrixLike):
            args = iter(other)
        elif isinstance(other, timedelta):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        args_x, args_y = itertools.tee(map(divmod, self, args))
        shape = self.shape
        return (
            IntegralMatrix(
                array=map(operator.itemgetter(0), args_x),
                shape=shape,
            ),
            TimedeltaMatrix(
                array=map(operator.itemgetter(1), args_y),
                shape=shape,
            ),
        )

    def __radd__(self: ShapedIterable[timedelta, M_co, N_co], other: timedelta) -> TimedeltaMatrixLike[M_co, N_co]:
        if isinstance(other, timedelta):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return TimedeltaMatrix(
            array=map(operator.__add__, args, self),
            shape=shape,
        )

    def __rsub__(self: ShapedIterable[timedelta, M_co, N_co], other: timedelta) -> TimedeltaMatrixLike[M_co, N_co]:
        if isinstance(other, timedelta):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return TimedeltaMatrix(
            array=map(operator.__sub__, args, self),
            shape=shape,
        )

    def __neg__(self: ShapedIterable[timedelta, M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]:
        shape = self.shape
        return TimedeltaMatrix(
            array=map(operator.__neg__, self),  # type: ignore[arg-type]
            shape=shape,
        )

    def __abs__(self: ShapedIterable[timedelta, M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]:
        shape = self.shape
        return TimedeltaMatrix(
            array=map(abs, self),  # type: ignore[arg-type]
            shape=shape,
        )

    @overload
    def lesser(self: ShapedIterable[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: ShapedIterable[timedelta, M_co, N_co], other: timedelta) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        if isinstance(other, TimedeltaMatrixLike):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__lt__, self, args),
            shape=shape,
        )

    @overload
    def lesser_equal(self: ShapedIterable[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: ShapedIterable[timedelta, M_co, N_co], other: timedelta) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        if isinstance(other, TimedeltaMatrixLike):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__le__, self, args),
            shape=shape,
        )

    @overload
    def greater(self: ShapedIterable[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: ShapedIterable[timedelta, M_co, N_co], other: timedelta) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        if isinstance(other, TimedeltaMatrixLike):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__gt__, self, args),
            shape=shape,
        )

    @overload
    def greater_equal(self: ShapedIterable[timedelta, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: ShapedIterable[timedelta, M_co, N_co], other: timedelta) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        if isinstance(other, TimedeltaMatrixLike):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__ge__, self, args),
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
    def __add__(self: ShapedIterable[datetime, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[datetime, M_co, N_co], other: timedelta) -> DatetimeMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        if isinstance(other, TimedeltaMatrixLike):
            args = iter(other)
        elif isinstance(other, timedelta):
            args = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        return DatetimeMatrix(
            array=map(operator.__add__, self, args),
            shape=shape,
        )

    @overload
    def __sub__(self: ShapedIterable[datetime, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[datetime, M_co, N_co], other: timedelta) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[datetime, M_co, N_co], other: DatetimeMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[datetime, M_co, N_co], other: datetime) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        if isinstance(other, (TimedeltaMatrixLike, DatetimeMatrixLike)):
            args = iter(other)
            if isinstance(other, DatetimeMatrixLike):
                Matrix = TimedeltaMatrix
            else:
                Matrix = DatetimeMatrix
        elif isinstance(other, (timedelta, datetime)):
            args = itertools.repeat(other)
            if isinstance(other, datetime):
                Matrix = TimedeltaMatrix
            else:
                Matrix = DatetimeMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__sub__, self, args),
            shape=shape,
        )

    @overload
    def __rsub__(self: ShapedIterable[datetime, M_co, N_co], other: TimedeltaMatrixLike[M_co, N_co]) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: ShapedIterable[datetime, M_co, N_co], other: timedelta) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: ShapedIterable[datetime, M_co, N_co], other: datetime) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __rsub__(self, other):
        if isinstance(other, TimedeltaMatrixLike):
            args   = iter(other)
            Matrix = DatetimeMatrix
        elif isinstance(other, (timedelta, datetime)):
            args = itertools.repeat(other)
            if isinstance(other, datetime):
                Matrix = TimedeltaMatrix
            else:
                Matrix = DatetimeMatrix
        else:
            return NotImplemented
        shape = self.shape
        return Matrix(
            array=map(operator.__sub__, args, self),
            shape=shape,
        )

    @overload
    def lesser(self: ShapedIterable[datetime, M_co, N_co], other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: ShapedIterable[datetime, M_co, N_co], other: datetime) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        if isinstance(other, DatetimeMatrixLike):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__lt__, self, args),
            shape=shape,
        )

    @overload
    def lesser_equal(self: ShapedIterable[datetime, M_co, N_co], other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: ShapedIterable[datetime, M_co, N_co], other: datetime) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        if isinstance(other, DatetimeMatrixLike):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__le__, self, args),
            shape=shape,
        )

    @overload
    def greater(self: ShapedIterable[datetime, M_co, N_co], other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: ShapedIterable[datetime, M_co, N_co], other: datetime) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        if isinstance(other, DatetimeMatrixLike):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__gt__, self, args),
            shape=shape,
        )

    @overload
    def greater_equal(self: ShapedIterable[datetime, M_co, N_co], other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: ShapedIterable[datetime, M_co, N_co], other: datetime) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        if isinstance(other, DatetimeMatrixLike):
            args = iter(other)
        else:
            args = itertools.repeat(other)
        shape = self.shape
        return IntegralMatrix(
            array=map(operator.__ge__, self, args),
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
