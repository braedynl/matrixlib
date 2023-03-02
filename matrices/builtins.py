from __future__ import annotations

import copy
import itertools
import operator
from collections.abc import Iterable, Iterator
from typing import (Any, Generic, Literal, Optional, SupportsIndex, TypeVar,
                    cast, overload)

from .abc import (ComplexMatrixLike, IntegralMatrixLike, MatrixLike,
                  RealMatrixLike, ShapedIndexable, ShapedIterable,
                  StringMatrixLike)
from .rule import COL, ROW, Rule
from .utilities.matrix_map import MatrixMap
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
    "ConstantMatrix",
    "ComplexConstantMatrix",
    "RealConstantMatrix",
    "IntegralConstantMatrix",
]

T = TypeVar("T")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)
P_co = TypeVar("P_co", covariant=True, bound=int)

MatrixT = TypeVar("MatrixT", bound="Matrix")
ConstantMatrixT = TypeVar("ConstantMatrixT", bound="ConstantMatrix")


class MatrixOperatorsMixin(Generic[T_co, M_co, N_co]):
    """Mixin class that defines the "operator methods" of ``MatrixLike`` using
    built-in matrix types
    """

    __slots__ = ()

    def equal(self: ShapedIterable[T_co, M_co, N_co], other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__eq__, self, other))

    def not_equal(self: ShapedIterable[T_co, M_co, N_co], other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__ne__, self, other))


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
        return f"{self.__class__.__name__}(array={self._array!r}, shape={self._shape!r})"

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
        array = self._array

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

    @classmethod
    def from_nesting(cls: type[MatrixT], nesting: Iterable[Iterable[T_co]]) -> MatrixT:
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
        from .views.builtins import MatrixTranspose
        return MatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T_co, M_co, N_co]:
        from .views.builtins import MatrixColFlip, MatrixRowFlip
        MatrixTransform = (MatrixRowFlip, MatrixColFlip)[by.value]
        return MatrixTransform(self)

    def reverse(self) -> MatrixLike[T_co, M_co, N_co]:
        from .views.builtins import MatrixReverse
        return MatrixReverse(self)


class StringMatrixOperatorsMixin(Generic[M_co, N_co]):

    __slots__ = ()

    def __add__(self: ShapedIterable[str, M_co, N_co], other: StringMatrixLike[M_co, N_co]) -> StringMatrixLike[M_co, N_co]:
        if isinstance(other, StringMatrixLike):
            return StringMatrix(MatrixMap(operator.__add__, self, other))
        return NotImplemented

    def __mul__(self: ShapedIterable[str, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> StringMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return StringMatrix(MatrixMap(operator.__mul__, self, other))
        return NotImplemented

    def __rmul__(self: ShapedIterable[str, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> StringMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return StringMatrix(MatrixMap(operator.__mul__, other, self))
        return NotImplemented

    def lesser(self: ShapedIterable[str, M_co, N_co], other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__lt__, self, other))

    def lesser_equal(self: ShapedIterable[str, M_co, N_co], other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__le__, self, other))

    def greater(self: ShapedIterable[str, M_co, N_co], other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__gt__, self, other))

    def greater_equal(self: ShapedIterable[str, M_co, N_co], other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__ge__, self, other))


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
    def __add__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, ComplexMatrixLike)):
            return ComplexMatrix(MatrixMap(operator.__add__, self, other))
        return NotImplemented

    @overload
    def __sub__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, ComplexMatrixLike)):
            return ComplexMatrix(MatrixMap(operator.__sub__, self, other))
        return NotImplemented

    @overload
    def __mul__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, ComplexMatrixLike)):
            return ComplexMatrix(MatrixMap(operator.__mul__, self, other))
        return NotImplemented

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
    def __truediv__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    def __truediv__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike, ComplexMatrixLike)):
            return ComplexMatrix(MatrixMap(operator.__truediv__, self, other))
        return NotImplemented

    @overload
    def __radd__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    def __radd__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return ComplexMatrix(MatrixMap(operator.__add__, other, self))
        return NotImplemented

    @overload
    def __rsub__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rsub__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return ComplexMatrix(MatrixMap(operator.__sub__, other, self))
        return NotImplemented

    @overload
    def __rmul__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rmul__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return ComplexMatrix(MatrixMap(operator.__mul__, other, self))
        return NotImplemented

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
    def __rtruediv__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rtruediv__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return ComplexMatrix(MatrixMap(operator.__truediv__, other, self))
        return NotImplemented

    def __neg__(self: ShapedIterable[complex, M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]:
        return ComplexMatrix(MatrixMap(operator.__neg__, self))

    def __abs__(self: ShapedIterable[complex, M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        return RealMatrix(MatrixMap(abs, self))

    def conjugate(self: ShapedIterable[complex, M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]:
        return ComplexMatrix(MatrixMap(lambda x: x.conjugate(), self))


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
        from .views.builtins import ComplexMatrixTranspose
        return ComplexMatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[M_co, N_co]:
        from .views.builtins import ComplexMatrixColFlip, ComplexMatrixRowFlip
        ComplexMatrixTransform = (ComplexMatrixRowFlip, ComplexMatrixColFlip)[by.value]
        return ComplexMatrixTransform(self)

    def reverse(self) -> ComplexMatrixLike[M_co, N_co]:
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
    def __add__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return RealMatrix(MatrixMap(operator.__add__, self, other))
        return NotImplemented

    @overload
    def __sub__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return RealMatrix(MatrixMap(operator.__sub__, self, other))
        return NotImplemented

    @overload
    def __mul__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return RealMatrix(MatrixMap(operator.__mul__, self, other))
        return NotImplemented

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
    def __truediv__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    def __truediv__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return RealMatrix(MatrixMap(operator.__truediv__, self, other))
        return NotImplemented

    @overload
    def __floordiv__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    def __floordiv__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return RealMatrix(MatrixMap(operator.__floordiv__, self, other))
        return NotImplemented

    @overload
    def __mod__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    def __mod__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            return RealMatrix(MatrixMap(operator.__mod__, self, other))
        return NotImplemented

    @overload
    def __divmod__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...
    @overload
    def __divmod__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...

    def __divmod__(self, other):
        if isinstance(other, (IntegralMatrixLike, RealMatrixLike)):
            xs, ys = itertools.tee(MatrixMap(divmod, self, other))
            shape = self.shape
            return (
                RealMatrix(
                    array=map(operator.itemgetter(0), xs),
                    shape=shape,
                ),
                RealMatrix(
                    array=map(operator.itemgetter(1), ys),
                    shape=shape,
                ),
            )
        return NotImplemented

    def __radd__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return RealMatrix(MatrixMap(operator.__add__, other, self))
        return NotImplemented

    def __rsub__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return RealMatrix(MatrixMap(operator.__sub__, other, self))
        return NotImplemented

    def __rmul__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return RealMatrix(MatrixMap(operator.__mul__, other, self))
        return NotImplemented

    def __rmatmul__(self: ShapedIndexable[float, M_co, N_co], other: IntegralMatrixLike[P_co, M_co]) -> RealMatrixLike[P_co, N_co]:  # type: ignore[misc]
        if isinstance(other, IntegralMatrixLike):
            return RealMatrix(MatrixProduct(other, self))
        return NotImplemented

    def __rtruediv__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return RealMatrix(MatrixMap(operator.__truediv__, other, self))
        return NotImplemented

    def __rfloordiv__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return RealMatrix(MatrixMap(operator.__floordiv__, other, self))
        return NotImplemented

    def __rmod__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return RealMatrix(MatrixMap(operator.__mod__, other, self))
        return NotImplemented

    def __rdivmod__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]:
        if isinstance(other, IntegralMatrixLike):
            xs, ys = itertools.tee(MatrixMap(divmod, other, self))
            shape = self.shape
            return (
                RealMatrix(
                    array=map(operator.itemgetter(0), xs),
                    shape=shape,
                ),
                RealMatrix(
                    array=map(operator.itemgetter(1), ys),
                    shape=shape,
                ),
            )
        return NotImplemented

    def __neg__(self: ShapedIterable[float, M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        return RealMatrix(MatrixMap(operator.__neg__, self))

    def __abs__(self: ShapedIterable[float, M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        return RealMatrix(MatrixMap(abs, self))

    @overload
    def lesser(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        return IntegralMatrix(MatrixMap(operator.__lt__, self, other))

    @overload
    def lesser_equal(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        return IntegralMatrix(MatrixMap(operator.__le__, self, other))

    @overload
    def greater(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        return IntegralMatrix(MatrixMap(operator.__gt__, self, other))

    @overload
    def greater_equal(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        return IntegralMatrix(MatrixMap(operator.__ge__, self, other))


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
        from .views.builtins import RealMatrixTranspose
        return RealMatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[M_co, N_co]:
        from .views.builtins import RealMatrixColFlip, RealMatrixRowFlip
        RealMatrixTransform = (RealMatrixRowFlip, RealMatrixColFlip)[by.value]
        return RealMatrixTransform(self)

    def reverse(self) -> RealMatrixLike[M_co, N_co]:
        from .views.builtins import RealMatrixReverse
        return RealMatrixReverse(self)


class IntegralMatrixOperatorsMixin(Generic[M_co, N_co]):
    """Mixin class that defines the "operator methods" of
    ``IntegralMatrixLike`` using built-in matrix types
    """

    __slots__ = ()

    def __add__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return IntegralMatrix(MatrixMap(operator.__add__, self, other))
        return NotImplemented

    def __sub__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return IntegralMatrix(MatrixMap(operator.__sub__, self, other))
        return NotImplemented

    def __mul__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return IntegralMatrix(MatrixMap(operator.__mul__, self, other))
        return NotImplemented

    def __matmul__(self: ShapedIndexable[int, M_co, N_co], other: IntegralMatrixLike[N_co, P_co]) -> IntegralMatrixLike[M_co, P_co]:
        if isinstance(other, IntegralMatrixLike):
            return IntegralMatrix(MatrixProduct(self, other))
        return NotImplemented

    def __truediv__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return RealMatrix(MatrixMap(operator.__truediv__, self, other))
        return NotImplemented

    def __floordiv__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return IntegralMatrix(MatrixMap(operator.__floordiv__, self, other))
        return NotImplemented

    def __mod__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return IntegralMatrix(MatrixMap(operator.__mod__, self, other))
        return NotImplemented

    def __divmod__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> tuple[IntegralMatrixLike[M_co, N_co], IntegralMatrixLike[M_co, N_co]]:
        if isinstance(other, IntegralMatrixLike):
            xs, ys = itertools.tee(MatrixMap(divmod, self, other))
            shape = self.shape
            return (
                IntegralMatrix(
                    array=map(operator.itemgetter(0), xs),
                    shape=shape,
                ),
                IntegralMatrix(
                    array=map(operator.itemgetter(1), ys),
                    shape=shape,
                ),
            )
        return NotImplemented

    def __lshift__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return IntegralMatrix(MatrixMap(operator.__lshift__, self, other))
        return NotImplemented

    def __rshift__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return IntegralMatrix(MatrixMap(operator.__rshift__, self, other))
        return NotImplemented

    def __and__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return IntegralMatrix(MatrixMap(operator.__and__, self, other))
        return NotImplemented

    def __xor__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return IntegralMatrix(MatrixMap(operator.__xor__, self, other))
        return NotImplemented

    def __or__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        if isinstance(other, IntegralMatrixLike):
            return IntegralMatrix(MatrixMap(operator.__or__, self, other))
        return NotImplemented

    def __neg__(self: ShapedIterable[int, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__neg__, self))

    def __abs__(self: ShapedIterable[int, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(abs, self))

    def __invert__(self: ShapedIterable[int, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__invert__, self))

    def lesser(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__lt__, self, other))

    def lesser_equal(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__le__, self, other))

    def greater(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__gt__, self, other))

    def greater_equal(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__ge__, self, other))


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
        from .views.builtins import IntegralMatrixTranspose
        return IntegralMatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> IntegralMatrixLike[M_co, N_co]:
        from .views.builtins import (IntegralMatrixColFlip,
                                     IntegralMatrixRowFlip)
        IntegralMatrixTransform = (IntegralMatrixRowFlip, IntegralMatrixColFlip)[by.value]
        return IntegralMatrixTransform(self)

    def reverse(self) -> IntegralMatrixLike[M_co, N_co]:
        from .views.builtins import IntegralMatrixReverse
        return IntegralMatrixReverse(self)


class ConstantMatrix(MatrixOperatorsMixin[T_co, M_co, N_co], MatrixLike[T_co, M_co, N_co]):

    __slots__ = ("_value", "_shape")

    @overload
    def __init__(self, value: ConstantMatrix[T_co, M_co, N_co]) -> None: ...
    @overload
    def __init__(self, value: T_co, shape: tuple[M_co, N_co]) -> None: ...

    def __init__(self, value, shape=None):
        if isinstance(value, ConstantMatrix):
            self._value = value.value
            self._shape = value.shape
            return

        self._value = value

        nrows = shape[0]
        ncols = shape[1]

        if nrows < 1 or ncols < 1:
            raise ValueError("dimensions must be positive")

        self._shape = shape

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(value={self._value!r}, shape={self._shape!r})"

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
        value = self._value

        if isinstance(key, tuple):
            row_key, col_key = key

            if isinstance(row_key, slice):
                row_indices = self._resolve_matrix_slice(row_key, by=ROW)

                if isinstance(col_key, slice):
                    col_indices = self._resolve_matrix_slice(col_key, by=COL)
                    return self.__class__(
                        value=value,
                        shape=(len(row_indices), len(col_indices)),
                    )

                self._resolve_matrix_index(col_key, by=COL)
                return self.__class__(
                    value=value,
                    shape=(len(row_indices), 1),
                )

            self._resolve_matrix_index(row_key, by=ROW)

            if isinstance(col_key, slice):
                col_indices = self._resolve_matrix_slice(col_key, by=COL)
                return self.__class__(
                    value=value,
                    shape=(1, len(col_indices)),
                )

            self._resolve_matrix_index(col_key, by=COL)
            return value

        if isinstance(key, slice):
            val_indices = self._resolve_vector_slice(key)
            return self.__class__(
                value=value,
                shape=(1, len(val_indices)),
            )

        self._resolve_vector_index(key)
        return value

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self._shape

    @property
    def value(self) -> T_co:
        return self._value

    def transpose(self) -> MatrixLike[T_co, N_co, M_co]:
        value = self._value
        shape = self._shape
        return self.__class__(
            value=value,
            shape=(shape[1], shape[0]),
        )

    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T_co, M_co, N_co]:
        return copy.copy(self)

    def reverse(self) -> MatrixLike[T_co, M_co, N_co]:
        return copy.copy(self)

    def values(self, *, by: Rule = Rule.ROW, reverse: bool = False) -> Iterator[T_co]:
        yield from itertools.repeat(self._value, times=len(self))


class ComplexConstantMatrix(ComplexMatrixOperatorsMixin[M_co, N_co], ComplexMatrixLike[M_co, N_co], ConstantMatrix[complex, M_co, N_co]):

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
        return ConstantMatrix.__getitem__(self, key)

    def transpose(self) -> ComplexMatrixLike[N_co, M_co]:
        return cast(ComplexMatrixLike[N_co, M_co], ConstantMatrix.transpose(self))

    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[M_co, N_co]:
        return cast(ComplexMatrixLike[M_co, N_co], ConstantMatrix.flip(self, by=by))

    def reverse(self) -> ComplexMatrixLike[M_co, N_co]:
        return cast(ComplexMatrixLike[M_co, N_co], ConstantMatrix.reverse(self))


class RealConstantMatrix(RealMatrixOperatorsMixin[M_co, N_co], RealMatrixLike[M_co, N_co], ConstantMatrix[float, M_co, N_co]):

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
        return ConstantMatrix.__getitem__(self, key)

    def transpose(self) -> RealMatrixLike[N_co, M_co]:
        return cast(RealMatrixLike[N_co, M_co], ConstantMatrix.transpose(self))

    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[M_co, N_co]:
        return cast(RealMatrixLike[M_co, N_co], ConstantMatrix.flip(self, by=by))

    def reverse(self) -> RealMatrixLike[M_co, N_co]:
        return cast(RealMatrixLike[M_co, N_co], ConstantMatrix.reverse(self))


class IntegralConstantMatrix(IntegralMatrixOperatorsMixin[M_co, N_co], IntegralMatrixLike[M_co, N_co], ConstantMatrix[int, M_co, N_co]):

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
        return ConstantMatrix.__getitem__(self, key)

    def transpose(self) -> IntegralMatrixLike[N_co, M_co]:
        return cast(IntegralMatrixLike[N_co, M_co], ConstantMatrix.transpose(self))

    def flip(self, *, by: Rule = Rule.ROW) -> IntegralMatrixLike[M_co, N_co]:
        return cast(IntegralMatrixLike[M_co, N_co], ConstantMatrix.flip(self, by=by))

    def reverse(self) -> IntegralMatrixLike[M_co, N_co]:
        return cast(IntegralMatrixLike[M_co, N_co], ConstantMatrix.reverse(self))
