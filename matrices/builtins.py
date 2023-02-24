from __future__ import annotations

import contextlib
import itertools
import operator
from collections.abc import Generator, Iterable
from typing import (Any, Generic, Literal, Optional, SupportsIndex, TypeVar,
                    overload)

from .abc import (ComplexMatrixLike, IntegralMatrixLike, MatrixLike,
                  RealMatrixLike, ShapedIndexable, ShapedIterable,
                  check_friendly)
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
]

T = TypeVar("T")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)
P_co = TypeVar("P_co", covariant=True, bound=int)

MatrixT = TypeVar("MatrixT", bound="Matrix")


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

    def __len__(self) -> int:
        return len(self._array)

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
                        (
                            array[row_index * ncols + col_index]
                            for row_index in row_indices
                            for col_index in col_indices
                        ),
                        shape=(len(row_indices), len(col_indices)),
                    )

                col_index = self._resolve_matrix_index(col_key, by=COL)
                return self.__class__(
                    (
                        array[row_index * ncols + col_index]
                        for row_index in row_indices
                    ),
                    shape=(len(row_indices), 1),
                )

            row_index = self._resolve_matrix_index(row_key, by=ROW)

            if isinstance(col_key, slice):
                col_indices = self._resolve_matrix_slice(col_key, by=COL)
                return self.__class__(
                    (
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
                (
                    array[val_index]
                    for val_index in val_indices
                ),
                shape=(1, len(val_indices)),
            )

        val_index = self._resolve_vector_index(key)
        return array[val_index]

    def __contains__(self, value: Any) -> bool:
        return value in self._array

    def __deepcopy__(self: MatrixT, memo: Optional[dict[int, Any]] = None) -> MatrixT:
        """Return the matrix"""
        return self

    __copy__ = __deepcopy__

    @classmethod
    def from_raw_parts(cls: type[MatrixT], array: tuple[T_co, ...], shape: tuple[M_co, N_co]) -> MatrixT:
        """Construct a matrix directly from its ``array`` and ``shape`` parts

        This method exists primarily for the benefit of matrix-producing
        functions that have "pre-validated" the array and shape. Should be used
        with caution.

        The following properties are required to construct a valid matrix:
        - ``array`` must be a flattened ``tuple``. That is, the elements of the
          matrix must be contained within the shallowest depth of the ``tuple``
          instance.
        - ``shape`` must be a ``tuple`` of two positive integers, where the
          product of its values equals the length of ``array``.

        Direct references to the given parts are kept. It is up to the caller
        to guarantee that the above criteria is met.
        """
        self = cls.__new__(cls)
        self._array = array
        self._shape = shape
        return self

    @classmethod
    def from_nesting(cls: type[MatrixT], nesting: Iterable[Iterable[T_co]]) -> MatrixT:
        """Construct a matrix from a singly-nested iterable, using the
        shallowest iterable's length to deduce the number of rows, and the
        nested iterables' lengths to deduce the number of columns

        Raises ``ValueError`` if the length of the nested iterables is
        inconsistent.
        """
        nrows = ncols = 0

        def flatten(nesting: Iterable[Iterable[T_co]]) -> Generator[T_co, None, None]:
            nonlocal nrows, ncols

            rows = iter(nesting)
            try:
                row = next(rows)
            except StopIteration:
                return

            nrows += 1
            for val in row:
                ncols += 1
                yield val

            for row in rows:
                nrows += 1
                n = 0
                for val in row:
                    n += 1
                    yield val
                if n != ncols:
                    raise ValueError(f"row {nrows} has length {n}, but precedent rows have length {ncols}")

        with contextlib.closing(flatten(nesting)) as values:
            array = tuple(values)
            shape = nrows, ncols

        return cls.from_raw_parts(
            array=array,
            shape=shape,
        )

    @property
    def array(self) -> tuple[T_co, ...]:
        return self._array

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self._shape

    def transpose(self) -> MatrixLike[T_co, N_co, M_co]:
        raise NotImplementedError

    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T_co, M_co, N_co]:
        raise NotImplementedError

    def reverse(self) -> MatrixLike[T_co, M_co, N_co]:
        raise NotImplementedError


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

    @check_friendly
    def __add__(self, other):
        return ComplexMatrix(MatrixMap(operator.__add__, self, other))

    @overload
    def __sub__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __sub__(self, other):
        return ComplexMatrix(MatrixMap(operator.__sub__, self, other))

    @overload
    def __mul__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __mul__(self, other):
        return ComplexMatrix(MatrixMap(operator.__mul__, self, other))

    @overload
    def __matmul__(self: ShapedIndexable[complex, M_co, N_co], other: IntegralMatrixLike[N_co, P_co]) -> ComplexMatrixLike[M_co, P_co]: ...
    @overload
    def __matmul__(self: ShapedIndexable[complex, M_co, N_co], other: RealMatrixLike[N_co, P_co]) -> ComplexMatrixLike[M_co, P_co]: ...
    @overload
    def __matmul__(self: ShapedIndexable[complex, M_co, N_co], other: ComplexMatrixLike[N_co, P_co]) -> ComplexMatrixLike[M_co, P_co]: ...

    @check_friendly
    def __matmul__(self, other):
        return ComplexMatrix(MatrixProduct(self, other))

    @overload
    def __truediv__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __truediv__(self, other):
        return ComplexMatrix(MatrixMap(operator.__truediv__, self, other))

    @overload
    def __radd__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: ShapedIterable[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __radd__(self, other):
        return ComplexMatrix(MatrixMap(operator.__add__, other, self))

    @overload
    def __rsub__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: ShapedIterable[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __rsub__(self, other):
        return ComplexMatrix(MatrixMap(operator.__sub__, other, self))

    @overload
    def __rmul__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: ShapedIterable[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __rmul__(self, other):
        return ComplexMatrix(MatrixMap(operator.__mul__, other, self))

    @overload
    def __rmatmul__(self: ShapedIndexable[complex, M_co, N_co], other: IntegralMatrixLike[P_co, M_co]) -> ComplexMatrixLike[P_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: ShapedIndexable[complex, M_co, N_co], other: RealMatrixLike[P_co, M_co]) -> ComplexMatrixLike[P_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: ShapedIndexable[complex, M_co, N_co], other: ComplexMatrixLike[P_co, M_co]) -> ComplexMatrixLike[P_co, N_co]: ...

    @check_friendly
    def __rmatmul__(self, other):
        return ComplexMatrix(MatrixProduct(other, self))

    @overload
    def __rtruediv__(self: ShapedIterable[complex, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: ShapedIterable[complex, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: ShapedIterable[complex, M_co, N_co], other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __rtruediv__(self, other):
        return ComplexMatrix(MatrixMap(operator.__truediv__, other, self))

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
    def __add__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __add__(self, other):
        return RealMatrix(MatrixMap(operator.__add__, self, other))

    @overload
    def __sub__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __sub__(self, other):
        return RealMatrix(MatrixMap(operator.__sub__, self, other))

    @overload
    def __mul__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __mul__(self, other):
        return RealMatrix(MatrixMap(operator.__mul__, self, other))

    @overload
    def __matmul__(self: ShapedIndexable[float, M_co, N_co], other: IntegralMatrixLike[N_co, P_co]) -> RealMatrixLike[M_co, P_co]: ...
    @overload
    def __matmul__(self: ShapedIndexable[float, M_co, N_co], other: RealMatrixLike[N_co, P_co]) -> RealMatrixLike[M_co, P_co]: ...

    @check_friendly
    def __matmul__(self, other):
        return RealMatrix(MatrixProduct(self, other))

    @overload
    def __truediv__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __truediv__(self, other):
        return RealMatrix(MatrixMap(operator.__truediv__, self, other))

    @overload
    def __floordiv__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __floordiv__(self, other):
        return RealMatrix(MatrixMap(operator.__floordiv__, self, other))

    @overload
    def __mod__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __mod__(self, other):
        return RealMatrix(MatrixMap(operator.__mod__, self, other))

    @overload
    def __divmod__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...
    @overload
    def __divmod__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...

    @check_friendly
    def __divmod__(self, other):
        xx, yx = itertools.tee(MatrixMap(divmod, self, other), 2)
        shape = self.shape
        return (
            RealMatrix.from_raw_parts(
                array=tuple(map(operator.itemgetter(0), xx)),
                shape=shape,
            ),
            RealMatrix.from_raw_parts(
                array=tuple(map(operator.itemgetter(1), yx)),
                shape=shape,
            ),
        )

    @overload
    def __radd__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __radd__(self, other):
        return RealMatrix(MatrixMap(operator.__add__, other, self))

    @overload
    def __rsub__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __rsub__(self, other):
        return RealMatrix(MatrixMap(operator.__sub__, other, self))

    @overload
    def __rmul__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __rmul__(self, other):
        return RealMatrix(MatrixMap(operator.__mul__, other, self))

    @overload
    def __rmatmul__(self: ShapedIndexable[float, M_co, N_co], other: IntegralMatrixLike[P_co, M_co]) -> RealMatrixLike[P_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: ShapedIndexable[float, M_co, N_co], other: RealMatrixLike[P_co, M_co]) -> RealMatrixLike[P_co, N_co]: ...

    @check_friendly
    def __rmatmul__(self, other):
        return RealMatrix(MatrixProduct(other, self))

    @overload
    def __rtruediv__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __rtruediv__(self, other):
        return RealMatrix(MatrixMap(operator.__truediv__, other, self))

    @overload
    def __rfloordiv__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __rfloordiv__(self, other):
        return RealMatrix(MatrixMap(operator.__floordiv__, other, self))

    @overload
    def __rmod__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...

    @check_friendly
    def __rmod__(self, other):
        return RealMatrix(MatrixMap(operator.__mod__, other, self))

    @overload
    def __rdivmod__(self: ShapedIterable[float, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: ShapedIterable[float, M_co, N_co], other: RealMatrixLike[M_co, N_co]) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...  # type: ignore[misc]

    @check_friendly
    def __rdivmod__(self, other):
        xx, yx = itertools.tee(MatrixMap(divmod, other, self), 2)
        shape = self.shape
        return (
            RealMatrix.from_raw_parts(
                array=tuple(map(operator.itemgetter(0), xx)),
                shape=shape,
            ),
            RealMatrix.from_raw_parts(
                array=tuple(map(operator.itemgetter(1), yx)),
                shape=shape,
            ),
        )

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

    @check_friendly
    def __add__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__add__, self, other))

    @check_friendly
    def __sub__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__sub__, self, other))

    @check_friendly
    def __mul__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__mul__, self, other))

    @check_friendly
    def __matmul__(self: ShapedIndexable[int, M_co, N_co], other: IntegralMatrixLike[N_co, P_co]) -> IntegralMatrixLike[M_co, P_co]:
        return IntegralMatrix(MatrixProduct(self, other))

    @check_friendly
    def __truediv__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        return RealMatrix(MatrixMap(operator.__truediv__, self, other))

    @check_friendly
    def __floordiv__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__floordiv__, self, other))

    @check_friendly
    def __mod__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__mod__, self, other))

    @check_friendly
    def __divmod__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> tuple[IntegralMatrixLike[M_co, N_co], IntegralMatrixLike[M_co, N_co]]:
        xx, yx = itertools.tee(MatrixMap(divmod, self, other), 2)
        shape = self.shape
        return (
            IntegralMatrix.from_raw_parts(
                array=tuple(map(operator.itemgetter(0), xx)),
                shape=shape,
            ),
            IntegralMatrix.from_raw_parts(
                array=tuple(map(operator.itemgetter(1), yx)),
                shape=shape,
            ),
        )

    @check_friendly
    def __lshift__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__lshift__, self, other))

    @check_friendly
    def __rshift__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__rshift__, self, other))

    @check_friendly
    def __and__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__and__, self, other))

    @check_friendly
    def __xor__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__xor__, self, other))

    @check_friendly
    def __or__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__or__, self, other))

    @check_friendly
    def __radd__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__add__, other, self))

    @check_friendly
    def __rsub__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__sub__, other, self))

    @check_friendly
    def __rmul__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__mul__, other, self))

    @check_friendly
    def __rmatmul__(self: ShapedIndexable[int, M_co, N_co], other: IntegralMatrixLike[P_co, M_co]) -> IntegralMatrixLike[P_co, N_co]:
        return IntegralMatrix(MatrixProduct(other, self))

    @check_friendly
    def __rtruediv__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]:
        return RealMatrix(MatrixMap(operator.__truediv__, other, self))

    @check_friendly
    def __rfloordiv__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__floordiv__, other, self))

    @check_friendly
    def __rmod__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__mod__, other, self))

    @check_friendly
    def __rdivmod__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> tuple[IntegralMatrixLike[M_co, N_co], IntegralMatrixLike[M_co, N_co]]:
        xx, yx = itertools.tee(MatrixMap(divmod, other, self), 2)
        shape = self.shape
        return (
            IntegralMatrix.from_raw_parts(
                array=tuple(map(operator.itemgetter(0), xx)),
                shape=shape,
            ),
            IntegralMatrix.from_raw_parts(
                array=tuple(map(operator.itemgetter(1), yx)),
                shape=shape,
            ),
        )

    @check_friendly
    def __rlshift__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__lshift__, other, self))

    @check_friendly
    def __rrshift__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__rshift__, other, self))

    @check_friendly
    def __rand__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__and__, other, self))

    @check_friendly
    def __rxor__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__xor__, other, self))

    @check_friendly
    def __ror__(self: ShapedIterable[int, M_co, N_co], other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]:
        return IntegralMatrix(MatrixMap(operator.__or__, other, self))

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
        raise NotImplementedError

    def flip(self, *, by: Rule = Rule.ROW) -> IntegralMatrixLike[M_co, N_co]:
        raise NotImplementedError

    def reverse(self) -> IntegralMatrixLike[M_co, N_co]:
        raise NotImplementedError
