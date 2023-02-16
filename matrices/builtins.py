from reprlib import recursive_repr
from typing import TypeVar

from views import SequenceView

from .abc import (ComplexMatrixLike, IntegralMatrixLike, MatrixLike,
                  RealMatrixLike, ShapedIterable, check_friendly)
from .rule import COL, ROW, Rule
from .utilities import matrix_operator
from .utilities.matrix_map import MatrixMap

__all__ = [
    "Matrix",
    "ComplexMatrix",
    "RealMatrix",
    "IntegralMatrix",
]

T = TypeVar("T")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)

ComplexT_co = TypeVar("ComplexT_co", covariant=True, bound=complex)
RealT_co = TypeVar("RealT_co", covariant=True, bound=float)
IntegralT_co = TypeVar("IntegralT_co", covariant=True, bound=int)


class Matrix(MatrixLike[T_co, M_co, N_co]):

    __slots__ = ("_array", "_shape")

    def __init__(self, array=None, shape=None):
        self._array = [] if array is None else list(array)

        if isinstance(array, (MatrixLike, MatrixMap, ShapedIterable)):
            self._shape = array.shape
            return

        n = len(self._array)

        if shape is None:
            self._shape = (1, n)
            return

        nrows, ncols = shape

        if nrows is None and ncols is None:
            shape = (1, n)

        elif nrows is None:
            nrows, loss = divmod(n, ncols) if ncols else (0, n)
            if loss:
                raise ValueError(f"cannot interpret array of size {n} as shape M × {ncols}")
            shape = (nrows, ncols)

        elif ncols is None:
            ncols, loss = divmod(n, nrows) if nrows else (0, n)
            if loss:
                raise ValueError(f"cannot interpret array of size {n} as shape {nrows} × N")
            shape = (nrows, ncols)

        else:
            m = nrows * ncols
            if n != m:
                raise ValueError(f"cannot interpret array of size {n} as shape {nrows} × {ncols}")

        if nrows < 0 or ncols < 0:
            raise ValueError("dimensions must be non-negative")

        self._shape = shape

    @recursive_repr("...")
    def __repr__(self):
        """Return a canonical representation of the matrix"""
        return f"{self.__class__.__name__}(array={self._array!r}, shape={self._shape!r})"

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
            ncols = self.ncols

            if isinstance(row_key, slice):
                row_indices = self._resolve_matrix_slice(row_key, by=ROW)

                if isinstance(col_key, slice):
                    col_indices = self._resolve_matrix_slice(col_key, by=COL)
                    return self.__class__.from_raw_parts(
                        array=[
                            self._array[row_index * ncols + col_index]
                            for row_index in row_indices
                            for col_index in col_indices
                        ],
                        shape=(len(row_indices), len(col_indices)),
                    )

                col_index = self._resolve_matrix_index(col_key, by=COL)
                return self.__class__.from_raw_parts(
                    array=[
                        self._array[row_index * ncols + col_index]
                        for row_index in row_indices
                    ],
                    shape=(len(row_indices), 1),
                )

            row_index = self._resolve_matrix_index(row_key, by=ROW)

            if isinstance(col_key, slice):
                col_indices = self._resolve_matrix_slice(col_key, by=COL)
                return self.__class__.from_raw_parts(
                    array=[
                        self._array[row_index * ncols + col_index]
                        for col_index in col_indices
                    ],
                    shape=(1, len(col_indices)),
                )

            col_index = self._resolve_matrix_index(col_key, by=COL)
            return self._array[row_index * ncols + col_index]

        if isinstance(key, slice):
            val_indices = self._resolve_vector_slice(key)
            return self.__class__.from_raw_parts(
                array=[
                    self._array[val_index]
                    for val_index in val_indices
                ],
                shape=(1, len(val_indices)),
            )

        val_index = self._resolve_vector_index(key)
        return self._array[val_index]

    def __contains__(self, value):
        return value in self._array

    def __deepcopy__(self, memo=None):
        """Return the matrix"""
        return self

    __copy__ = __deepcopy__

    @classmethod
    def from_raw_parts(cls, array, shape):
        """Construct a matrix directly from its ``array`` and ``shape`` parts

        This method exists primarily for the benefit of matrix-producing
        functions that have "pre-validated" the array and shape. Should be used
        with caution.

        The following properties are required to construct a valid matrix:
        - ``array`` must be a flattened ``list``. That is, the elements of the
          matrix must be contained within the shallowest depth of the ``list``
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
    def from_nested(cls, nested):
        """Construct a matrix from a singly-nested iterable, using the
        shallowest iterable's length to deduce the number of rows, and the
        nested iterables' lengths to deduce the number of columns

        Raises ``ValueError`` if the length of the nested iterables is
        inconsistent.
        """
        array = []

        rows = iter(nested)
        try:
            row = next(rows)
        except StopIteration:
            return cls.from_raw_parts(array=array, shape=(0, 0))
        else:
            array.extend(row)

        nrows1 = 1
        ncols1 = len(array)

        for nrows1, row in enumerate(rows, start=2):
            ncols2 = 0
            for ncols2, val in enumerate(row, start=1):
                array.append(val)
            if ncols1 != ncols2:
                raise ValueError(f"row {nrows1} has length {ncols2}, but precedent rows have length {ncols1}")

        return cls.from_raw_parts(array=array, shape=(nrows1, ncols1))

    @property
    def array(self):
        return SequenceView(self._array)

    @property
    def shape(self):
        return self._shape

    def equal(self, other):
        return IntegralMatrix(matrix_operator.__eq__(self, other))

    def not_equal(self, other):
        return IntegralMatrix(matrix_operator.__ne__(self, other))

    def transpose(self):
        from .views.builtins import MatrixTranspose
        return MatrixTranspose(self)

    def flip(self, *, by=Rule.ROW):
        from .views.builtins import MatrixColFlip, MatrixRowFlip
        MatrixTransform = (MatrixRowFlip, MatrixColFlip)[by.value]
        return MatrixTransform(self)

    def reverse(self):
        from .views.builtins import MatrixReverse
        return MatrixReverse(self)

    def values(self, *, by=Rule.ROW, reverse=False):
        values = reversed if reverse else iter
        if by is Rule.ROW:
            yield from values(self._array)
            return
        nrows = self.nrows
        ncols = self.ncols
        row_indices = range(nrows)
        col_indices = range(ncols)
        for col_index in values(col_indices):
            for row_index in values(row_indices):
                yield self._array[row_index * ncols + col_index]


class ComplexMatrix(ComplexMatrixLike[ComplexT_co, M_co, N_co], Matrix[ComplexT_co, M_co, N_co]):

    __slots__ = ()

    @check_friendly
    def __add__(self, other):
        return ComplexMatrix(matrix_operator.__add__(self, other))

    @check_friendly
    def __sub__(self, other):
        return ComplexMatrix(matrix_operator.__sub__(self, other))

    @check_friendly
    def __mul__(self, other):
        return ComplexMatrix(matrix_operator.__mul__(self, other))

    @check_friendly
    def __matmul__(self, other):
        return ComplexMatrix(matrix_operator.__matmul__(self, other))

    @check_friendly
    def __truediv__(self, other):
        return ComplexMatrix(matrix_operator.__truediv__(self, other))

    @check_friendly
    def __radd__(self, other):
        return ComplexMatrix(matrix_operator.__add__(other, self))

    @check_friendly
    def __rsub__(self, other):
        return ComplexMatrix(matrix_operator.__sub__(other, self))

    @check_friendly
    def __rmul__(self, other):
        return ComplexMatrix(matrix_operator.__mul__(other, self))

    @check_friendly
    def __rmatmul__(self, other):
        return ComplexMatrix(matrix_operator.__matmul__(other, self))

    @check_friendly
    def __rtruediv__(self, other):
        return ComplexMatrix(matrix_operator.__truediv__(other, self))

    def __neg__(self):
        return ComplexMatrix(matrix_operator.__neg__(self))

    def __abs__(self):
        return RealMatrix(matrix_operator.__abs__(self))

    def conjugate(self):
        return ComplexMatrix(matrix_operator.conjugate(self))

    # TODO: override transpose(), flip(), and reverse()


class RealMatrix(RealMatrixLike[RealT_co, M_co, N_co], Matrix[RealT_co, M_co, N_co]):

    __slots__ = ()

    @check_friendly
    def __add__(self, other):
        return RealMatrix(matrix_operator.__add__(self, other))

    @check_friendly
    def __sub__(self, other):
        return RealMatrix(matrix_operator.__sub__(self, other))

    @check_friendly
    def __mul__(self, other):
        return RealMatrix(matrix_operator.__mul__(self, other))

    @check_friendly
    def __matmul__(self, other):
        return RealMatrix(matrix_operator.__matmul__(self, other))

    @check_friendly
    def __truediv__(self, other):
        return RealMatrix(matrix_operator.__truediv__(self, other))

    @check_friendly
    def __floordiv__(self, other):
        return RealMatrix(matrix_operator.__floordiv__(self, other))

    @check_friendly
    def __mod__(self, other):
        return RealMatrix(matrix_operator.__mod__(self, other))

    @check_friendly
    def __divmod__(self, other):
        return Matrix(matrix_operator.__divmod__(self, other))

    @check_friendly
    def __radd__(self, other):
        return RealMatrix(matrix_operator.__add__(other, self))

    @check_friendly
    def __rsub__(self, other):
        return RealMatrix(matrix_operator.__sub__(other, self))

    @check_friendly
    def __rmul__(self, other):
        return RealMatrix(matrix_operator.__mul__(other, self))

    @check_friendly
    def __rmatmul__(self, other):
        return RealMatrix(matrix_operator.__matmul__(other, self))

    @check_friendly
    def __rtruediv__(self, other):
        return RealMatrix(matrix_operator.__truediv__(other, self))

    @check_friendly
    def __rfloordiv__(self, other):
        return RealMatrix(matrix_operator.__floordiv__(other, self))

    @check_friendly
    def __rmod__(self, other):
        return RealMatrix(matrix_operator.__mod__(other, self))

    @check_friendly
    def __rdivmod__(self, other):
        return Matrix(matrix_operator.__divmod__(other, self))

    def __neg__(self):
        return RealMatrix(matrix_operator.__neg__(self))

    def __abs__(self):
        return RealMatrix(matrix_operator.__abs__(self))

    def lesser(self, other):
        return IntegralMatrix(matrix_operator.__lt__(self, other))

    def lesser_equal(self, other):
        return IntegralMatrix(matrix_operator.__le__(self, other))

    def greater(self, other):
        return IntegralMatrix(matrix_operator.__gt__(self, other))

    def greater_equal(self, other):
        return IntegralMatrix(matrix_operator.__ge__(self, other))

    # TODO: override transpose(), flip(), and reverse()


class IntegralMatrix(IntegralMatrixLike[IntegralT_co, M_co, N_co], Matrix[IntegralT_co, M_co, N_co]):

    __slots__ = ()

    @check_friendly
    def __add__(self, other):
        return IntegralMatrix(matrix_operator.__add__(self, other))

    @check_friendly
    def __sub__(self, other):
        return IntegralMatrix(matrix_operator.__sub__(self, other))

    @check_friendly
    def __mul__(self, other):
        return IntegralMatrix(matrix_operator.__mul__(self, other))

    @check_friendly
    def __matmul__(self, other):
        return IntegralMatrix(matrix_operator.__matmul__(self, other))

    @check_friendly
    def __truediv__(self, other):
        return RealMatrix(matrix_operator.__truediv__(self, other))

    @check_friendly
    def __floordiv__(self, other):
        return IntegralMatrix(matrix_operator.__floordiv__(self, other))

    @check_friendly
    def __mod__(self, other):
        return IntegralMatrix(matrix_operator.__mod__(self, other))

    @check_friendly
    def __divmod__(self, other):
        return Matrix(matrix_operator.__divmod__(self, other))

    @check_friendly
    def __lshift__(self, other):
        return IntegralMatrix(matrix_operator.__lshift__(self, other))

    @check_friendly
    def __rshift__(self, other):
        return IntegralMatrix(matrix_operator.__rshift__(self, other))

    @check_friendly
    def __radd__(self, other):
        return IntegralMatrix(matrix_operator.__add__(other, self))

    @check_friendly
    def __rsub__(self, other):
        return IntegralMatrix(matrix_operator.__sub__(other, self))

    @check_friendly
    def __rmul__(self, other):
        return IntegralMatrix(matrix_operator.__mul__(other, self))

    @check_friendly
    def __rmatmul__(self, other):
        return IntegralMatrix(matrix_operator.__matmul__(other, self))

    @check_friendly
    def __rtruediv__(self, other):
        return RealMatrix(matrix_operator.__truediv__(other, self))

    @check_friendly
    def __rfloordiv__(self, other):
        return IntegralMatrix(matrix_operator.__floordiv__(other, self))

    @check_friendly
    def __rmod__(self, other):
        return IntegralMatrix(matrix_operator.__mod__(other, self))

    @check_friendly
    def __rdivmod__(self, other):
        return Matrix(matrix_operator.__divmod__(other, self))

    @check_friendly
    def __rlshift__(self, other):
        return IntegralMatrix(matrix_operator.__lshift__(other, self))

    @check_friendly
    def __rrshift__(self, other):
        return IntegralMatrix(matrix_operator.__rshift__(other, self))

    def __neg__(self):
        return IntegralMatrix(matrix_operator.__neg__(self))

    def __abs__(self):
        return IntegralMatrix(matrix_operator.__abs__(self))

    @check_friendly
    def __and__(self, other):
        return IntegralMatrix(matrix_operator.__and__(self, other))

    @check_friendly
    def __xor__(self, other):
        return IntegralMatrix(matrix_operator.__xor__(self, other))

    @check_friendly
    def __or__(self, other):
        return IntegralMatrix(matrix_operator.__or__(self, other))

    @check_friendly
    def __rand__(self, other):
        return IntegralMatrix(matrix_operator.__and__(other, self))

    @check_friendly
    def __rxor__(self, other):
        return IntegralMatrix(matrix_operator.__xor__(other, self))

    @check_friendly
    def __ror__(self, other):
        return IntegralMatrix(matrix_operator.__or__(other, self))

    def __invert__(self):
        return IntegralMatrix(matrix_operator.__invert__(self))

    def lesser(self, other):
        return IntegralMatrix(matrix_operator.__lt__(self, other))

    def lesser_equal(self, other):
        return IntegralMatrix(matrix_operator.__le__(self, other))

    def greater(self, other):
        return IntegralMatrix(matrix_operator.__gt__(self, other))

    def greater_equal(self, other):
        return IntegralMatrix(matrix_operator.__ge__(self, other))

    # TODO: override transpose(), flip(), and reverse()
