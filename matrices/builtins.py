from typing import Generic, TypeVar

from .abc import (ComplexMatrixLike, IntegralMatrixLike, MatrixLike,
                  RealMatrixLike, ShapedIterable, check_friendly)
from .rule import COL, ROW, Rule
from .unsafe import unsafe
from .utilities.matrix_map import MatrixMap
from .utilities.matrix_operator import (__abs__, __add__, __and__, __divmod__,
                                        __eq__, __floordiv__, __ge__, __gt__,
                                        __invert__, __le__, __lshift__, __lt__,
                                        __matmul__, __mod__, __mul__, __ne__,
                                        __neg__, __or__, __pos__, __rshift__,
                                        __sub__, __truediv__, __xor__,
                                        conjugate)
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

ComplexT_co = TypeVar("ComplexT_co", covariant=True, bound=complex)
RealT_co = TypeVar("RealT_co", covariant=True, bound=float)
IntegralT_co = TypeVar("IntegralT_co", covariant=True, bound=int)


class MatrixOperatorsMixin(Generic[T_co, M_co, N_co]):
    """Mixin class that defines the "operator methods" of ``MatrixLike`` using
    built-in matrix types
    """

    __slots__ = ()

    def equal(self, other):
        return IntegralMatrix(__eq__(self, other))

    def not_equal(self, other):
        return IntegralMatrix(__ne__(self, other))


class Matrix(MatrixOperatorsMixin[T_co, M_co, N_co], MatrixLike[T_co, M_co, N_co]):

    __slots__ = ("_array", "_shape")

    STORAGE_FACTORY = tuple

    def __init__(self, array=(), shape=(None, None)):
        self._array = self.STORAGE_FACTORY(array)

        if isinstance(array, (MatrixMap, MatrixProduct, MatrixLike, ShapedIterable)):
            self._shape = array.shape
            return

        nrows = shape[0]
        ncols = shape[1]
        size  = len(self._array)

        if nrows is None and ncols is None:
            shape = (1, size)
        elif nrows is None:
            nrows, loss = divmod(size, ncols) if ncols else (0, size)
            if loss:
                raise ValueError(f"cannot interpret array of size {size} as shape M × {ncols}")
            shape = (nrows, ncols)
        elif ncols is None:
            ncols, loss = divmod(size, nrows) if nrows else (0, size)
            if loss:
                raise ValueError(f"cannot interpret array of size {size} as shape {nrows} × N")
            shape = (nrows, ncols)
        else:
            if nrows * ncols != size:
                raise ValueError(f"cannot interpret array of size {size} as shape {nrows} × {ncols}")
        if shape[0] < 0 or shape[1] < 0:
            raise ValueError("dimensions must be non-negative")

        self._shape = shape

    def __repr__(self):
        """Return a canonical representation of the matrix"""
        return f"{self.__class__.__name__}(array={self._array!r}, shape={self._shape!r})"

    def __len__(self):
        return len(self._array)

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

    def __contains__(self, value):
        return value in self._array

    def __deepcopy__(self, memo=None):
        """Return the matrix"""
        return self

    __copy__ = __deepcopy__

    @classmethod
    @unsafe
    def from_raw_parts(cls, array, shape):
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
    def from_nesting(cls, nesting):
        """Construct a matrix from a singly-nested iterable, using the
        shallowest iterable's length to deduce the number of rows, and the
        nested iterables' lengths to deduce the number of columns

        Raises ``ValueError`` if the length of the nested iterables is
        inconsistent.
        """
        nrows = 0
        ncols = 0

        def flatten(nesting):
            nonlocal nrows, ncols

            rows = iter(nesting)
            try:
                row = next(rows)
            except StopIteration:
                return
            else:
                nrows += 1

            for val in row:
                yield val
                ncols += 1

            for row in rows:
                n = 0
                for val in row:
                    yield val
                    n += 1
                if n != ncols:
                    raise ValueError(f"row {nrows} has length {n}, but precedent rows have length {ncols}")
                nrows += 1

        return cls(flatten(nesting), shape=(nrows, ncols))

    @property
    def array(self):
        return self._array

    @property
    def shape(self):
        return self._shape

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


class ComplexMatrixOperatorsMixin(Generic[ComplexT_co, M_co, N_co]):
    """Mixin class that defines the "operator methods" of ``ComplexMatrixLike``
    using built-in matrix types
    """

    __slots__ = ()

    @check_friendly
    def __add__(self, other):
        return ComplexMatrix(__add__(self, other))

    @check_friendly
    def __sub__(self, other):
        return ComplexMatrix(__sub__(self, other))

    @check_friendly
    def __mul__(self, other):
        return ComplexMatrix(__mul__(self, other))

    @check_friendly
    def __matmul__(self, other):
        return ComplexMatrix(__matmul__(self, other))

    @check_friendly
    def __truediv__(self, other):
        return ComplexMatrix(__truediv__(self, other))

    @check_friendly
    def __radd__(self, other):
        return ComplexMatrix(__add__(other, self))

    @check_friendly
    def __rsub__(self, other):
        return ComplexMatrix(__sub__(other, self))

    @check_friendly
    def __rmul__(self, other):
        return ComplexMatrix(__mul__(other, self))

    @check_friendly
    def __rmatmul__(self, other):
        return ComplexMatrix(__matmul__(other, self))

    @check_friendly
    def __rtruediv__(self, other):
        return ComplexMatrix(__truediv__(other, self))

    def __neg__(self):
        return ComplexMatrix(__neg__(self))

    def __abs__(self):
        return RealMatrix(__abs__(self))

    def conjugate(self):
        return ComplexMatrix(conjugate(self))


class ComplexMatrix(
    ComplexMatrixOperatorsMixin[ComplexT_co, M_co, N_co],
    ComplexMatrixLike[ComplexT_co, M_co, N_co],
    Matrix[ComplexT_co, M_co, N_co],
):

    __slots__ = ()

    def transpose(self):
        from .views.builtins import ComplexMatrixTranspose
        return ComplexMatrixTranspose(self)

    def flip(self, *, by=Rule.ROW):
        from .views.builtins import ComplexMatrixColFlip, ComplexMatrixRowFlip
        MatrixTransform = (ComplexMatrixRowFlip, ComplexMatrixColFlip)[by.value]
        return MatrixTransform(self)

    def reverse(self):
        from .views.builtins import ComplexMatrixReverse
        return ComplexMatrixReverse(self)


class RealMatrixOperatorsMixin(Generic[RealT_co, M_co, N_co]):
    """Mixin class that defines the "operator methods" of ``RealMatrixLike``
    using built-in matrix types
    """

    __slots__ = ()

    @check_friendly
    def __add__(self, other):
        return RealMatrix(__add__(self, other))

    @check_friendly
    def __sub__(self, other):
        return RealMatrix(__sub__(self, other))

    @check_friendly
    def __mul__(self, other):
        return RealMatrix(__mul__(self, other))

    @check_friendly
    def __matmul__(self, other):
        return RealMatrix(__matmul__(self, other))

    @check_friendly
    def __truediv__(self, other):
        return RealMatrix(__truediv__(self, other))

    @check_friendly
    def __floordiv__(self, other):
        return RealMatrix(__floordiv__(self, other))

    @check_friendly
    def __mod__(self, other):
        return RealMatrix(__mod__(self, other))

    @check_friendly
    def __divmod__(self, other):
        return Matrix(__divmod__(self, other))

    @check_friendly
    def __radd__(self, other):
        return RealMatrix(__add__(other, self))

    @check_friendly
    def __rsub__(self, other):
        return RealMatrix(__sub__(other, self))

    @check_friendly
    def __rmul__(self, other):
        return RealMatrix(__mul__(other, self))

    @check_friendly
    def __rmatmul__(self, other):
        return RealMatrix(__matmul__(other, self))

    @check_friendly
    def __rtruediv__(self, other):
        return RealMatrix(__truediv__(other, self))

    @check_friendly
    def __rfloordiv__(self, other):
        return RealMatrix(__floordiv__(other, self))

    @check_friendly
    def __rmod__(self, other):
        return RealMatrix(__mod__(other, self))

    @check_friendly
    def __rdivmod__(self, other):
        return Matrix(__divmod__(other, self))

    def __neg__(self):
        return RealMatrix(__neg__(self))

    def __abs__(self):
        return RealMatrix(__abs__(self))

    def lesser(self, other):
        return IntegralMatrix(__lt__(self, other))

    def lesser_equal(self, other):
        return IntegralMatrix(__le__(self, other))

    def greater(self, other):
        return IntegralMatrix(__gt__(self, other))

    def greater_equal(self, other):
        return IntegralMatrix(__ge__(self, other))


class RealMatrix(
    RealMatrixOperatorsMixin[RealT_co, M_co, N_co],
    RealMatrixLike[RealT_co, M_co, N_co],
    Matrix[RealT_co, M_co, N_co],
):

    __slots__ = ()

    def transpose(self):
        from .views.builtins import RealMatrixTranspose
        return RealMatrixTranspose(self)

    def flip(self, *, by=Rule.ROW):
        from .views.builtins import RealMatrixColFlip, RealMatrixRowFlip
        MatrixTransform = (RealMatrixRowFlip, RealMatrixColFlip)[by.value]
        return MatrixTransform(self)

    def reverse(self):
        from .views.builtins import RealMatrixReverse
        return RealMatrixReverse(self)


class IntegralMatrixOperatorsMixin(Generic[IntegralT_co, M_co, N_co]):
    """Mixin class that defines the "operator methods" of
    ``IntegralMatrixLike`` using built-in matrix types
    """

    __slots__ = ()

    @check_friendly
    def __add__(self, other):
        return IntegralMatrix(__add__(self, other))

    @check_friendly
    def __sub__(self, other):
        return IntegralMatrix(__sub__(self, other))

    @check_friendly
    def __mul__(self, other):
        return IntegralMatrix(__mul__(self, other))

    @check_friendly
    def __matmul__(self, other):
        return IntegralMatrix(__matmul__(self, other))

    @check_friendly
    def __truediv__(self, other):
        return RealMatrix(__truediv__(self, other))

    @check_friendly
    def __floordiv__(self, other):
        return IntegralMatrix(__floordiv__(self, other))

    @check_friendly
    def __mod__(self, other):
        return IntegralMatrix(__mod__(self, other))

    @check_friendly
    def __divmod__(self, other):
        return Matrix(__divmod__(self, other))

    @check_friendly
    def __lshift__(self, other):
        return IntegralMatrix(__lshift__(self, other))

    @check_friendly
    def __rshift__(self, other):
        return IntegralMatrix(__rshift__(self, other))

    @check_friendly
    def __and__(self, other):
        return IntegralMatrix(__and__(self, other))

    @check_friendly
    def __xor__(self, other):
        return IntegralMatrix(__xor__(self, other))

    @check_friendly
    def __or__(self, other):
        return IntegralMatrix(__or__(self, other))

    @check_friendly
    def __radd__(self, other):
        return IntegralMatrix(__add__(other, self))

    @check_friendly
    def __rsub__(self, other):
        return IntegralMatrix(__sub__(other, self))

    @check_friendly
    def __rmul__(self, other):
        return IntegralMatrix(__mul__(other, self))

    @check_friendly
    def __rmatmul__(self, other):
        return IntegralMatrix(__matmul__(other, self))

    @check_friendly
    def __rtruediv__(self, other):
        return RealMatrix(__truediv__(other, self))

    @check_friendly
    def __rfloordiv__(self, other):
        return IntegralMatrix(__floordiv__(other, self))

    @check_friendly
    def __rmod__(self, other):
        return IntegralMatrix(__mod__(other, self))

    @check_friendly
    def __rdivmod__(self, other):
        return Matrix(__divmod__(other, self))

    @check_friendly
    def __rlshift__(self, other):
        return IntegralMatrix(__lshift__(other, self))

    @check_friendly
    def __rrshift__(self, other):
        return IntegralMatrix(__rshift__(other, self))

    @check_friendly
    def __rand__(self, other):
        return IntegralMatrix(__and__(other, self))

    @check_friendly
    def __rxor__(self, other):
        return IntegralMatrix(__xor__(other, self))

    @check_friendly
    def __ror__(self, other):
        return IntegralMatrix(__or__(other, self))

    def __neg__(self):
        return IntegralMatrix(__neg__(self))

    def __abs__(self):
        return IntegralMatrix(__abs__(self))

    def __invert__(self):
        return IntegralMatrix(__invert__(self))

    def lesser(self, other):
        return IntegralMatrix(__lt__(self, other))

    def lesser_equal(self, other):
        return IntegralMatrix(__le__(self, other))

    def greater(self, other):
        return IntegralMatrix(__gt__(self, other))

    def greater_equal(self, other):
        return IntegralMatrix(__ge__(self, other))


class IntegralMatrix(
    IntegralMatrixOperatorsMixin[IntegralT_co, M_co, N_co],
    IntegralMatrixLike[IntegralT_co, M_co, N_co],
    Matrix[IntegralT_co, M_co, N_co],
):

    __slots__ = ()

    def transpose(self):
        from .views.builtins import IntegralMatrixTranspose
        return IntegralMatrixTranspose(self)

    def flip(self, *, by=Rule.ROW):
        from .views.builtins import (IntegralMatrixColFlip,
                                     IntegralMatrixRowFlip)
        MatrixTransform = (IntegralMatrixRowFlip, IntegralMatrixColFlip)[by.value]
        return MatrixTransform(self)

    def reverse(self):
        from .views.builtins import IntegralMatrixReverse
        return IntegralMatrixReverse(self)
