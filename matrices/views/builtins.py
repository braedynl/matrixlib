from reprlib import recursive_repr
from typing import TypeVar

from ..abc import (ComplexMatrixLike, IntegralMatrixLike, RealMatrixLike,
                   check_friendly)
from ..builtins import ComplexMatrix, IntegralMatrix, Matrix, RealMatrix
from ..rule import COL, ROW, Rule
from ..utilities import matrix_operator
from .abc import MatrixViewLike

__all__ = [
    # Basic viewers
    "MatrixView", "ComplexMatrixView", "RealMatrixView", "IntegralMatrixView",

    # Base transformers
    "MatrixTransform", "ComplexMatrixTransform", "RealMatrixTransform",
    "IntegralMatrixTransform",

    # Matrix transformers
    "MatrixTranspose", "MatrixRowFlip", "MatrixColFlip", "MatrixReverse",

    # Complex Matrix transformers
    "ComplexMatrixTranspose", "ComplexMatrixRowFlip", "ComplexMatrixColFlip",
    "ComplexMatrixReverse",

    # Real Matrix transformers
    "RealMatrixTranspose", "RealMatrixRowFlip", "RealMatrixColFlip",
    "RealMatrixReverse",

    # Integral Matrix transformers
    "IntegralMatrixTranspose", "IntegralMatrixRowFlip",
    "IntegralMatrixColFlip", "IntegralMatrixReverse",
]

T = TypeVar("T")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)

ComplexT = TypeVar("ComplexT", bound=complex)
RealT = TypeVar("RealT", bound=float)
IntegralT = TypeVar("IntegralT", bound=int)


class MatrixView(MatrixViewLike[T, M, N]):

    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    @recursive_repr("...")
    def __repr__(self):
        """Return a canonical representation of the view"""
        return f"{self.__class__.__name__}(target={self._target!r})"

    def __getitem__(self, key):
        return self._target.__getitem__(key)

    def __deepcopy__(self, memo=None):
        """Return the view"""
        return self

    __copy__ = __deepcopy__

    @property
    def array(self):
        return self._target.array

    @property
    def shape(self):
        return self._target.shape

    def equal(self, other):
        return self._target.equal(other)

    def not_equal(self, other):
        return self._target.not_equal(other)

    def transpose(self):
        return self._target.transpose()

    def flip(self, *, by=Rule.ROW):
        return self._target.flip(by=by)

    def reverse(self):
        return self._target.reverse()


class ComplexMatrixView(ComplexMatrixLike[ComplexT, M, N], MatrixView[ComplexT, M, N]):

    __slots__ = ()

    def __add__(self, other):
        return self._target.__add__(other)

    def __sub__(self, other):
        return self._target.__sub__(other)

    def __mul__(self, other):
        return self._target.__mul__(other)

    def __matmul__(self, other):
        return self._target.__matmul__(other)

    def __truediv__(self, other):
        return self._target.__truediv__(other)

    def __radd__(self, other):
        return self._target.__radd__(other)

    def __rsub__(self, other):
        return self._target.__rsub__(other)

    def __rmul__(self, other):
        return self._target.__rmul__(other)

    def __rmatmul__(self, other):
        return self._target.__rmatmul__(other)

    def __rtruediv__(self, other):
        return self._target.__rtruediv__(other)

    def __neg__(self):
        return self._target.__neg__()

    def __abs__(self):
        return self._target.__abs__()

    def conjugate(self):
        return self._target.conjugate()


class RealMatrixView(RealMatrixLike[RealT, M, N], MatrixView[RealT, M, N]):

    __slots__ = ()

    def __lt__(self, other):
        return self._target.__lt__(other)

    def __le__(self, other):
        return self._target.__le__(other)

    def __gt__(self, other):
        return self._target.__gt__(other)

    def __ge__(self, other):
        return self._target.__ge__(other)

    def __add__(self, other):
        return self._target.__add__(other)

    def __sub__(self, other):
        return self._target.__sub__(other)

    def __mul__(self, other):
        return self._target.__mul__(other)

    def __matmul__(self, other):
        return self._target.__matmul__(other)

    def __truediv__(self, other):
        return self._target.__truediv__(other)

    def __floordiv__(self, other):
        return self._target.__floordiv__(other)

    def __mod__(self, other):
        return self._target.__mod__(other)

    def __divmod__(self, other):
        return self._target.__divmod__(other)

    def __radd__(self, other):
        return self._target.__radd__(other)

    def __rsub__(self, other):
        return self._target.__rsub__(other)

    def __rmul__(self, other):
        return self._target.__rmul__(other)

    def __rmatmul__(self, other):
        return self._target.__rmatmul__(other)

    def __rtruediv__(self, other):
        return self._target.__rtruediv__(other)

    def __rfloordiv__(self, other):
        return self._target.__rfloordiv__(other)

    def __rmod__(self, other):
        return self._target.__rmod__(other)

    def __rdivmod__(self, other):
        return self._target.__rdivmod__(other)

    def __neg__(self):
        return self._target.__neg__()

    def __abs__(self):
        return self._target.__abs__()

    def lesser(self, other):
        return self._target.lesser(other)

    def lesser_equal(self, other):
        return self._target.lesser_equal(other)

    def greater(self, other):
        return self._target.greater(other)

    def greater_equal(self, other):
        return self._target.greater_equal(other)


class IntegralMatrixView(IntegralMatrixLike[IntegralT, M, N], MatrixView[IntegralT, M, N]):

    __slots__ = ()

    def __lt__(self, other):
        return self._target.__lt__(other)

    def __le__(self, other):
        return self._target.__le__(other)

    def __gt__(self, other):
        return self._target.__gt__(other)

    def __ge__(self, other):
        return self._target.__ge__(other)

    def __add__(self, other):
        return self._target.__add__(other)

    def __sub__(self, other):
        return self._target.__sub__(other)

    def __mul__(self, other):
        return self._target.__mul__(other)

    def __matmul__(self, other):
        return self._target.__matmul__(other)

    def __truediv__(self, other):
        return self._target.__truediv__(other)

    def __floordiv__(self, other):
        return self._target.__floordiv__(other)

    def __mod__(self, other):
        return self._target.__mod__(other)

    def __divmod__(self, other):
        return self._target.__divmod__(other)

    def __lshift__(self, other):
        return self._target.__lshift__(other)

    def __rshift__(self, other):
        return self._target.__rshift__(other)

    def __and__(self, other):
        return self._target.__and__(other)

    def __xor__(self, other):
        return self._target.__xor__(other)

    def __or__(self, other):
        return self._target.__or__(other)

    def __radd__(self, other):
        return self._target.__radd__(other)

    def __rsub__(self, other):
        return self._target.__rsub__(other)

    def __rmul__(self, other):
        return self._target.__rmul__(other)

    def __rmatmul__(self, other):
        return self._target.__rmatmul__(other)

    def __rtruediv__(self, other):
        return self._target.__rtruediv__(other)

    def __rfloordiv__(self, other):
        return self._target.__rfloordiv__(other)

    def __rmod__(self, other):
        return self._target.__rmod__(other)

    def __rdivmod__(self, other):
        return self._target.__rdivmod__(other)

    def __rlshift__(self, other):
        return self._target.__rlshift__(other)

    def __rrshift__(self, other):
        return self._target.__rrshift__(other)

    def __rand__(self, other):
        return self._target.__rand__(other)

    def __rxor__(self, other):
        return self._target.__rxor__(other)

    def __ror__(self, other):
        return self._target.__ror__(other)

    def __neg__(self):
        return self._target.__neg__()

    def __abs__(self):
        return self._target.__abs__()

    def __invert__(self):
        return self._target.__invert__()

    def lesser(self, other):
        return self._target.lesser(other)

    def lesser_equal(self, other):
        return self._target.lesser_equal(other)

    def greater(self, other):
        return self._target.greater(other)

    def greater_equal(self, other):
        return self._target.greater_equal(other)


class MatrixTransform(MatrixViewLike[T, M, N]):

    __slots__ = ("_target",)

    BASE_TYPE = Matrix
    VIEW_TYPE = MatrixView

    def __init__(self, target) -> None:
        self._target = target

    @recursive_repr("...")
    def __repr__(self):
        """Return a canonical representation of the view"""
        return f"{self.__class__.__name__}(target={self._target!r})"

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key

            if isinstance(row_key, slice):
                row_indices = self._resolve_matrix_slice(row_key, by=ROW)

                if isinstance(col_key, slice):
                    col_indices = self._resolve_matrix_slice(col_key, by=COL)
                    return self.BASE_TYPE.from_raw_parts(
                        array=[
                            self._target[
                                self._permute_matrix_index(
                                    row_index=row_index,
                                    col_index=col_index,
                                )
                            ]
                            for row_index in row_indices
                            for col_index in col_indices
                        ],
                        shape=(len(row_indices), len(col_indices)),
                    )

                col_index = self._resolve_matrix_index(col_key, by=COL)
                return self.BASE_TYPE.from_raw_parts(
                    array=[
                        self._target[
                            self._permute_matrix_index(
                                row_index=row_index,
                                col_index=col_index,
                            )
                        ]
                        for row_index in row_indices
                    ],
                    shape=(len(row_indices), 1),
                )

            row_index = self._resolve_matrix_index(row_key, by=ROW)

            if isinstance(col_key, slice):
                col_indices = self._resolve_matrix_slice(col_key, by=COL)
                return self.BASE_TYPE.from_raw_parts(
                    array=[
                        self._target[
                            self._permute_matrix_index(
                                row_index=row_index,
                                col_index=col_index,
                            )
                        ]
                        for col_index in col_indices
                    ],
                    shape=(1, len(col_indices)),
                )

            col_index = self._resolve_matrix_index(col_key, by=COL)
            return self._target[
                self._permute_matrix_index(
                    row_index=row_index,
                    col_index=col_index,
                )
            ]

        if isinstance(key, slice):
            val_indices = self._resolve_vector_slice(key)
            return self.BASE_TYPE.from_raw_parts(
                array=[
                    self._target[
                        self._permute_vector_index(
                            val_index=val_index,
                        )
                    ]
                    for val_index in val_indices
                ],
                shape=(1, len(val_indices)),
            )

        val_index = self._resolve_vector_index(key)
        return self._target[
            self._permute_vector_index(
                val_index=val_index,
            )
        ]

    def __deepcopy__(self, memo=None):
        return self

    __copy__ = __deepcopy__

    @property
    def array(self):
        return list(self.values())

    @property
    def shape(self):
        return self._target.shape

    def equal(self, other):
        return IntegralMatrix(matrix_operator.__eq__(self, other))

    def not_equal(self, other):
        return IntegralMatrix(matrix_operator.__ne__(self, other))

    def transpose(self):
        return MatrixTranspose(self)

    def flip(self, *, by=Rule.ROW):
        MatrixTransform = (MatrixRowFlip, MatrixColFlip)[by.value]
        return MatrixTransform(self)

    def reverse(self):
        return MatrixReverse(self)

    def _permute_vector_index(self, val_index):
        return val_index

    def _permute_matrix_index(self, row_index, col_index):
        return row_index * self.ncols + col_index


class ComplexMatrixTransform(ComplexMatrixLike[ComplexT, M, N], MatrixTransform[ComplexT, M, N]):

    __slots__ = ()

    BASE_TYPE = ComplexMatrix
    VIEW_TYPE = ComplexMatrixView

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

    def transpose(self):
        return ComplexMatrixTranspose(self)

    def flip(self, *, by=Rule.ROW):
        MatrixTransform = (ComplexMatrixRowFlip, ComplexMatrixColFlip)[by.value]
        return MatrixTransform(self)

    def reverse(self):
        return ComplexMatrixReverse(self)

    def conjugate(self):
        return ComplexMatrix(matrix_operator.conjugate(self))


class RealMatrixTransform(RealMatrixLike[RealT, M, N], MatrixTransform[RealT, M, N]):

    __slots__ = ()

    BASE_TYPE = RealMatrix
    VIEW_TYPE = RealMatrixView

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

    def transpose(self):
        return RealMatrixTranspose(self)

    def flip(self, *, by=Rule.ROW):
        MatrixTransform = (RealMatrixRowFlip, RealMatrixColFlip)[by.value]
        return MatrixTransform(self)

    def reverse(self):
        return RealMatrixReverse(self)


class IntegralMatrixTransform(IntegralMatrixLike[IntegralT, M, N], MatrixTransform[IntegralT, M, N]):

    __slots__ = ()

    BASE_TYPE = IntegralMatrix
    VIEW_TYPE = IntegralMatrixView

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
    def __and__(self, other):
        return IntegralMatrix(matrix_operator.__and__(self, other))

    @check_friendly
    def __xor__(self, other):
        return IntegralMatrix(matrix_operator.__xor__(self, other))

    @check_friendly
    def __or__(self, other):
        return IntegralMatrix(matrix_operator.__or__(self, other))

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

    @check_friendly
    def __rand__(self, other):
        return IntegralMatrix(matrix_operator.__and__(other, self))

    @check_friendly
    def __rxor__(self, other):
        return IntegralMatrix(matrix_operator.__xor__(other, self))

    @check_friendly
    def __ror__(self, other):
        return IntegralMatrix(matrix_operator.__or__(other, self))

    def __neg__(self):
        return IntegralMatrix(matrix_operator.__neg__(self))

    def __abs__(self):
        return IntegralMatrix(matrix_operator.__abs__(self))

    def __invert__(self):
        return IntegralMatrix(matrix_operator.__invert__(self))

    def transpose(self):
        return IntegralMatrixTranspose(self)

    def flip(self, *, by=Rule.ROW):
        MatrixTransform = (IntegralMatrixRowFlip, IntegralMatrixColFlip)[by.value]
        return MatrixTransform(self)

    def reverse(self):
        return IntegralMatrixReverse(self)


class TransposeMixin:

    __slots__ = ()

    @property
    def shape(self):
        return tuple(reversed(self._target.shape))

    @property
    def nrows(self):
        return self._target.ncols

    @property
    def ncols(self):
        return self._target.nrows

    def transpose(self):
        return self.VIEW_TYPE(self._target)

    def n(self, by):
        return self._target.n(~by)

    def _permute_vector_index(self, val_index):
        row_index, col_index = divmod(val_index, self.ncols)
        return self._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )

    def _permute_matrix_index(self, row_index, col_index):
        return col_index * self.nrows + row_index


class RowFlipMixin:

    __slots__ = ()

    def flip(self, *, by=Rule.ROW):
        if by is Rule.ROW:
            return self.VIEW_TYPE(self._target)
        return super().flip(by=by)

    def _permute_vector_index(self, val_index):
        row_index, col_index = divmod(val_index, self.ncols)
        return self._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )

    def _permute_matrix_index(self, row_index, col_index):
        row_index = self.nrows - row_index - 1
        return super()._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )


class ColFlipMixin:

    __slots__ = ()

    def flip(self, *, by=Rule.ROW):
        if by is Rule.COL:
            return self.VIEW_TYPE(self._target)
        return super().flip(by=by)

    def _permute_vector_index(self, val_index):
        row_index, col_index = divmod(val_index, self.ncols)
        return self._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )

    def _permute_matrix_index(self, row_index, col_index):
        col_index = self.ncols - col_index - 1
        return super()._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )


class ReverseMixin:

    __slots__ = ()

    def reverse(self):
        return self.VIEW_TYPE(self._target)

    def _permute_vector_index(self, val_index):
        return self.size - val_index - 1

    def _permute_matrix_index(self, row_index, col_index):
        return self._permute_vector_index(
            val_index=super()._permute_matrix_index(
                row_index=row_index,
                col_index=col_index,
            ),
        )


class MatrixTranspose(TransposeMixin, MatrixTransform[T, M, N]):
    __slots__ = ()
class MatrixRowFlip(RowFlipMixin, MatrixTransform[T, M, N]):
    __slots__ = ()
class MatrixColFlip(ColFlipMixin, MatrixTransform[T, M, N]):
    __slots__ = ()
class MatrixReverse(ReverseMixin, MatrixTransform[T, M, N]):
    __slots__ = ()
class ComplexMatrixTranspose(TransposeMixin, ComplexMatrixTransform[ComplexT, M, N]):
    __slots__ = ()
class ComplexMatrixRowFlip(RowFlipMixin, ComplexMatrixTransform[ComplexT, M, N]):
    __slots__ = ()
class ComplexMatrixColFlip(ColFlipMixin, ComplexMatrixTransform[ComplexT, M, N]):
    __slots__ = ()
class ComplexMatrixReverse(ReverseMixin, ComplexMatrixTransform[ComplexT, M, N]):
    __slots__ = ()
class RealMatrixTranspose(TransposeMixin, RealMatrixTransform[RealT, M, N]):
    __slots__ = ()
class RealMatrixRowFlip(RowFlipMixin, RealMatrixTransform[RealT, M, N]):
    __slots__ = ()
class RealMatrixColFlip(ColFlipMixin, RealMatrixTransform[RealT, M, N]):
    __slots__ = ()
class RealMatrixReverse(ReverseMixin, RealMatrixTransform[RealT, M, N]):
    __slots__ = ()
class IntegralMatrixTranspose(TransposeMixin, IntegralMatrixTransform[IntegralT, M, N]):
    __slots__ = ()
class IntegralMatrixRowFlip(RowFlipMixin, IntegralMatrixTransform[IntegralT, M, N]):
    __slots__ = ()
class IntegralMatrixColFlip(ColFlipMixin, IntegralMatrixTransform[IntegralT, M, N]):
    __slots__ = ()
class IntegralMatrixReverse(ReverseMixin, IntegralMatrixTransform[IntegralT, M, N]):
    __slots__ = ()
