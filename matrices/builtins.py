import copy
import itertools
import operator
from reprlib import recursive_repr
from typing import TypeVar

from .abc import MatrixLike
from .rule import COL, ROW, Rule
from .shapes.builtins import Shape
from .utilities.checked_map import checked_map
from .utilities.logical_operator import logical_and, logical_not, logical_or

__all__ = ["FrozenMatrix", "Matrix"]

T = TypeVar("T")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class FrozenMatrix(MatrixLike[T_co, M_co, N_co]):

    __slots__ = ("_array", "_shape")

    def __init__(self, array=None, shape=None):
        array = [] if array is None else list(array)

        n = len(array)
        if shape is None:
            nrows, ncols = (1, n)
        else:
            nrows, ncols = shape
            if nrows is None and ncols is None:
                nrows = 1
                ncols = n
            elif nrows is None:
                nrows, loss = divmod(n, ncols) if ncols else (0, n)
                if loss:
                    raise ValueError(f"cannot interpret array of size {n} as shape M × {ncols}")
            elif ncols is None:
                ncols, loss = divmod(n, nrows) if nrows else (0, n)
                if loss:
                    raise ValueError(f"cannot interpret array of size {n} as shape {nrows} × N")
            else:
                m = nrows * ncols
                if m != n:
                    raise ValueError(f"cannot interpret array of size {n} as shape {nrows} × {ncols}")

        shape = Shape(nrows, ncols)

        self._array = array
        self._shape = shape

    @recursive_repr("...")
    def __repr__(self):
        """Return a canonical representation of the matrix"""
        array = self._array
        shape = self._shape
        return f"{self.__class__.__name__}({array!r}, shape=({shape[0]!r}, {shape[1]!r}))"

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
            ncols = self.ncols

            if isinstance(row_key, slice):
                row_indices = self._resolve_matrix_slice(row_key, by=ROW)

                if isinstance(col_key, slice):
                    col_indices = self._resolve_matrix_slice(col_key, by=COL)
                    return FrozenMatrix.wrap(
                        [
                            self._array[row_index * ncols + col_index]
                            for row_index in row_indices
                            for col_index in col_indices
                        ],
                        shape=Shape(len(row_indices), len(col_indices)),
                    )

                col_index = self._resolve_matrix_index(col_key, by=COL)
                return FrozenMatrix.wrap(
                    [
                        self._array[row_index * ncols + col_index]
                        for row_index in row_indices
                    ],
                    shape=Shape(len(row_indices), 1),
                )

            row_index = self._resolve_matrix_index(row_key, by=ROW)

            if isinstance(col_key, slice):
                col_indices = self._resolve_matrix_slice(col_key, by=COL)
                return FrozenMatrix.wrap(
                    [
                        self._array[row_index * ncols + col_index]
                        for col_index in col_indices
                    ],
                    shape=Shape(1, len(col_indices)),
                )

            col_index = self._resolve_matrix_index(col_key, by=COL)
            return self._array[row_index * ncols + col_index]

        if isinstance(key, slice):
            val_indices = self._resolve_vector_slice(key)
            return FrozenMatrix.wrap(
                [
                    self._array[val_index]
                    for val_index in val_indices
                ],
                shape=Shape(1, len(val_indices)),
            )

        val_index = self._resolve_vector_index(key)
        return self._array[val_index]

    def __contains__(self, value):
        return value in self._array

    def __add__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix.wrap(
                list(checked_map(operator.__add__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix.wrap(
                list(checked_map(operator.__sub__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix.wrap(
                list(checked_map(operator.__mul__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix.wrap(
                list(checked_map(operator.__truediv__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix.wrap(
                list(checked_map(operator.__floordiv__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __mod__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix.wrap(
                list(checked_map(operator.__mod__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __divmod__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix.wrap(
                list(checked_map(divmod, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix.wrap(
                list(checked_map(operator.__pow__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __lshift__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix.wrap(
                list(checked_map(operator.__lshift__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __rshift__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix.wrap(
                list(checked_map(operator.__rshift__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __and__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix.wrap(
                list(checked_map(operator.__and__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __xor__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix.wrap(
                list(checked_map(operator.__xor__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __or__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix.wrap(
                list(checked_map(operator.__or__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __matmul__(self, other):  # TODO
        pass

    def __neg__(self):
        return FrozenMatrix.wrap(
            list(map(operator.__neg__, self)),
            shape=self.shape,
        )

    def __pos__(self):
        return FrozenMatrix.wrap(
            list(map(operator.__pos__, self)),
            shape=self.shape,
        )

    def __abs__(self):
        return FrozenMatrix.wrap(
            list(map(abs, self)),
            shape=self.shape,
        )

    def __invert__(self):
        return FrozenMatrix.wrap(
            list(map(operator.__invert__, self)),
            shape=self.shape,
        )

    def __deepcopy__(self, memo=None):
        """Return the matrix"""
        return self

    __copy__ = __deepcopy__

    @classmethod
    def wrap(cls, array, shape):
        """Construct a matrix directly from a mutable sequence and shape

        This method exists primarily for the benefit of matrix-producing
        functions that have "pre-validated" the data and its dimensions. Should
        be used with caution - this method is not marked as internal because
        its usage is not entirely discouraged if you're aware of the dangers.

        The following properties are required to construct a valid matrix:
        - `array` must be a flattened `MutableSequence`. That is, the elements
          of the matrix must be on the shallowest depth. A nested sequence
          would imply a matrix that contains sequence instances.
        - `shape` must be a `Shape`, where the product of its values must equal
          the length of the sequence.
        """
        self = cls.__new__(cls)
        self._array = array
        self._shape = shape
        return self

    @classmethod
    def fill(cls, value, shape):
        """Construct a matrix of shape `nrows` × `ncols`, comprised solely of
        `value`
        """
        nrows, ncols = shape
        return cls.wrap([value] * (nrows * ncols), shape=Shape(nrows, ncols))

    @classmethod
    def infer(cls, rows):
        """Construct a matrix from a singly-nested iterable, using the
        shallowest iterable's length to deduce the number of rows, and the
        nested iterables' lengths to deduce the number of columns

        Raises `ValueError` if the length of the nested iterables is
        inconsistent (i.e., a representation of an irregular matrix).
        """
        array = []

        rows = iter(rows)
        try:
            row = next(rows)
        except StopIteration:
            return cls.wrap(array, shape=Shape(0, 0))
        else:
            array.extend(row)

        m = 1
        n = len(array)

        for m, row in enumerate(rows, start=2):
            k = 0
            for k, val in enumerate(row, start=1):
                array.append(val)
            if n != k:
                raise ValueError(f"row {m} has length {k}, but precedent rows have length {n}")

        return cls.wrap(array, shape=Shape(m, n))

    @property
    def shape(self):
        return self._shape.copy()

    @property
    def nrows(self):
        return self._shape[0]

    @property
    def ncols(self):
        return self._shape[1]

    @property
    def size(self):
        shape = self._shape
        return shape[0] * shape[1]

    def lesser(self, other):
        return FrozenMatrix.wrap(
            list(checked_map(operator.__lt__, self, other)),
            shape=self.shape,
        )

    def lesser_equal(self, other):
        return FrozenMatrix.wrap(
            list(checked_map(operator.__le__, self, other)),
            shape=self.shape,
        )

    def equal(self, other):
        return FrozenMatrix.wrap(
            list(checked_map(operator.__eq__, self, other)),
            shape=self.shape,
        )

    def not_equal(self, other):
        return FrozenMatrix.wrap(
            list(checked_map(operator.__ne__, self, other)),
            shape=self.shape,
        )

    def greater(self, other):
        return FrozenMatrix.wrap(
            list(checked_map(operator.__gt__, self, other)),
            shape=self.shape,
        )

    def greater_equal(self, other):
        return FrozenMatrix.wrap(
            list(checked_map(operator.__ge__, self, other)),
            shape=self.shape,
        )

    def logical_and(self, other):
        return FrozenMatrix.wrap(
            list(checked_map(logical_and, self, other)),
            shape=self.shape,
        )

    def logical_or(self, other):
        return FrozenMatrix.wrap(
            list(checked_map(logical_or, self, other)),
            shape=self.shape,
        )

    def logical_not(self):
        return FrozenMatrix.wrap(
            list(map(logical_not, self)),
            shape=self.shape,
        )

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

    def n(self, by):
        return self._shape[by.value]

    def items(self, *, by=Rule.ROW, reverse=False):
        it = reversed if reverse else iter
        if by is Rule.ROW:
            yield from it(self._array)  # Fast path: the array is already in row-major order
            return
        nrows = self.nrows
        ncols = self.ncols
        row_indices = range(nrows)
        col_indices = range(ncols)
        for col_index in it(col_indices):
            for row_index in it(row_indices):
                yield self._array[row_index * ncols + col_index]


class Matrix(FrozenMatrix[T, M, N]):

    __slots__ = ()

    # TODO: __setitem__()

    def __deepcopy__(self, memo=None):
        """Return a deep copy of the matrix"""
        return self.__class__.wrap(
            array=copy.deepcopy(self._array, memo=memo),
            shape=copy.deepcopy(self._shape, memo=memo),
        )

    def __copy__(self):
        """Return a shallow copy of the matrix"""
        return self.__class__.wrap(
            array=copy.copy(self._array),
            shape=copy.copy(self._shape),
        )

    copy = __copy__

    def stack(self, other, *, by=Rule.ROW):
        """Return the matrix with ``other`` stacked by row or column"""
        if self is other:
            other = copy.copy(other)

        shape_l =  self.shape
        shape_r = other.shape

        dy = ~by

        if self.n(dy) != other.n(dy):
            raise ValueError(f"incompatible shapes {shape_l}, {shape_r} by {dy.handle}")

        nrows_l, ncols_l = shape_l
        _      , ncols_r = shape_r

        if by is Rule.COL and nrows_l > 1:
            values_l =  self.items()
            values_r = other.items()
            self._array = [
                value
                for _ in range(nrows_l)
                for value in itertools.chain(
                    itertools.islice(values_l, ncols_l),
                    itertools.islice(values_r, ncols_r),
                )
            ]
        else:
            self._array.extend(other)

        self._shape[by.value] += other.n(by)

        return self
