"""Dynamic, vectorized sequence types in ≤ 2 dimensions.

Contains the `Matrix` definition, along with its related utilities and
variants.

Author: Braedyn L
Version: 0.3.0
Documentation: https://github.com/braedynl/matrices-py/wiki
"""

import functools
import operator
from reprlib import recursive_repr
from typing import TypeVar

from .abc import *
from .shapes import *
from .utilities import *

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
                row_indices = self._resolve_slice(row_key, by=ROW)

                if isinstance(col_key, slice):
                    col_indices = self._resolve_slice(col_key, by=COL)
                    return self.__class__.wrap(
                        [
                            self._array[row_index * ncols + col_index]
                            for row_index in row_indices
                            for col_index in col_indices
                        ],
                        shape=Shape(len(row_indices), len(col_indices)),
                    )

                col_index = self._resolve_index(col_key, by=COL)
                return self.__class__.wrap(
                    [
                        self._array[row_index * ncols + col_index]
                        for row_index in row_indices
                    ],
                    shape=Shape(len(row_indices), 1),
                )

            row_index = self._resolve_index(row_key, by=ROW)

            if isinstance(col_key, slice):
                col_indices = self._resolve_slice(col_key, by=COL)
                return self.__class__.wrap(
                    [
                        self._array[row_index * ncols + col_index]
                        for col_index in col_indices
                    ],
                    shape=Shape(1, len(col_indices)),
                )

            col_index = self._resolve_index(col_key, by=COL)
            return self._array[row_index * ncols + col_index]

        if isinstance(key, slice):
            val_indices = self._resolve_slice(key)
            return self.__class__.wrap(
                [
                    self._array[val_index]
                    for val_index in val_indices
                ],
                shape=Shape(1, len(val_indices)),
            )

        val_index = self._resolve_index(key)
        return self._array[val_index]

    def __contains__(self, value):
        return value in self._array

    def __add__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__.wrap(
                list(checked_map(operator.__add__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__.wrap(
                list(checked_map(operator.__sub__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__.wrap(
                list(checked_map(operator.__mul__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__.wrap(
                list(checked_map(operator.__truediv__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__.wrap(
                list(checked_map(operator.__floordiv__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __mod__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__.wrap(
                list(checked_map(operator.__mod__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __divmod__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__.wrap(
                list(checked_map(divmod, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__.wrap(
                list(checked_map(operator.__pow__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __lshift__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__.wrap(
                list(checked_map(operator.__lshift__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __rshift__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__.wrap(
                list(checked_map(operator.__rshift__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __and__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__.wrap(
                list(checked_map(operator.__and__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __xor__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__.wrap(
                list(checked_map(operator.__xor__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __or__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__.wrap(
                list(checked_map(operator.__or__, self, other)),
                shape=self.shape,
            )
        return NotImplemented

    def __matmul__(self, other):
        if not isinstance(other, MatrixLike):
            return NotImplemented

        (m, n), (p, q) = self.shape, other.shape

        if n != p:
            raise ValueError
        if not n:
            raise ValueError

        ix = range(m)
        jx = range(q)
        kx = range(n)

        return self.__class__.wrap(
            [
                functools.reduce(
                    operator.add,
                    map(lambda k: self[i * n + k] * other[k * q + j], kx),
                )
                for i in ix
                for j in jx
            ],
            shape=Shape(m, q),
        )

    def __neg__(self):
        return self.__class__.wrap(
            list(map(operator.__neg__, self)),
            shape=self.shape,
        )

    def __pos__(self):
        return self.__class__.wrap(
            list(map(operator.__pos__, self)),
            shape=self.shape,
        )

    def __abs__(self):
        return self.__class__.wrap(
            list(map(operator.__abs__, self)),
            shape=self.shape,
        )

    def __invert__(self):
        return self.__class__.wrap(
            list(map(operator.__invert__, self)),
            shape=self.shape,
        )

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
        return cls.wrap([value] * (nrows * ncols), shape=shape)

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

    def ndims(self, by):
        return self._shape[by.value]

    def equal(self, other):
        return self.__class__.wrap(
            list(checked_map(operator.__eq__, self, other)),
            shape=self.shape,
        )

    def not_equal(self, other):
        return self.__class__.wrap(
            list(checked_map(operator.__ne__, self, other)),
            shape=self.shape,
        )

    def lesser(self, other):
        return self.__class__.wrap(
            list(checked_map(operator.__lt__, self, other)),
            shape=self.shape,
        )

    def lesser_equal(self, other):
        return self.__class__.wrap(
            list(checked_map(operator.__le__, self, other)),
            shape=self.shape,
        )

    def greater(self, other):
        return self.__class__.wrap(
            list(checked_map(operator.__gt__, self, other)),
            shape=self.shape,
        )

    def greater_equal(self, other):
        return self.__class__.wrap(
            list(checked_map(operator.__ge__, self, other)),
            shape=self.shape,
        )

    def logical_and(self, other):
        return self.__class__.wrap(
            list(checked_map(lambda x, y: not not (x and y), self, other)),
            shape=self.shape,
        )

    def logical_or(self, other):
        return self.__class__.wrap(
            list(checked_map(lambda x, y: not not (x or y), self, other)),
            shape=self.shape,
        )

    def logical_not(self):
        return self.__class__.wrap(
            list(map(lambda x: not x, self)),
            shape=self.shape,
        )

    def conjugate(self):
        return self.__class__.wrap(
            list(map(lambda x: x.conjugate(), self)),
            shape=self.shape,
        )

    def transpose(self):
        from .views import MatrixTranspose
        return MatrixTranspose(self)

    def flip(self, *, by=Rule.ROW):
        from .views import MatrixColFlip, MatrixRowFlip
        MatrixTransform = (MatrixRowFlip, MatrixColFlip)[by.value]
        return MatrixTransform(self)

    def reverse(self):
        from .views import MatrixReverse
        return MatrixReverse(self)

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
