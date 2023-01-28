import operator
from reprlib import recursive_repr
from typing import TypeVar

from . import FrozenMatrix
from .abc import MatrixLike
from .shapes import Shape
from .utilities import COL, ROW, checked_map

__all__ = ["MatrixView", "MatrixTranspose"]

T = TypeVar("T")
M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


class MatrixView(MatrixLike[T, M, N]):

    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    @recursive_repr("...")
    def __repr__(self):
        """Return a canonical representation of the view"""
        return f"{self.__class__.__name__}(target={self._target!r})"

    def __getitem__(self, key):
        return self._target[key]

    def __iter__(self):
        yield from iter(self._target)

    def __reversed__(self):
        yield from reversed(self._target)

    def __contains__(self, value):
        return value in self._target

    def __add__(self, other):
        return self._target + other

    def __sub__(self, other):
        return self._target - other

    def __mul__(self, other):
        return self._target * other

    def __truediv__(self, other):
        return self._target / other

    def __floordiv__(self, other):
        return self._target // other

    def __mod__(self, other):
        return self._target % other

    def __divmod__(self, other):
        return divmod(self._target, other)

    def __pow__(self, other):
        return self._target ** other

    def __lshift__(self, other):
        return self._target << other

    def __rshift__(self, other):
        return self._target >> other

    def __and__(self, other):
        return self._target & other

    def __xor__(self, other):
        return self._target ^ other

    def __or__(self, other):
        return self._target | other

    def __matmul__(self, other):
        return self._target @ other

    def __neg__(self):
        return -self._target

    def __pos__(self):
        return +self._target

    def __abs__(self):
        return abs(self._target)

    def __invert__(self):
        return ~self._target

    @property
    def shape(self):
        return self._target.shape

    @property
    def nrows(self):
        return self._target.nrows

    @property
    def ncols(self):
        return self._target.ncols

    @property
    def size(self):
        return self._target.size

    def equal(self, other):
        return self._target.equal(other)

    def not_equal(self, other):
        return self._target.not_equal(other)

    def lesser(self, other):
        return self._target.lesser(other)

    def lesser_equal(self, other):
        return self._target.lesser_equal(other)

    def greater(self, other):
        return self._target.greater(other)

    def greater_equal(self, other):
        return self._target.greater_equal(other)

    def logical_and(self, other):
        return self._target.logical_and(other)

    def logical_or(self, other):
        return self._target.logical_or(other)

    def logical_not(self):
        return self._target.logical_not()

    def conjugate(self):
        return self._target.conjugate()

    def transpose(self):
        return self._target.transpose()


class MatrixTranspose(MatrixView[T, M, N]):

    __slots__ = ()

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key

            if isinstance(row_key, slice):
                row_indices = self._resolve_slice(row_key, by=ROW)

                if isinstance(col_key, slice):
                    col_indices = self._resolve_slice(col_key, by=COL)
                    return FrozenMatrix.wrap(
                        [
                            self._target[
                                self._permute_index_double(
                                    row_index=row_index,
                                    col_index=col_index,
                                )
                            ]
                            for row_index in row_indices
                            for col_index in col_indices
                        ],
                        shape=Shape(len(row_indices), len(col_indices)),
                    )

                col_index = self._resolve_index(col_key, by=COL)
                return FrozenMatrix.wrap(
                    [
                        self._target[
                            self._permute_index_double(
                                row_index=row_index,
                                col_index=col_index,
                            )
                        ]
                        for row_index in row_indices
                    ],
                    shape=Shape(len(row_indices), 1),
                )

            row_index = self._resolve_index(row_key, by=ROW)

            if isinstance(col_key, slice):
                col_indices = self._resolve_slice(col_key, by=COL)
                return FrozenMatrix.wrap(
                    [
                        self._target[
                            self._permute_index_double(
                                row_index=row_index,
                                col_index=col_index,
                            )
                        ]
                        for col_index in col_indices
                    ],
                    shape=Shape(1, len(col_indices)),
                )

            col_index = self._resolve_index(col_key, by=COL)
            return self._target[
                self._permute_index_double(
                    row_index=row_index,
                    col_index=col_index,
                )
            ]

        if isinstance(key, slice):
            val_indices = self._resolve_slice(key)
            return FrozenMatrix.wrap(
                [
                    self._target[
                        self._permute_index_single(
                            val_index=val_index,
                        )
                    ]
                    for val_index in val_indices
                ],
                shape=Shape(1, len(val_indices)),
            )

        val_index = self._resolve_index(key)
        return self._target[
            self._permute_index_single(
                val_index=val_index,
            )
        ]

    def __iter__(self, *, iter=iter):
        nrows = self.nrows
        ncols = self.ncols
        row_indices = range(nrows)
        col_indices = range(ncols)
        for row_index in iter(row_indices):
            for col_index in iter(col_indices):
                yield self._target[col_index * nrows + row_index]

    def __reversed__(self):
        yield from self.__iter__(iter=reversed)

    @property
    def shape(self):
        return self._target.shape.reverse()

    @property
    def nrows(self):
        return self._target.ncols

    @property
    def ncols(self):
        return self._target.nrows

    # The transpose of a transpose nets no change to the matrix. Thus, we
    # simply return a view on the target.

    def transpose(self):
        return MatrixView(self._target)

    def _permute_index_single(self, val_index):
        row_index, col_index = divmod(val_index, self.ncols)
        return self._permute_index_double(
            row_index=row_index,
            col_index=col_index,
        )

    def _permute_index_double(self, row_index, col_index):
        return col_index * self.nrows + row_index
