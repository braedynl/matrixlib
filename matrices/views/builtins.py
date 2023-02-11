from reprlib import recursive_repr
from typing import TypeVar

from ..builtins import FrozenMatrix
from ..rule import COL, ROW, Rule
from ..utilities.checked_map import checked_map
from ..utilities.operator import (equal, logical_and, logical_not, logical_or,
                                  not_equal)
from .abc import MatrixViewLike

__all__ = [
    "MatrixView",
    "MatrixTransform",
    "MatrixTranspose",
    "MatrixRowFlip",
    "MatrixColFlip",
    "MatrixReverse",
]

T = TypeVar("T")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


class MatrixView(MatrixViewLike[T, M, N]):

    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    @recursive_repr("...")
    def __repr__(self):
        return f"{self.__class__.__name__}(target={self._target!r})"

    def __len__(self):
        return len(self._target)

    def __getitem__(self, key):
        return self._target[key]

    def __iter__(self):
        yield from iter(self._target)

    def __reversed__(self):
        yield from reversed(self._target)

    def __contains__(self, value):
        return value in self._target

    @property
    def array(self):
        return self._target.array

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

    def logical_and(self, other):
        return self._target.logical_and(other)

    def logical_or(self, other):
        return self._target.logical_or(other)

    def logical_not(self):
        return self._target.logical_not()

    def transpose(self):
        return self._target.transpose()

    def flip(self, *, by=Rule.ROW):
        return self._target.flip(by=by)

    def reverse(self):
        return self._target.reverse()

    def n(self, by):
        return self._target.n(by)

    def values(self, *, by=Rule.ROW, reverse=False):
        yield from self._target.values(by=by, reverse=reverse)

    def slices(self, *, by=Rule.ROW, reverse=False):
        yield from self._target.slices(by=by, reverse=reverse)

    def _resolve_vector_index(self, key):
        return self._target._resolve_vector_index(key)

    def _resolve_matrix_index(self, key, *, by=Rule.ROW):
        return self._target._resolve_matrix_index(key, by=by)

    def _resolve_vector_slice(self, key):
        return self._target._resolve_vector_slice(key)

    def _resolve_matrix_slice(self, key, *, by=Rule.ROW):
        return self._target._resolve_matrix_slice(key, by=by)


class MatrixTransform(MatrixViewLike[T, M, N]):

    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    @recursive_repr("...")
    def __repr__(self):
        return f"{self.__class__.__name__}(target={self._target!r})"

    def __getitem__(self, key, *, cls=FrozenMatrix):
        if isinstance(key, tuple):
            row_key, col_key = key

            if isinstance(row_key, slice):
                row_indices = self._resolve_matrix_slice(row_key, by=ROW)

                if isinstance(col_key, slice):
                    col_indices = self._resolve_matrix_slice(col_key, by=COL)
                    return cls.wrap(
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
                return cls.wrap(
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
                return cls.wrap(
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
            return cls.wrap(
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

    @property
    def array(self):
        return list(self.values())

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
        return FrozenMatrix.wrap(
            array=list(checked_map(equal, self, other)),
            shape=self.shape,
        )

    def not_equal(self, other):
        return FrozenMatrix.wrap(
            array=list(checked_map(not_equal, self, other)),
            shape=self.shape,
        )

    def logical_and(self, other):
        return FrozenMatrix.wrap(
            array=list(checked_map(logical_and, self, other)),
            shape=self.shape,
        )

    def logical_or(self, other):
        return FrozenMatrix.wrap(
            array=list(checked_map(logical_or, self, other)),
            shape=self.shape,
        )

    def logical_not(self):
        return FrozenMatrix.wrap(
            array=list(map(logical_not, self)),
            shape=self.shape,
        )

    def transpose(self):
        return MatrixTranspose(self)

    def flip(self, *, by=Rule.ROW):
        MatrixTransform = (MatrixRowFlip, MatrixColFlip)[by.value]
        return MatrixTransform(self)

    def reverse(self):
        return MatrixReverse(self)

    def n(self, by):
        return self._target.n(by)

    def _permute_vector_index(self, val_index):
        return val_index

    def _permute_matrix_index(self, row_index, col_index):
        return row_index * self.ncols + col_index


class MatrixTranspose(MatrixTransform[T, M, N]):

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
        return MatrixView(self._target)

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


class MatrixRowFlip(MatrixTransform[T, M, N]):

    __slots__ = ()

    def flip(self, *, by=Rule.ROW):
        if by is Rule.ROW:
            return MatrixView(self._target)
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


class MatrixColFlip(MatrixTransform[T, M, N]):

    __slots__ = ()

    def flip(self, *, by=Rule.ROW):
        if by is Rule.COL:
            return MatrixView(self._target)
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


class MatrixReverse(MatrixTransform[T, M, N]):

    __slots__ = ()

    def reverse(self):
        return MatrixView(self._target)

    def _permute_vector_index(self, val_index):
        return self.size - val_index - 1

    def _permute_matrix_index(self, row_index, col_index):
        return self._permute_vector_index(
            val_index=super()._permute_matrix_index(
                row_index=row_index,
                col_index=col_index,
            ),
        )
