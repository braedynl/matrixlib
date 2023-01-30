import operator
from reprlib import recursive_repr
from typing import TypeVar

from . import FrozenMatrix
from .abc import MatrixLike
from .shapes import Shape
from .utilities import COL, ROW, Rule, checked_map

__all__ = [
    "MatrixView",
    "MatrixTransform",
    "MatrixTranspose",
    "MatrixRowFlip",
    "MatrixColFlip",
]

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

    def flip(self, *, by=Rule.ROW):
        return self._target.flip(by=by)

    def compare(self, other):
        return self._target.compare(other)

    def items(self, *, by=Rule.ROW, reverse=False):
        yield from self._target.items(by=by, reverse=reverse)

    def slices(self, *, by=Rule.ROW, reverse=False):
        yield from self._target.slices(by=by, reverse=reverse)


class MatrixTransform(MatrixView[T, M, N]):

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

    def __iter__(self):
        yield from super(MatrixView, self).__iter__()

    def __reversed__(self):
        yield from super(MatrixView, self).__reversed__()

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
            list(map(operator.__abs__, self)),
            shape=self.shape,
        )

    def __invert__(self):
        return FrozenMatrix.wrap(
            list(map(operator.__invert__, self)),
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
            list(checked_map(lambda x, y: not not (x and y), self, other)),
            shape=self.shape,
        )

    def logical_or(self, other):
        return FrozenMatrix.wrap(
            list(checked_map(lambda x, y: not not (x or y), self, other)),
            shape=self.shape,
        )

    def logical_not(self):
        return FrozenMatrix.wrap(
            list(map(lambda x: not x, self)),
            shape=self.shape,
        )

    def conjugate(self):
        return FrozenMatrix.wrap(
            list(map(lambda x: x.conjugate(), self)),
            shape=self.shape,
        )

    def transpose(self):
        return MatrixTranspose(self)

    def flip(self, *, by=Rule.ROW):
        MatrixTransform = (MatrixRowFlip, MatrixColFlip)[by.value]
        return MatrixTransform(self)

    def compare(self, other):
        return super(MatrixView, self).compare(other)

    def items(self, *, by=Rule.ROW, reverse=False):
        yield from super(MatrixView, self).items(by=by, reverse=reverse)

    def slices(self, *, by=Rule.ROW, reverse=False):
        yield from super(MatrixView, self).slices(by=by, reverse=reverse)

    def _permute_index_single(self, val_index):
        """Permute and return the given singular index as its "canonical" form

        This method is typically called post-resolution, and pre-retrieval (the
        return value of this function being the final index used in the
        retrieval process). At the base level, this function simply returns
        `val_index`.
        """
        return val_index

    def _permute_index_double(self, row_index, col_index):
        """Permute and return the given paired index as its "canonical" form

        This method is typically called post-resolution, and pre-retrieval (the
        return value of this function being the final index used in the
        retrieval process). At the base level, this function simply returns
        `row_index * self.ncols + col_index`.
        """
        return row_index * self.ncols + col_index


class MatrixTranspose(MatrixTransform[T, M, N]):

    __slots__ = ()

    @property
    def shape(self):
        return self._target.shape.reverse()

    @property
    def nrows(self):
        return self._target.ncols

    @property
    def ncols(self):
        return self._target.nrows

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


class MatrixRowFlip(MatrixTransform[T, M, N]):

    __slots__ = ()

    def flip(self, *, by=Rule.ROW):
        if by is Rule.ROW:
            return MatrixView(self._target)
        return MatrixColFlip(self)

    def _permute_index_single(self, val_index):
        row_index, col_index = divmod(val_index, self.ncols)
        return self._permute_index_double(
            row_index=row_index,
            col_index=col_index,
        )

    def _permute_index_double(self, row_index, col_index):
        return super()._permute_index_double(
            row_index=self.nrows - row_index - 1,
            col_index=col_index,
        )


class MatrixColFlip(MatrixTransform[T, M, N]):

    __slots__ = ()

    def flip(self, *, by=Rule.ROW):
        if by is Rule.COL:
            return MatrixView(self._target)
        return MatrixRowFlip(self)

    def _permute_index_single(self, val_index):
        row_index, col_index = divmod(val_index, self.ncols)
        return self._permute_index_double(
            row_index=row_index,
            col_index=col_index,
        )

    def _permute_index_double(self, row_index, col_index):
        return super()._permute_index_double(
            row_index=row_index,
            col_index=self.ncols - col_index - 1,
        )
