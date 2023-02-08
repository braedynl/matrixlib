import operator
from typing import TypeVar

from ..abc import MatrixLike
from ..builtins import FrozenMatrix
from ..rule import COL, ROW, Rule
from ..shapes.builtins import Shape
from ..utilities.checked_map import checked_map
from ..utilities.logical_operator import logical_and, logical_not, logical_or
from .abc import MatrixLikeView

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


class MatrixView(MatrixLikeView[T, M, N]):

    __slots__ = ()

    def __lt__(self, other):
        return self._target < other

    def __le__(self, other):
        return self._target <= other

    def __eq__(self, other):
        return self._target == other

    def __ne__(self, other):
        return self._target != other

    def __gt__(self, other):
        return self._target > other

    def __ge__(self, other):
        return self._target >= other

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

    def lesser(self, other):
        return self._target.lesser(other)

    def lesser_equal(self, other):
        return self._target.lesser_equal(other)

    def equal(self, other):
        return self._target.equal(other)

    def not_equal(self, other):
        return self._target.not_equal(other)

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

    def transpose(self):
        return self._target.transpose()

    def flip(self, *, by=Rule.ROW):
        return self._target.flip(by=by)

    def reverse(self):
        return self._target.reverse()

    def compare(self, other):
        return self._target.compare(other)

    def n(self, by):
        return self._target.n(by)

    def items(self, *, by=Rule.ROW, reverse=False):
        yield from self._target.items(by=by, reverse=reverse)

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


class MatrixTransform(MatrixLikeView[T, M, N]):
    """A type of `MatrixLikeView` whose indices are "permuted" before retrieval
    of items from the target matrix occurs

    A basic `MatrixTransform` does nothing on its own. This class simply
    provides overrides and some utilities for sub-classes to specify the
    permutation function.

    If you wish to check that a view is also one that permutes its indices,
    then performing an `isinstance()` check with `MatrixTransform` is a viable
    means of doing so (for built-in matrices, that is).
    """

    __slots__ = ()

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key

            if isinstance(row_key, slice):
                row_indices = self._resolve_matrix_slice(row_key, by=ROW)

                if isinstance(col_key, slice):
                    col_indices = self._resolve_matrix_slice(col_key, by=COL)
                    return FrozenMatrix.wrap(
                        [
                            self._target[
                                self._permute_matrix_index(
                                    row_index=row_index,
                                    col_index=col_index,
                                )
                            ]
                            for row_index in row_indices
                            for col_index in col_indices
                        ],
                        shape=Shape(len(row_indices), len(col_indices)),
                    )

                col_index = self._resolve_matrix_index(col_key, by=COL)
                return FrozenMatrix.wrap(
                    [
                        self._target[
                            self._permute_matrix_index(
                                row_index=row_index,
                                col_index=col_index,
                            )
                        ]
                        for row_index in row_indices
                    ],
                    shape=Shape(len(row_indices), 1),
                )

            row_index = self._resolve_matrix_index(row_key, by=ROW)

            if isinstance(col_key, slice):
                col_indices = self._resolve_matrix_slice(col_key, by=COL)
                return FrozenMatrix.wrap(
                    [
                        self._target[
                            self._permute_matrix_index(
                                row_index=row_index,
                                col_index=col_index,
                            )
                        ]
                        for col_index in col_indices
                    ],
                    shape=Shape(1, len(col_indices)),
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
            return FrozenMatrix.wrap(
                [
                    self._target[
                        self._permute_vector_index(
                            val_index=val_index,
                        )
                    ]
                    for val_index in val_indices
                ],
                shape=Shape(1, len(val_indices)),
            )

        val_index = self._resolve_vector_index(key)
        return self._target[
            self._permute_vector_index(
                val_index=val_index,
            )
        ]

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

    @property
    def shape(self):
        return Shape(self.nrows, self.ncols)

    @property
    def nrows(self):
        return self._target.nrows

    @property
    def ncols(self):
        return self._target.ncols

    @property
    def size(self):
        return self._target.size

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

    # If you're able to provide a view on the target, rather than on the view
    # itself, you should. View "stacking" can cause significant slow downs.

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
        """Return a given `val_index` as its permuted form

        By default, this method simply returns `val_index`.
        """
        return val_index

    def _permute_matrix_index(self, row_index, col_index):
        """Return a given `row_index`, `col_index` pair as its permuted form

        By default, this method simply returns
        `row_index * self.ncols + col_index`.
        """
        return row_index * self.ncols + col_index


class MatrixTranspose(MatrixTransform[T, M, N]):
    """A type of `MatrixTransform` that transposes its indices

    Independent construction of a `MatrixTranspose` is not recommended.
    Instead, use the matrix's `transpose()` method (if provided).
    """

    __slots__ = ()

    @property
    def nrows(self):
        return self._target.ncols

    @property
    def ncols(self):
        return self._target.nrows

    def transpose(self):
        return MatrixView(self._target)  # Fast path: transpose of transpose -> original matrix

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
    """A type of `MatrixTransform` that reverses its row indices

    Independent construction of a `MatrixRowFlip` is not recommended. Instead,
    use the matrix's `flip()` method (if provided).
    """

    __slots__ = ()

    def flip(self, *, by=Rule.ROW):
        if by is Rule.ROW:
            return MatrixView(self._target)  # Fast path: row-flip of row-flip -> original matrix
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
    """A type of `MatrixTransform` that reverses its column indices

    Independent construction of a `MatrixColFlip` is not recommended. Instead,
    use the matrix's `flip()` method (if provided).
    """

    __slots__ = ()

    def flip(self, *, by=Rule.ROW):
        if by is Rule.COL:
            return MatrixView(self._target)  # Fast path: col-flip of col-flip -> original matrix
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
    """A type of `MatrixTransform` that reverses its indices

    Independent construction of a `MatrixReverse` is not recommended. Instead,
    use the matrix's `reverse()` method (if provided).
    """

    __slots__ = ()

    def reverse(self):
        return MatrixView(self._target)  # Fast path: reverse of reverse -> original matrix

    def _permute_vector_index(self, val_index):
        return self.size - val_index - 1

    def _permute_matrix_index(self, row_index, col_index):
        return self._permute_vector_index(
            val_index=super()._permute_matrix_index(
                row_index=row_index,
                col_index=col_index,
            ),
        )
