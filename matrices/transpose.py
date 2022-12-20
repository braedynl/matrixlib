import itertools
import operator
from reprlib import recursive_repr
from typing import TypeVar

from .abstract import MatrixLike, matmap, matmul
from .frozen import FrozenMatrix
from .shapes import Shape
from .utilities import COL, ROW, Rule

T = TypeVar("T")
M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


class MatrixTranspose(MatrixLike[T, M, N]):

    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    @recursive_repr("...")
    def __repr__(self):
        return f"{self.__class__.__name__}(target={self._target!r})"

    def __getitem__(self, key):
        target = self._target
        shape  = self.shape

        if isinstance(key, tuple):
            rowkey, colkey = key

            if isinstance(rowkey, slice):
                ix = shape.resolve_slice(rowkey, by=ROW)

                if isinstance(colkey, slice):
                    jx = shape.resolve_slice(colkey, by=COL)
                    n  = shape.nrows
                    result = FrozenMatrix.wrap(
                        [target[j * n + i] for i, j in itertools.product(ix, jx)],
                        shape=Shape(len(ix), len(jx)),
                    )

                else:
                    j = shape.resolve_index(colkey, by=COL)
                    n = shape.nrows
                    result = FrozenMatrix.wrap(
                        [target[j * n + i] for i in ix],
                        shape=Shape(len(ix), 1),
                    )

            else:
                i = shape.resolve_index(rowkey, by=ROW)

                if isinstance(colkey, slice):
                    jx = shape.resolve_slice(colkey, by=COL)
                    n  = shape.nrows
                    result = FrozenMatrix.wrap(
                        [target[j * n + i] for j in jx],
                        shape=Shape(1, len(jx)),
                    )

                else:
                    j = shape.resolve_index(colkey, by=COL)
                    n = shape.nrows
                    result = target[j * n + i]

        elif isinstance(key, slice):
            ix = range(*key.indices(shape.nrows * shape.ncols))

            result = FrozenMatrix.wrap(
                [target[self.permute_index(i)] for i in ix],
                shape=Shape(1, len(ix)),
            )

        else:
            result = target[self.permute_index(key)]

        return result

    def __iter__(self, *, iter=iter):
        target = self._target
        nrows, ncols = target.shape
        ix = range(nrows)
        jx = range(ncols)
        for j in iter(jx):
            for i in iter(ix):
                yield target[i * ncols + j]

    def __reversed__(self):
        yield from self.__iter__(iter=reversed)

    def __contains__(self, value):
        return value in self._target

    def __add__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix(
                matmap(operator.__add__, self, other),
                shape=(self.nrows, self.ncols),
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix(
                matmap(operator.__sub__, self, other),
                shape=(self.nrows, self.ncols),
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix(
                matmap(operator.__mul__, self, other),
                shape=(self.nrows, self.ncols),
            )
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix(
                matmap(operator.__truediv__, self, other),
                shape=(self.nrows, self.ncols),
            )
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix(
                matmap(operator.__floordiv__, self, other),
                shape=(self.nrows, self.ncols),
            )
        return NotImplemented

    def __mod__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix(
                matmap(operator.__mod__, self, other),
                shape=(self.nrows, self.ncols),
            )
        return NotImplemented

    def __divmod__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix(
                matmap(divmod, self, other),
                shape=(self.nrows, self.ncols),
            )
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix(
                matmap(operator.__pow__, self, other),
                shape=(self.nrows, self.ncols),
            )
        return NotImplemented

    def __lshift__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix(
                matmap(operator.__lshift__, self, other),
                shape=(self.nrows, self.ncols),
            )
        return NotImplemented

    def __rshift__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix(
                matmap(operator.__rshift__, self, other),
                shape=(self.nrows, self.ncols),
            )
        return NotImplemented

    def __and__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix(
                matmap(operator.__and__, self, other),
                shape=(self.nrows, self.ncols),
            )
        return NotImplemented

    def __xor__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix(
                matmap(operator.__xor__, self, other),
                shape=(self.nrows, self.ncols),
            )
        return NotImplemented

    def __or__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix(
                matmap(operator.__or__, self, other),
                shape=(self.nrows, self.ncols),
            )
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, MatrixLike):
            return FrozenMatrix(
                matmul(self, other),
                shape=(self.nrows, other.ncols),
            )
        return NotImplemented

    def __neg__(self):
        return FrozenMatrix(
            matmap(operator.__neg__, self),
            shape=(self.nrows, self.ncols),
        )

    def __pos__(self):
        return FrozenMatrix(
            matmap(operator.__pos__, self),
            shape=(self.nrows, self.ncols),
        )

    def __abs__(self):
        return FrozenMatrix(
            matmap(operator.__abs__, self),
            shape=(self.nrows, self.ncols),
        )

    def __invert__(self):
        return FrozenMatrix(
            matmap(operator.__invert__, self),
            shape=(self.nrows, self.ncols),
        )

    @property
    def shape(self):
        nrows = self._target.nrows
        ncols = self._target.ncols
        return Shape(ncols, nrows)

    @property
    def nrows(self):
        return self._target.ncols

    @property
    def ncols(self):
        return self._target.nrows

    @property
    def size(self):
        return self._target.size

    @property
    def target(self):
        return self._target

    def equal(self, other):
        return FrozenMatrix(
            matmap(operator.__eq__, self, other),
            shape=(self.nrows, self.ncols),
        )

    def not_equal(self, other):
        return FrozenMatrix(
            matmap(operator.__ne__, self, other),
            shape=(self.nrows, self.ncols),
        )

    def lesser(self, other):
        return FrozenMatrix(
            matmap(operator.__lt__, self, other),
            shape=(self.nrows, self.ncols),
        )

    def lesser_equal(self, other):
        return FrozenMatrix(
            matmap(operator.__le__, self, other),
            shape=(self.nrows, self.ncols),
        )

    def greater(self, other):
        return FrozenMatrix(
            matmap(operator.__gt__, self, other),
            shape=(self.nrows, self.ncols),
        )

    def greater_equal(self, other):
        return FrozenMatrix(
            matmap(operator.__ge__, self, other),
            shape=(self.nrows, self.ncols),
        )

    def logical_and(self, other):
        return FrozenMatrix(
            matmap(lambda a, b: not not (a and b), self, other),
            shape=(self.nrows, self.ncols),
        )

    def logical_or(self, other):
        return FrozenMatrix(
            matmap(lambda a, b: not not (a or b), self, other),
            shape=(self.nrows, self.ncols),
        )

    def logical_not(self):
        return FrozenMatrix(
            matmap(lambda a: not a, self),
            shape=(self.nrows, self.ncols),
        )

    def conjugate(self):
        return FrozenMatrix(
            matmap(lambda a: a.conjugate(), self),
            shape=(self.nrows, self.ncols),
        )

    def slices(self, *, by=Rule.ROW):  # TODO
        ...

    def transpose(self):
        return self._target

    def permute_index(self, key):
        """Return an index `key` as its transposed equivalent with respect to
        the target matrix

        Raises `IndexError` if the key is out of range.
        """
        nrows = self._target.nrows
        ncols = self._target.ncols
        n = nrows * ncols
        i = operator.index(key)
        i += n * (i < 0)
        if i < 0 or i >= n:
            raise IndexError(f"there are {n} items but index is {key}")
        j = n - 1
        return i if i == j else (i * ncols) % j
