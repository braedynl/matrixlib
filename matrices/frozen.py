import itertools
import operator
from reprlib import recursive_repr
from typing import TypeVar

from .abstract import MatrixLike, matrix_map
from .shapes import Shape, ShapeView
from .utilities import COL, ROW, Rule

DTypeT_co = TypeVar("DTypeT_co", covariant=True)
NRowsT_co = TypeVar("NRowsT_co", covariant=True, bound=int)
NColsT_co = TypeVar("NColsT_co", covariant=True, bound=int)


class FrozenMatrix(MatrixLike[DTypeT_co, NRowsT_co, NColsT_co]):

    __slots__ = ("_array", "_shape")

    def __init__(self, data=None, nrows=None, ncols=None) -> None:
        self._array = [] if data is None else list(data)
        self._shape = Shape.from_size(
            size=len(self._array),
            nrows=nrows,
            ncols=ncols,
        )

    @recursive_repr("...")
    def __repr__(self):
        """Return a canonical representation of the matrix"""
        array = self._array
        shape = self._shape
        return f"{self.__class__.__name__}({array!r}, nrows={shape.nrows!r}, ncols={shape.ncols!r})"

    def __getitem__(self, key):
        array = self._array
        shape = self._shape

        if isinstance(key, tuple):

            def getitems(keys, nrows, ncols):
                n = shape.ncols
                return self.__class__.wrap(
                    [array[i * n + j] for i, j in keys],
                    shape=Shape(nrows, ncols),
                )

            rowkey, colkey = key

            if isinstance(rowkey, slice):
                ix = shape.resolve_slice(rowkey, by=ROW)

                if isinstance(colkey, slice):
                    jx = shape.resolve_slice(colkey, by=COL)
                    result = getitems(
                        itertools.product(ix, jx),
                        nrows=len(ix),
                        ncols=len(jx),
                    )

                else:
                    j = shape.resolve_index(colkey, by=COL)
                    result = getitems(
                        zip(ix, itertools.repeat(j)),
                        nrows=len(ix),
                        ncols=1,
                    )

            else:
                i = shape.resolve_index(rowkey, by=ROW)

                if isinstance(colkey, slice):
                    jx = shape.resolve_slice(colkey, by=COL)
                    result = getitems(
                        zip(itertools.repeat(i), jx),
                        nrows=1,
                        ncols=len(jx),
                    )

                else:
                    j = shape.resolve_index(colkey, by=COL)
                    result = array[i * shape.ncols + j]

            return result

        n = shape.nrows * shape.ncols

        if isinstance(key, slice):

            def getitems(keys, nrows, ncols):
                return self.__class__.wrap(
                    [array[i] for i in keys],
                    shape=Shape(nrows, ncols),
                )

            ix = range(*key.indices(n))
            result = getitems(
                ix,
                nrows=1,
                ncols=len(ix),
            )

            return result

        try:
            result = array[key]
        except IndexError:
            raise IndexError(f"there are {n} items but index is {key}") from None
        else:
            return result

    def __iter__(self):
        yield from self._array

    def __reversed__(self):
        yield from reversed(self._array)

    def __contains__(self, value):
        return value in self._array

    def __add__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.add, self, other),
                *self._shape,
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.sub, self, other),
                *self._shape,
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.mul, self, other),
                *self._shape,
            )
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.truediv, self, other),
                *self._shape,
            )
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.floordiv, self, other),
                *self._shape,
            )
        return NotImplemented

    def __mod__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.mod, self, other),
                *self._shape,
            )
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.pow, self, other),
                *self._shape,
            )
        return NotImplemented

    def __and__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(logical_and, self, other),
                *self._shape,
            )
        return NotImplemented

    def __or__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(logical_or, self, other),
                *self._shape,
            )
        return NotImplemented

    def __xor__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(logical_xor, self, other),
                *self._shape,
            )
        return NotImplemented

    def __neg__(self):
        return self.__class__(
            matrix_map(operator.neg, self),
            *self._shape,
        )

    def __pos__(self):
        return self.__class__(
            matrix_map(operator.pos, self),
            *self._shape,
        )

    def __abs__(self):
        return self.__class__(
            matrix_map(operator.abs, self),
            *self._shape,
        )

    def __invert__(self):
        return self.__class__(
            matrix_map(logical_not, self),
            *self._shape,
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

    @property
    def shape(self):
        return ShapeView(self._shape)

    def eq(self, other):
        return self.__class__(
            matrix_map(operator.eq, self, other),
            *self._shape,
        )

    def ne(self, other):
        return self.__class__(
            matrix_map(operator.ne, self, other),
            *self._shape,
        )

    def lt(self, other):
        return self.__class__(
            matrix_map(operator.lt, self, other),
            *self._shape,
        )

    def le(self, other):
        return self.__class__(
            matrix_map(operator.le, self, other),
            *self._shape,
        )

    def gt(self, other):
        return self.__class__(
            matrix_map(operator.gt, self, other),
            *self._shape,
        )

    def ge(self, other):
        return self.__class__(
            matrix_map(operator.ge, self, other),
            *self._shape,
        )

    def conjugate(self):
        return self.__class__(
            matrix_map(conjugate, self),
            *self._shape,
        )

    def slices(self, *, by=Rule.ROW):
        array = self._array
        shape = self._shape
        subshape = shape.subshape(by=by)
        for i in range(shape[by.value]):
            temp = array[shape.slice(i, by=by)]
            yield self.__class__.wrap(temp, shape=subshape.copy())


def logical_and(a, b, /): return not not (a and b)
def logical_or(a, b, /): return not not (a or b)
def logical_xor(a, b, /): return (not not a) is not (not not b)
def logical_not(a, /): return not a

def conjugate(x, /): return x.conjugate()
