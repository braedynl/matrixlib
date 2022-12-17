import itertools
import operator
from reprlib import recursive_repr
from typing import TypeVar

from .abstract import MatrixLike, matrix_map, matrix_multiply
from .shapes import Shape, ShapeView
from .utilities import COL, ROW, Rule

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class FrozenMatrix(MatrixLike[T_co, M_co, N_co]):

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
                matrix_map(operator.__add__, self, other),
                self.nrows,
                self.ncols,
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.__sub__, self, other),
                self.nrows,
                self.ncols,
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.__mul__, self, other),
                self.nrows,
                self.ncols,
            )
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.__truediv__, self, other),
                self.nrows,
                self.ncols,
            )
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.__floordiv__, self, other),
                self.nrows,
                self.ncols,
            )
        return NotImplemented

    def __mod__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.__mod__, self, other),
                self.nrows,
                self.ncols,
            )
        return NotImplemented

    def __divmod__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(divmod, self, other),
                self.nrows,
                self.ncols,
            )
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.__pow__, self, other),
                self.nrows,
                self.ncols,
            )
        return NotImplemented

    def __lshift__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.__lshift__, self, other),
                self.nrows,
                self.ncols,
            )
        return NotImplemented

    def __rshift__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.__rshift__, self, other),
                self.nrows,
                self.ncols,
            )
        return NotImplemented

    def __and__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.__and__, self, other),
                self.nrows,
                self.ncols,
            )
        return NotImplemented

    def __xor__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.__xor__, self, other),
                self.nrows,
                self.ncols,
            )
        return NotImplemented

    def __or__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_map(operator.__or__, self, other),
                self.nrows,
                self.ncols,
            )
        return NotImplemented

    def __neg__(self):
        return self.__class__(
            matrix_map(operator.__neg__, self),
            self.nrows,
            self.ncols,
        )

    def __pos__(self):
        return self.__class__(
            matrix_map(operator.__pos__, self),
            self.nrows,
            self.ncols,
        )

    def __abs__(self):
        return self.__class__(
            matrix_map(operator.__abs__, self),
            self.nrows,
            self.ncols,
        )

    def __invert__(self):
        return self.__class__(
            matrix_map(operator.__invert__, self),
            self.nrows,
            self.ncols,
        )

    def __matmul__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matrix_multiply(self, other),
                 self.nrows,
                other.ncols,
            )
        return NotImplemented

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
        return ShapeView(self._shape)

    @property
    def nrows(self):
        return self._shape.nrows

    @property
    def ncols(self):
        return self._shape.ncols

    @property
    def size(self):
        nrows, ncols = self._shape
        return nrows * ncols

    def eq(self, other):
        return self.__class__(
            matrix_map(operator.__eq__, self, other),
            self.nrows,
            self.ncols,
        )

    def ne(self, other):
        return self.__class__(
            matrix_map(operator.__ne__, self, other),
            self.nrows,
            self.ncols,
        )

    def lt(self, other):
        return self.__class__(
            matrix_map(operator.__lt__, self, other),
            self.nrows,
            self.ncols,
        )

    def le(self, other):
        return self.__class__(
            matrix_map(operator.__le__, self, other),
            self.nrows,
            self.ncols,
        )

    def gt(self, other):
        return self.__class__(
            matrix_map(operator.__gt__, self, other),
            self.nrows,
            self.ncols,
        )

    def ge(self, other):
        return self.__class__(
            matrix_map(operator.__ge__, self, other),
            self.nrows,
            self.ncols,
        )

    def conjugate(self):
        return self.__class__(
            matrix_map(lambda x: x.conjugate(), self),
            self.nrows,
            self.ncols,
        )

    def slices(self, *, by=Rule.ROW):
        array = self._array
        shape = self._shape
        subshape = shape.subshape(by=by)
        for i in range(shape[by.value]):
            temp = array[shape.slice(i, by=by)]
            yield self.__class__.wrap(temp, shape=subshape.copy())
