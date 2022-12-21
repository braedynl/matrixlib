import operator
from reprlib import recursive_repr
from typing import TypeVar

from .abstract import MatrixLike, matmap, matmul
from .shapes import Shape, ShapeLike, ShapeView
from .utilities import COL, ROW

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class FrozenMatrix(MatrixLike[T_co, M_co, N_co]):

    __slots__ = ("_array", "_shape")

    def __init__(self, array=None, shape=None):
        array = [] if array is None else list(array)

        match shape:
            case None | (None, None):
                nrows, ncols = (1, len(array))
            case (None, ncols):
                n = len(array)
                nrows, loss = divmod(n, ncols) if ncols else (0, n)
                if loss:
                    raise ValueError(f"cannot interpret array of size {n} as shape M × {ncols}")
            case (nrows, None):
                n = len(array)
                ncols, loss = divmod(n, nrows) if nrows else (0, n)
                if loss:
                    raise ValueError(f"cannot interpret array of size {n} as shape {nrows} × N")
            case (nrows, ncols) | ShapeLike(nrows, ncols):
                n = len(array)
                if n != nrows * ncols:
                    raise ValueError(f"cannot interpret array of size {n} as shape {nrows} × {ncols}")
            case _:
                raise ValueError("shape must contain two dimensions")

        shape = Shape(nrows, ncols)

        self._array = array
        self._shape = shape

    @recursive_repr("...")
    def __repr__(self):
        """Return a canonical representation of the matrix"""
        array = self._array
        shape = self._shape
        return f"{self.__class__.__name__}({array!r}, shape=({shape.nrows!r}, {shape.ncols!r}))"

    def __getitem__(self, key):
        array = self._array
        shape = self._shape

        if isinstance(key, tuple):
            rowkey, colkey = key

            if isinstance(rowkey, slice):
                ix = shape.resolve_slice(rowkey, by=ROW)

                if isinstance(colkey, slice):
                    jx = shape.resolve_slice(colkey, by=COL)
                    n  = shape.ncols
                    result = self.__class__.wrap(
                        [array[i * n + j] for i in ix for j in jx],
                        shape=Shape(len(ix), len(jx)),
                    )

                else:
                    j = shape.resolve_index(colkey, by=COL)
                    n = shape.ncols
                    result = self.__class__.wrap(
                        [array[i * n + j] for i in ix],
                        shape=Shape(len(ix), 1),
                    )

            else:
                i = shape.resolve_index(rowkey, by=ROW)

                if isinstance(colkey, slice):
                    jx = shape.resolve_slice(colkey, by=COL)
                    n  = shape.ncols
                    result = self.__class__.wrap(
                        [array[i * n + j] for j in jx],
                        shape=Shape(1, len(jx)),
                    )

                else:
                    j = shape.resolve_index(colkey, by=COL)
                    n = shape.ncols
                    result = array[i * n + j]

        elif isinstance(key, slice):
            ix = range(*key.indices(shape.nrows * shape.ncols))

            result = self.__class__.wrap(
                [array[i] for i in ix],
                shape=Shape(1, len(ix)),
            )

        else:
            try:
                result = array[key]
            except IndexError as error:
                raise IndexError(f"there are {shape.nrows * shape.ncols} items but index is {key}") from error

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
                matmap(operator.__add__, self, other),
                shape=self._shape,
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matmap(operator.__sub__, self, other),
                shape=self._shape,
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matmap(operator.__mul__, self, other),
                shape=self._shape,
            )
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matmap(operator.__truediv__, self, other),
                shape=self._shape,
            )
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matmap(operator.__floordiv__, self, other),
                shape=self._shape,
            )
        return NotImplemented

    def __mod__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matmap(operator.__mod__, self, other),
                shape=self._shape,
            )
        return NotImplemented

    def __divmod__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matmap(divmod, self, other),
                shape=self._shape,
            )
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matmap(operator.__pow__, self, other),
                shape=self._shape,
            )
        return NotImplemented

    def __lshift__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matmap(operator.__lshift__, self, other),
                shape=self._shape,
            )
        return NotImplemented

    def __rshift__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matmap(operator.__rshift__, self, other),
                shape=self._shape,
            )
        return NotImplemented

    def __and__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matmap(operator.__and__, self, other),
                shape=self._shape,
            )
        return NotImplemented

    def __xor__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matmap(operator.__xor__, self, other),
                shape=self._shape,
            )
        return NotImplemented

    def __or__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matmap(operator.__or__, self, other),
                shape=self._shape,
            )
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, MatrixLike):
            return self.__class__(
                matmul(self, other),
                shape=(self.nrows, other.ncols),
            )
        return NotImplemented

    def __neg__(self):
        return self.__class__(
            map(operator.__neg__, self),
            shape=self._shape,
        )

    def __pos__(self):
        return self.__class__(
            map(operator.__pos__, self),
            shape=self._shape,
        )

    def __abs__(self):
        return self.__class__(
            map(operator.__abs__, self),
            shape=self._shape,
        )

    def __invert__(self):
        return self.__class__(
            map(operator.__invert__, self),
            shape=self._shape,
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

    def equal(self, other):
        return self.__class__(
            matmap(operator.__eq__, self, other),
            shape=self._shape,
        )

    def not_equal(self, other):
        return self.__class__(
            matmap(operator.__ne__, self, other),
            shape=self._shape,
        )

    def lesser(self, other):
        return self.__class__(
            matmap(operator.__lt__, self, other),
            shape=self._shape,
        )

    def lesser_equal(self, other):
        return self.__class__(
            matmap(operator.__le__, self, other),
            shape=self._shape,
        )

    def greater(self, other):
        return self.__class__(
            matmap(operator.__gt__, self, other),
            shape=self._shape,
        )

    def greater_equal(self, other):
        return self.__class__(
            matmap(operator.__ge__, self, other),
            shape=self._shape,
        )

    def logical_and(self, other):
        return self.__class__(
            matmap(lambda a, b: not not (a and b), self, other),
            shape=self._shape,
        )

    def logical_or(self, other):
        return self.__class__(
            matmap(lambda a, b: not not (a or b), self, other),
            shape=self._shape,
        )

    def logical_not(self):
        return self.__class__(
            map(lambda a: not a, self),
            shape=self._shape,
        )

    def conjugate(self):
        return self.__class__(
            map(lambda a: a.conjugate(), self),
            shape=self._shape,
        )

    def transpose(self):
        """Return the transpose of the matrix

        This method produces a `MatrixTranspose` object, which is a kind of
        dynamic matrix view that permutes its indices before passing them
        to the "target" matrix (i.e., the matrix that is being viewed, which
        will be the enclosed instance).

        Creation of a `MatrixTranspose` is extremely quick. Item access takes
        a slight performance hit, but remains in constant time.
        """
        from .transpose import \
            MatrixTranspose  # XXX: Avoids circular import - MatrixTranspose often returns frozen matrices
        return MatrixTranspose(self)
