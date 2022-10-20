import copy
import functools
import itertools
import operator
import reprlib
from collections.abc import Sequence
from io import StringIO

from .protocols import (ComplexLike, ComplexMatrixLike, IntegralLike,
                        IntegralMatrixLike, MatrixLike, RealLike,
                        RealMatrixLike)
from .rule import Rule
from .shape import Shape
from .utilities import (conjugate, logical_and, logical_not, logical_or,
                        logical_xor)

__all__ = [
    "Matrix",
    "ComplexMatrix",
    "RealMatrix",
    "IntegralMatrix",
]


def scalar_map(func, a, /):
    if (n := a.size) != 1:
        raise ValueError(f"matrix of size {n} cannot be demoted to a scalar")
    return func(a.data[0])


MISSING = object()  # In case user is operating with None

def matrix_map(func, a, b=MISSING, /, *, reverse=False):
    if b is MISSING:
        return map(func, a.data)
    if isinstance(b, MatrixLike):
        if (h := a.shape) != (k := b.shape):
            raise ValueError(f"matrix of shape {h} is incompatible with operand of shape {k}")
        b = iter(b)
    else:
        if 0 in a.shape:
            raise ValueError("matrix of size 0 is incompatible with operand constant")
        b = itertools.repeat(b)
    return map(func, b, a.data) if reverse else map(func, a.data, b)


class Matrix(Sequence):
    """A sequence type for manipulating arbitrary data types in both one and
    two dimensions

    The `Matrix` type implements the built-in `collections.abc.Sequence`
    interface. Matrices are mutable, and will perform certain operations in
    place - most will additionally return `self`, allowing for methods to be
    chained.

    This class behaves like a `Sequence[T]`, but interfaces
    `Sequence[Sequence[T]]`-like functionality. Indexing and iteration is
    always done in row-major order unless stated otherwise.
    """

    __slots__ = ("data", "shape")

    def __init__(self, values, nrows, ncols):
        """Construct a matrix from the elements of `values`, interpreting it as
        shape `nrows` × `ncols`
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("dimensions must be non-negative")
        data, shape = list(values), Shape(nrows, ncols)
        if (n := len(data)) != (nrows * ncols):
            raise ValueError(f"cannot interpret size {n} iterable as shape {shape}")
        self.data, self.shape = data, shape

    @classmethod
    def wrap(cls, data, shape):
        """Construct a matrix directly from a data list and shape

        This method exists primarily for the benefit of matrix-producing
        functions that have "pre-validated" the data and shape. Should be used
        with caution - this method is not marked as internal because its usage
        is not entirely discouraged if you're aware of the dangers.

        The following properties are required to construct a valid matrix:
        - `data` must be a flattened `list`. That is, the elements of the
          matrix must be on the shallowest depth. A nested list would imply a
          matrix that contains `list` instances.
        - `shape` must be a `Shape`, where the product of its values must equal
          `len(data)`.
        """
        self = cls.__new__(cls)
        self.data, self.shape = data, shape
        return self

    @classmethod
    def fill(cls, value, nrows, ncols):
        """Construct a matrix of shape `nrows` × `ncols`, comprised solely of
        `value`
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("dimensions must be non-negative")
        data = [value] * (nrows * ncols)
        return cls.wrap(data, shape=Shape(nrows, ncols))

    @classmethod
    def refer(cls, other):
        """Construct a matrix from the contents of `other`, sharing the memory
        between them

        Can be used as a sort of "cast" to a different matrix subclass.
        """
        return cls.wrap(other.data, other.shape)

    @classmethod
    def infer(cls, other):
        """Construct a matrix from a singly-nested iterable, using the
        shallowest iterable's length to deduce the number of rows, and the
        nested iterables' lengths to deduce the number of columns

        Raises `ValueError` if the length of the nested iterables is
        inconsistent (i.e., a representation of an irregular matrix).
        """
        data = []

        rows = iter(other)
        try:
            row = next(rows)
        except StopIteration:
            return cls.wrap(data, shape=Shape())
        else:
            data.extend(row)

        m = 1
        n = len(data)

        for m, row in enumerate(rows, start=2):
            k = 0
            for k, val in enumerate(row, start=1):
                data.append(val)
            if n != k:
                raise ValueError(f"row {m} has length {k} but precedent rows have length {n}")

        return cls.wrap(data, shape=Shape(m, n))

    @reprlib.recursive_repr(fillvalue="...")
    def __repr__(self):
        """Return a canonical representation of the matrix"""
        return f"{self.__class__.__name__}({self.data!r}, nrows={self.nrows!r}, ncols={self.ncols!r})"

    def __str__(self):
        """Return a string representation of the matrix

        Elements of the matrix are organized into right-justified fields of
        width 10. If an element's string representation goes beyond this width,
        it will be truncated to 9 characters, using `…` as a suffix. A single
        space is used to delineate elements.

        Customizable string formatting (through `__format__()`) is a planned
        addition. For the time being, custom formatting must be done manually.
        The implementation for this method is subject to change.
        """
        h = self.shape

        if not h.size:
            return f"Empty matrix ({h})"

        result = StringIO()
        items  = iter(self)

        max_width = 10

        for _ in range(h.nrows):
            result.write("| ")

            for _ in range(h.ncols):
                chars = str(next(items))
                if len(chars) > max_width:
                    result.write(f"{chars[:max_width - 1]}…")
                else:
                    result.write(chars.rjust(max_width))
                result.write(" ")

            result.write("|\n")

        result.write(f"({h})")

        return result.getvalue()

    def __len__(self):
        """Return the matrix's size"""
        return self.size

    def __getitem__(self, key):
        """Return the element or sub-matrix corresponding to `key`

        If `key` is an integer or slice, it is treated as if it is indexing the
        flattened matrix, returning the corresponding value(s). A slice will
        always return a new matrix of shape `(1, N)`, where `N` is the length
        of the slice's range.

        If `key` is a tuple, the first index is applied against the rows, while
        the second is applied against the columns.

        A tuple of two integers, `(i, j)`, will return the element at row `i`,
        column `j`. All other tuple variations will return a new sub-matrix of
        shape `(M, N)`, where `M` is the length of the first slice's range, and
        `N` is the length of the second slice's range - integers are treated as
        length 1 slices if mixed with at least one other slice.
        """
        data = self.data
        h = self.shape

        if isinstance(key, tuple):

            def getitems(keys, nrows, ncols):
                n = self.ncols
                return type(self).wrap(
                    [data[i * n + j] for i, j in keys],
                    shape=Shape(nrows, ncols),
                )

            rowkey, colkey = key

            if isinstance(rowkey, slice):
                ix = h.resolve_slice(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = h.resolve_slice(colkey, by=Rule.COL)
                    result = getitems(
                        itertools.product(ix, jx),
                        nrows=len(ix),
                        ncols=len(jx),
                    )

                else:
                    j = h.resolve_index(colkey, by=Rule.COL)
                    result = getitems(
                        zip(ix, itertools.repeat(j)),
                        nrows=len(ix),
                        ncols=1,
                    )

            else:
                i = h.resolve_index(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = h.resolve_slice(colkey, by=Rule.COL)
                    result = getitems(
                        zip(itertools.repeat(i), jx),
                        nrows=1,
                        ncols=len(jx),
                    )

                else:
                    j = h.resolve_index(colkey, by=Rule.COL)
                    result = data[i * h.ncols + j]

            return result

        if isinstance(key, slice):

            def getitems(keys, nrows, ncols):
                return type(self).wrap(
                    [data[i] for i in keys],
                    shape=Shape(nrows, ncols),
                )

            ix = range(*key.indices(h.size))
            result = getitems(
                ix,
                nrows=1,
                ncols=len(ix),
            )

            return result

        try:
            result = data[key]
        except IndexError:
            raise IndexError(f"there are {h.size} items but index is {key}") from None
        else:
            return result

    def __setitem__(self, key, other):
        """Overwrite the element or sub-matrix corresponding to `key`

        If `key` is an integer or slice, it is treated as if it is indexing the
        flattened matrix, overwriting the corresponding value(s).

        If `key` is a tuple, the first index is applied against the rows, while
        the second is applied against the columns.

        A tuple of two integers, `(i, j)`, will overwrite the element at row
        `i`, column `j`. All other tuple variations will overwrite a sub-matrix
        of shape `(M, N)`, where `M` is the length of the first slice's range,
        and `N` is the length of the second slice's range - integers are
        treated as length 1 slices if mixed with at least one other slice.
        """
        data = self.data
        h = self.shape

        if isinstance(key, tuple):

            def setitems(keys, nrows, ncols):
                if (h := Shape(nrows, ncols)) != (k := other.shape):
                    raise ValueError(f"selection of shape {h} is incompatible with operand of shape {k}")
                n = self.ncols
                for (i, j), x in zip(keys, other):
                    data[i * n + j] = x

            rowkey, colkey = key

            if isinstance(rowkey, slice):
                ix = h.resolve_slice(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = h.resolve_slice(colkey, by=Rule.COL)
                    setitems(
                        itertools.product(ix, jx),
                        nrows=len(ix),
                        ncols=len(jx),
                    )

                else:
                    j = h.resolve_index(colkey, by=Rule.COL)
                    setitems(
                        zip(ix, itertools.repeat(j)),
                        nrows=len(ix),
                        ncols=1,
                    )

            else:
                i = h.resolve_index(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = h.resolve_slice(colkey, by=Rule.COL)
                    setitems(
                        zip(itertools.repeat(i), jx),
                        nrows=1,
                        ncols=len(jx),
                    )

                else:
                    j = h.resolve_index(colkey, by=Rule.COL)
                    data[i * h.ncols + j] = other

            return

        if isinstance(key, slice):

            def setitems(keys, nrows, ncols):
                if (h := Shape(nrows, ncols)) != (k := other.shape):
                    raise ValueError(f"selection of shape {h} is incompatible with operand of shape {k}")
                for i, x in zip(keys, other):
                    data[i] = x

            ix = range(*key.indices(h.size))
            setitems(
                ix,
                nrows=1,
                ncols=len(ix),
            )

            return

        try:
            data[key] = other
        except IndexError:
            raise IndexError(f"there are {h.size} items but index is {key}") from None

    def __iter__(self):
        """Return an iterator over the elements of the matrix"""
        yield from iter(self.data)

    def __reversed__(self):
        """Return a reverse iterator over the elements of the matrix"""
        yield from reversed(self.data)

    def __contains__(self, value):
        """Return true if the matrix contains `value`, otherwise false"""
        return value in self.data

    def __deepcopy__(self, memo=None):
        """Return a deep copy of the matrix"""
        data = self.data
        h = self.shape
        return type(self).wrap(
            copy.deepcopy(data, memo=memo),
            shape=h.copy(),
        )

    def __copy__(self):
        """Return a shallow copy of the matrix"""
        data = self.data
        h = self.shape
        return type(self).wrap(
            copy.copy(data),
            shape=h.copy(),
        )

    def __eq__(self, other):
        """Return true if element-wise `a == b` is true for all element pairs,
        `a` and `b`, otherwise false

        For a matrix of each comparison result, use the `eq()` method.
        """
        return all(matrix_map(operator.eq, self, other))

    def __and__(self, other, *, reverse=False):
        """Return element-wise `logical_and(a, b)`"""
        return IntegralMatrix(
            matrix_map(
                logical_and,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rand__(self, other):
        """Return element-wise `logical_and(b, a)`"""
        return self.__and__(other, reverse=True)

    def __or__(self, other, *, reverse=False):
        """Return element-wise `logical_or(a, b)`"""
        return IntegralMatrix(
            matrix_map(
                logical_or,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __ror__(self, other):
        """Return element-wise `logical_or(b, a)`"""
        return self.__or__(other, reverse=True)

    def __xor__(self, other, *, reverse=False):
        """Return element-wise `logical_xor(a, b)`"""
        return IntegralMatrix(
            matrix_map(
                logical_xor,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rxor__(self, other):
        """Return element-wise `logical_xor(b, a)`"""
        return self.__xor__(other, reverse=True)

    def __invert__(self):
        """Return element-wise `logical_not(a)`"""
        return IntegralMatrix(
            matrix_map(
                logical_not,
                self,
            ),
            *self.shape,
        )

    @property
    def nrows(self):
        """The matrix's number of rows"""
        return self.shape.nrows

    @property
    def ncols(self):
        """The matrix's number of columns"""
        return self.shape.ncols

    @property
    def size(self):
        """The product of the number of rows and columns"""
        return self.shape.size

    def index(self, value, start=0, stop=None):
        """Return the index of the first element equal to `value`

        Raises `ValueError` if the value could not be found in the matrix.
        """
        try:
            index = super().index(value, start, stop)  # type: ignore[arg-type]
        except ValueError:
            raise ValueError("value not found") from None
        else:
            return index

    def count(self, value):
        """Return the number of times `value` appears in the matrix"""
        return super().count(value)

    def reverse(self):
        """Reverse the matrix's elements in place"""
        self.data.reverse()
        return self

    def eq(self, other):
        """Return element-wise `a == b`"""
        return IntegralMatrix(
            matrix_map(
                operator.eq,
                self,
                other,
            ),
            *self.shape,
        )

    def ne(self, other):
        """Return element-wise `a != b`"""
        return IntegralMatrix(
            matrix_map(
                operator.ne,
                self,
                other,
            ),
            *self.shape,
        )

    def reshape(self, nrows, ncols):
        """Re-interpret the matrix's shape

        Raises `ValueError` if any of the given dimensions are negative, or if
        their product does not equal the matrix's current size.
        """
        h = self.shape
        if nrows < 0 or ncols < 0:
            raise ValueError("dimensions must be non-negative")
        if (n := h.size) != (nrows * ncols):
            raise ValueError(f"cannot re-shape size {n} matrix as shape {nrows} × {ncols}")
        h.nrows = nrows
        h.ncols = ncols
        return self

    def slices(self, *, by=Rule.ROW):
        """Return an iterator that yields shallow copies of each row or column"""
        data = self.data
        h = self.shape
        k = h.subshape(by=by)
        cls = type(self)
        for i in range(h[by]):
            temp = data[h.slice(i, by=by)]
            yield cls.wrap(temp, shape=k.copy())

    def mask(self, selector, null):
        """Replace the elements who have a true parallel value in `selector`
        with `null`

        Raises `ValueError` if operand matrices have incompatible shapes.
        """
        data = self.data
        if (h := self.shape) != (k := selector.shape):
            raise ValueError(f"matrix of shape {h} is incompatible with operand of shape {k}")
        for i, x in enumerate(selector):
            if x: data[i] = null
        return self

    def replace(self, old, new, *, times=None):
        """Replace elements equal to `old` with `new`

        If `times` is given, only the first `times` occurrences of `old` will
        be replaced.

        This method considers two objects equal if a comparison by identity or
        equality is satisfied, which can sometimes be helpful for replacing
        objects such as `math.nan`.
        """
        data = self.data
        for i in itertools.islice(
            (i for i, x in enumerate(data) if x is old or x == old),
            times,
        ):
            data[i] = new
        return self

    def swap(self, key1, key2, *, by=Rule.ROW):
        """Swap the two rows or columns beginning at the given indices

        Note that, due to how `Matrix` stores its data, swapping is performed
        in linear time with respect to the specified dimension.
        """
        data = self.data
        h = self.shape
        i = h.resolve_index(key1, by=by)
        j = h.resolve_index(key2, by=by)
        for x, y in zip(h.range(i, by=by), h.range(j, by=by)):
            data[x], data[y] = data[y], data[x]
        return self

    def flip(self, *, by=Rule.ROW):
        """Reverse the matrix's rows or columns in place"""
        data = self.data
        h = self.shape
        n = h[by]
        for i in range(n // 2):
            j = n - i - 1
            for x, y in zip(h.range(i, by=by), h.range(j, by=by)):
                data[x], data[y] = data[y], data[x]
        return self

    def flatten(self, *, by=Rule.ROW):
        """Re-shape the matrix to a row or column vector

        If flattened to a column vector, the elements are arranged into
        column-major order.
        """
        if by: self.transpose()  # For column-major order
        h = self.shape
        h[not by] = h.size
        h[by] = 1
        return self

    def transpose(self):
        """Transpose the matrix in place"""
        h = self.shape
        if (m := h.nrows) > 1 and (n := h.ncols) > 1:
            data = self.data
            ix = range(m)
            jx = range(n)
            data[:] = (data[i * n + j] for j in jx for i in ix)
        h.reverse()
        return self

    def stack(self, other, *, by=Rule.ROW):
        """Stack a matrix along the rows or columns

        Raises `ValueError` if the inverse dimension differs between the
        operand matrices.
        """
        if self is other:
            other = other.copy()

        data = self.data
        h, k = self.shape, other.shape

        dy = by.inverse
        if (m := h[dy]) != (n := k[dy]):
            raise ValueError(f"matrix has {m} {dy.handle}s but operand has {n}")

        (m, n), (_, q) = (h, k)

        if by and m > 1:
            left, right = iter(self), iter(other)
            data[:] = (
                x
                for _ in range(m)
                for x in itertools.chain(itertools.islice(left, n), itertools.islice(right, q))
            )
        else:
            data.extend(other)

        h[by] += k[by]

        return self

    def pull(self, key=-1, *, by=Rule.ROW):
        """Remove and return the row or column corresponding to `key`

        Raises `IndexError` if the matrix is empty, or if the index is out of
        range.
        """
        data = self.data
        h = self.shape

        keys = h.slice(h.resolve_index(key, by=by), by=by)
        temp = data[keys]
        h[by] -= 1

        del data[keys]

        return type(self).wrap(temp, shape=h.subshape(by=by))

    def copy(self, *, deep=False):
        """Return a shallow or deep copy of the matrix"""
        return copy.deepcopy(self) if deep else copy.copy(self)


def multiply(a, b, /, *, reverse=False):
    (m, n), (p, q) = (b.shape, a.shape) if reverse else (a.shape, b.shape)

    if n != p:
        raise ValueError(f"left-hand side matrix has {n} columns, but right-hand side has {p} rows")
    if not n:
        return itertools.repeat(0, m * q)  # Use int 0 here since it supports all numeric operations

    a = a.data

    ix = range(m)
    jx = range(q)
    kx = range(n)

    return (
        functools.reduce(
            operator.add,
            (a[i * n + k] * b[k * q + j] for k in kx),
        )
        for i in ix
        for j in jx
    )


class ComplexMatrix(Matrix):
    """Subclass of `Matrix` that adds operations for complex-like objects"""

    __slots__ = ()

    def __add__(self, other, *, reverse=False):
        """Return element-wise `a + b`"""
        if isinstance(other, ComplexLike):
            cls = ComplexMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.add,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __radd__(self, other):
        """Return element-wise `b + a`"""
        return self.__add__(other, reverse=True)

    def __sub__(self, other, *, reverse=False):
        """Return element-wise `a - b`"""
        if isinstance(other, ComplexLike):
            cls = ComplexMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.sub,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rsub__(self, other):
        """Return element-wise `b - a`"""
        return self.__sub__(other, reverse=True)

    def __mul__(self, other, *, reverse=False):
        """Return element-wise `a * b`"""
        if isinstance(other, ComplexLike):
            cls = ComplexMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.mul,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rmul__(self, other):
        """Return element-wise `b * a`"""
        return self.__mul__(other, reverse=True)

    def __truediv__(self, other, *, reverse=False):
        """Return element-wise `a / b`"""
        if isinstance(other, ComplexLike):
            cls = ComplexMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.truediv,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rtruediv__(self, other):
        """Return element-wise `b / a`"""
        return self.__truediv__(other, reverse=True)

    def __pow__(self, other, *, reverse=False):
        """Return element-wise `a ** b`"""
        if isinstance(other, ComplexLike):
            cls = ComplexMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.pow,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rpow__(self, other):
        """Return element-wise `b ** a`"""
        return self.__pow__(other, reverse=True)

    def __matmul__(self, other, *, reverse=False):
        """Return the matrix product `a @ b`"""
        if isinstance(other, ComplexMatrixLike):
            cls = ComplexMatrix
        elif isinstance(other, MatrixLike):
            cls = Matrix
        else:
            return NotImplemented
        values = multiply(self, other, reverse=reverse)
        if reverse:
            return cls(values, nrows=other.nrows, ncols=self.ncols)
        else:
            return cls(values, nrows=self.nrows, ncols=other.ncols)

    def __rmatmul__(self, other):
        """Return the matrix product `b @ a`"""
        return self.__matmul__(other, reverse=True)

    def __neg__(self):
        """Return element-wise `-a`"""
        return ComplexMatrix(
            matrix_map(
                operator.neg,
                self,
            ),
            *self.shape,
        )

    def __pos__(self):
        """Return element-wise `+a`"""
        return ComplexMatrix(
            matrix_map(
                operator.pos,
                self,
            ),
            *self.shape,
        )

    def __abs__(self):
        """Return element-wise `abs(a)`"""
        return RealMatrix(
            matrix_map(
                abs,
                self,
            ),
            *self.shape,
        )

    def __complex__(self):
        """Return the matrix as a built-in `complex` instance"""
        return scalar_map(complex, self)

    def conjugate(self):
        """Return element-wise `conjugate(a)`"""
        return ComplexMatrix(
            matrix_map(
                conjugate,
                self,
            ),
            *self.shape,
        )

    def complex(self):
        """Return element-wise `complex(a)`"""
        return ComplexMatrix(
            matrix_map(
                complex,
                self,
            ),
            *self.shape,
        )


class RealMatrix(Matrix):
    """Subclass of `Matrix` that adds operations for real-like objects"""

    __slots__ = ()

    def __lt__(self, other):
        """Return true if element-wise `a < b` is true for all element pairs,
        `a` and `b`, otherwise false

        For a matrix of each comparison result, use the `lt()` method.
        """
        return all(matrix_map(operator.lt, self, other))

    def __le__(self, other):
        """Return true if element-wise `a <= b` is true for all element pairs,
        `a` and `b`, otherwise false

        For a matrix of each comparison result, use the `le()` method.
        """
        return all(matrix_map(operator.le, self, other))

    def __gt__(self, other):
        """Return true if element-wise `a > b` is true for all element pairs,
        `a` and `b`, otherwise false

        For a matrix of each comparison result, use the `gt()` method.
        """
        return all(matrix_map(operator.gt, self, other))

    def __ge__(self, other):
        """Return true if element-wise `a >= b` is true for all element pairs,
        `a` and `b`, otherwise false

        For a matrix of each comparison result, use the `ge()` method.
        """
        return all(matrix_map(operator.ge, self, other))

    def __add__(self, other, *, reverse=False):
        """Return element-wise `a + b`"""
        if isinstance(other, RealLike):
            cls = RealMatrix
        elif isinstance(other, ComplexLike):
            cls = ComplexMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.add,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __radd__(self, other):
        """Return element-wise `b + a`"""
        return self.__add__(other, reverse=True)

    def __sub__(self, other, *, reverse=False):
        """Return element-wise `a - b`"""
        if isinstance(other, RealLike):
            cls = RealMatrix
        elif isinstance(other, ComplexLike):
            cls = ComplexMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.sub,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rsub__(self, other):
        """Return element-wise `b - a`"""
        return self.__sub__(other, reverse=True)

    def __mul__(self, other, *, reverse=False):
        """Return element-wise `a * b`"""
        if isinstance(other, RealLike):
            cls = RealMatrix
        elif isinstance(other, ComplexLike):
            cls = ComplexMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.mul,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rmul__(self, other):
        """Return element-wise `b * a`"""
        return self.__mul__(other, reverse=True)

    def __truediv__(self, other, *, reverse=False):
        """Return element-wise `a / b`"""
        if isinstance(other, RealLike):
            cls = RealMatrix
        elif isinstance(other, ComplexLike):
            cls = ComplexMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.truediv,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rtruediv__(self, other):
        """Return element-wise `b / a`"""
        return self.__truediv__(other, reverse=True)

    def __pow__(self, other, *, reverse=False):
        """Return element-wise `a ** b`"""
        if isinstance(other, ComplexLike):
            cls = ComplexMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.pow,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rpow__(self, other):
        """Return element-wise `b ** a`"""
        return self.__pow__(other, reverse=True)

    def __matmul__(self, other, *, reverse=False):
        """Return the matrix product `a @ b`"""
        if isinstance(other, RealMatrixLike):
            cls = RealMatrix
        elif isinstance(other, ComplexMatrixLike):
            cls = ComplexMatrix
        elif isinstance(other, MatrixLike):
            cls = Matrix
        else:
            return NotImplemented
        values = multiply(self, other, reverse=reverse)
        if reverse:
            return cls(values, nrows=other.nrows, ncols=self.ncols)
        else:
            return cls(values, nrows=self.nrows, ncols=other.ncols)

    def __rmatmul__(self, other):
        """Return the matrix product `b @ a`"""
        return self.__matmul__(other, reverse=True)

    def __floordiv__(self, other, *, reverse=False):
        """Return element-wise `a // b`"""
        if isinstance(other, RealLike):
            cls = RealMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.floordiv,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rfloordiv__(self, other):
        """Return element-wise `b // a`"""
        return self.__floordiv__(other, reverse=True)

    def __mod__(self, other, *, reverse=False):
        """Return element-wise `a % b`"""
        if isinstance(other, RealLike):
            cls = RealMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.mod,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rmod__(self, other):
        """Return element-wise `b % a`"""
        return self.__mod__(other, reverse=True)

    def __neg__(self):
        """Return element-wise `-a`"""
        return RealMatrix(
            matrix_map(
                operator.neg,
                self,
            ),
            *self.shape,
        )

    def __pos__(self):
        """Return element-wise `+a`"""
        return RealMatrix(
            matrix_map(
                operator.pos,
                self,
            ),
            *self.shape,
        )

    def __abs__(self):
        """Return element-wise `abs(a)`"""
        return RealMatrix(
            matrix_map(
                abs,
                self,
            ),
            *self.shape,
        )

    def __complex__(self):
        """Return the matrix as a built-in `complex` instance"""
        return scalar_map(complex, self)

    def __float__(self):
        """Return the matrix as a built-in `float` instance"""
        return scalar_map(float, self)

    def lt(self, other):
        """Return element-wise `a < b`"""
        return IntegralMatrix(
            matrix_map(
                operator.lt,
                self,
                other,
            ),
            *self.shape,
        )

    def le(self, other):
        """Return element-wise `a <= b`"""
        return IntegralMatrix(
            matrix_map(
                operator.le,
                self,
                other,
            ),
            *self.shape,
        )

    def gt(self, other):
        """Return element-wise `a > b`"""
        return IntegralMatrix(
            matrix_map(
                operator.gt,
                self,
                other,
            ),
            *self.shape,
        )

    def ge(self, other):
        """Return element-wise `a >= b`"""
        return IntegralMatrix(
            matrix_map(
                operator.ge,
                self,
                other,
            ),
            *self.shape,
        )

    def conjugate(self):
        """Return element-wise `conjugate(a)`"""
        return RealMatrix(
            matrix_map(
                conjugate,
                self,
            ),
            *self.shape,
        )

    def complex(self):
        """Return element-wise `complex(a)`"""
        return ComplexMatrix(
            matrix_map(
                complex,
                self,
            ),
            *self.shape,
        )

    def float(self):
        """Return element-wise `float(a)`"""
        return RealMatrix(
            matrix_map(
                float,
                self,
            ),
            *self.shape,
        )


class IntegralMatrix(Matrix):
    """Subclass of `Matrix` that adds operations for integral-like objects"""

    __slots__ = ()

    def __lt__(self, other):
        """Return true if element-wise `a < b` is true for all element pairs,
        `a` and `b`, otherwise false

        For a matrix of each comparison result, use the `lt()` method.
        """
        return all(matrix_map(operator.lt, self, other))

    def __le__(self, other):
        """Return true if element-wise `a <= b` is true for all element pairs,
        `a` and `b`, otherwise false

        For a matrix of each comparison result, use the `le()` method.
        """
        return all(matrix_map(operator.le, self, other))

    def __gt__(self, other):
        """Return true if element-wise `a > b` is true for all element pairs,
        `a` and `b`, otherwise false

        For a matrix of each comparison result, use the `gt()` method.
        """
        return all(matrix_map(operator.gt, self, other))

    def __ge__(self, other):
        """Return true if element-wise `a >= b` is true for all element pairs,
        `a` and `b`, otherwise false

        For a matrix of each comparison result, use the `ge()` method.
        """
        return all(matrix_map(operator.ge, self, other))

    def __add__(self, other, *, reverse=False):
        """Return element-wise `a + b`"""
        if isinstance(other, IntegralLike):
            cls = IntegralMatrix
        elif isinstance(other, RealLike):
            cls = RealMatrix
        elif isinstance(other, ComplexLike):
            cls = ComplexMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.add,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __radd__(self, other):
        """Return element-wise `b + a`"""
        return self.__add__(other, reverse=True)

    def __sub__(self, other, *, reverse=False):
        """Return element-wise `a - b`"""
        if isinstance(other, IntegralLike):
            cls = IntegralMatrix
        elif isinstance(other, RealLike):
            cls = RealMatrix
        elif isinstance(other, ComplexLike):
            cls = ComplexMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.sub,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rsub__(self, other):
        """Return element-wise `b - a`"""
        return self.__sub__(other, reverse=True)

    def __mul__(self, other, *, reverse=False):
        """Return element-wise `a * b`"""
        if isinstance(other, IntegralLike):
            cls = IntegralMatrix
        elif isinstance(other, RealLike):
            cls = RealMatrix
        elif isinstance(other, ComplexLike):
            cls = ComplexMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.mul,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rmul__(self, other):
        """Return element-wise `b * a`"""
        return self.__mul__(other, reverse=True)

    def __truediv__(self, other, *, reverse=False):
        """Return element-wise `a / b`"""
        if isinstance(other, RealLike):
            cls = RealMatrix
        elif isinstance(other, ComplexLike):
            cls = ComplexMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.truediv,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rtruediv__(self, other):
        """Return element-wise `b / a`"""
        return self.__truediv__(other, reverse=True)

    def __pow__(self, other, *, reverse=False):
        """Return element-wise `a ** b`"""
        if isinstance(other, ComplexLike):
            cls = ComplexMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.pow,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rpow__(self, other):
        """Return element-wise `b ** a`"""
        return self.__pow__(other, reverse=True)

    def __matmul__(self, other, *, reverse=False):
        """Return the matrix product `a @ b`"""
        if isinstance(other, IntegralMatrixLike):
            cls = IntegralMatrix
        elif isinstance(other, RealMatrixLike):
            cls = RealMatrix
        elif isinstance(other, ComplexMatrixLike):
            cls = ComplexMatrix
        elif isinstance(other, MatrixLike):
            cls = Matrix
        else:
            return NotImplemented
        values = multiply(self, other, reverse=reverse)
        if reverse:
            return cls(values, nrows=other.nrows, ncols=self.ncols)
        else:
            return cls(values, nrows=self.nrows, ncols=other.ncols)

    def __rmatmul__(self, other):
        """Return the matrix product `b @ a`"""
        return self.__matmul__(other, reverse=True)

    def __floordiv__(self, other, *, reverse=False):
        """Return element-wise `a // b`"""
        if isinstance(other, IntegralLike):
            cls = IntegralMatrix
        elif isinstance(other, RealLike):
            cls = RealMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.floordiv,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rfloordiv__(self, other):
        """Return element-wise `b // a`"""
        return self.__floordiv__(other, reverse=True)

    def __mod__(self, other, *, reverse=False):
        """Return element-wise `a % b`"""
        if isinstance(other, IntegralLike):
            cls = IntegralMatrix
        elif isinstance(other, RealLike):
            cls = RealMatrix
        else:
            cls = Matrix
        return cls(
            matrix_map(
                operator.mod,
                self,
                other,
                reverse=reverse,
            ),
            *self.shape,
        )

    def __rmod__(self, other):
        """Return element-wise `b % a`"""
        return self.__mod__(other, reverse=True)

    def __neg__(self):
        """Return element-wise `-a`"""
        return IntegralMatrix(
            matrix_map(
                operator.neg,
                self,
            ),
            *self.shape,
        )

    def __pos__(self):
        """Return element-wise `+a`"""
        return IntegralMatrix(
            matrix_map(
                operator.pos,
                self,
            ),
            *self.shape,
        )

    def __abs__(self):
        """Return element-wise `abs(a)`"""
        return IntegralMatrix(
            matrix_map(
                abs,
                self,
            ),
            *self.shape,
        )

    def __complex__(self):
        """Return the matrix as a built-in `complex` instance"""
        return scalar_map(complex, self)

    def __float__(self):
        """Return the matrix as a built-in `float` instance"""
        return scalar_map(float, self)

    def __int__(self):
        """Return the matrix as a built-in `int` instance"""
        return scalar_map(int, self)

    def __index__(self):
        """Return the matrix as a built-in `int` instance, losslessly"""
        return scalar_map(operator.index, self)

    def lt(self, other):
        """Return element-wise `a < b`"""
        return IntegralMatrix(
            matrix_map(
                operator.lt,
                self,
                other,
            ),
            *self.shape,
        )

    def le(self, other):
        """Return element-wise `a <= b`"""
        return IntegralMatrix(
            matrix_map(
                operator.le,
                self,
                other,
            ),
            *self.shape,
        )

    def gt(self, other):
        """Return element-wise `a > b`"""
        return IntegralMatrix(
            matrix_map(
                operator.gt,
                self,
                other,
            ),
            *self.shape,
        )

    def ge(self, other):
        """Return element-wise `a >= b`"""
        return IntegralMatrix(
            matrix_map(
                operator.ge,
                self,
                other,
            ),
            *self.shape,
        )

    def conjugate(self):
        """Return element-wise `conjugate(a)`"""
        return IntegralMatrix(
            matrix_map(
                conjugate,
                self,
            ),
            *self.shape,
        )

    def complex(self):
        """Return element-wise `complex(a)`"""
        return ComplexMatrix(
            matrix_map(
                complex,
                self,
            ),
            *self.shape,
        )

    def float(self):
        """Return element-wise `float(a)`"""
        return RealMatrix(
            matrix_map(
                float,
                self,
            ),
            *self.shape,
        )

    def int(self):
        """Return element-wise `int(a)`"""
        return IntegralMatrix(
            matrix_map(
                int,
                self,
            ),
            *self.shape,
        )
