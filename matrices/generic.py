import copy
import itertools
import operator
import reprlib
from collections.abc import Sequence
from enum import Enum
from io import StringIO

# We need IntegralMatrix in some methods (which derives from GenericMatrix), so
# this must be module-level to avoid a circular import
from . import numeric

__all__ = [
    "Rule",
    "IncompatibilityError",
    "Shape",
    "GenericMatrix",
]


class Rule(Enum):
    """The direction by which to operate within a matrix

    The value of a rule member is usable as an index that retrieves the rule's
    corresponding dimension from a matrix's shape (or any two-element sequence
    type).
    """

    ROW = 0
    COL = 1

    def __index__(self):
        """Return the rule's value

        This exists so that members can be used directly as an index for
        sequences that coerce integer keys via `operator.index()`.
        """
        return self.value

    def __invert__(self):
        """Return the rule's inverse"""
        return self.inverse

    @property
    def inverse(self):
        """The rule's inverse

        The column-rule if row-rule, or the row-rule if column-rule. Equivalent
        to `Rule(not self.value)`.
        """
        return Rule(not self.value)

    @property
    def true_name(self):
        """The rule's unformatted name"""
        return "column" if self is Rule.COL else "row"

    def subshape(self, shape):
        """Return the rule's shape given the matrix's shape"""
        shape = shape.copy()
        shape[self] = 1
        return shape

    def serialize(self, index, shape):
        """Return the start, stop, and step values required to create a range
        or slice object of the rule's shape beginning at `index`

        The input `index` must be positive - negative indices may produce
        unexpected results. This requirement is not checked for.
        """
        major_index = self.value
        minor_index = not major_index

        major = shape[major_index]
        minor = shape[minor_index]

        major_step = major_index * major + minor_index
        minor_step = minor_index * minor + major_index

        start = minor_step * index
        stop  = major_step * minor + start

        return start, stop, major_step

    def range(self, index, shape):
        """Return a range of indices that can be used to construct a
        sub-sequence of the rule's shape beginning at `index`

        See `serialize()` for more details.
        """
        return range(*self.serialize(index, shape))

    def slice(self, index, shape):
        """Return a slice that can be used to construct a sub-sequence of the
        rule's shape beginning at `index`

        See `serialize()` for more details.
        """
        return slice(*self.serialize(index, shape))


class IncompatibilityError(ValueError):
    """Raised if two matrices are incompatible with each other

    Matrices are considered to be "compatible" if at least one of the following
    criteria is met:
    - Their shapes are equivalent
    - Their sizes are equivalent and both matrices are vector-like

    In a nutshell: vectors are compatible so long as their sizes match -
    matrices are compatible so long as their shapes match.

    A matrix is considered "vector-like" if at least one dimension of its shape
    is equal to 1. This can be easily checked with `1 in matrix.shape`.
    """
    pass


class Shape:
    """A mutable sequence-like type for storing the dimensions of a matrix

    Instances of `Shape` support integer-indexing (both reading and writing),
    but not slice-indexing. Negative components are never checked for during
    write operations.
    """

    __slots__ = ("data",)

    def __init__(self, nrows=0, ncols=0):
        """Construct a shape from its two dimensions"""
        self.data = [nrows, ncols]

    def __repr__(self):
        """Return a canonical representation of the shape"""
        return f"Shape(nrows={self.nrows!r}, ncols={self.ncols!r})"

    def __str__(self):
        """Return a string representation of the shape"""
        return f"{self.nrows} × {self.ncols}"

    def __eq__(self, other):
        """Return true if the two shapes have equal dimensions, otherwise false"""
        if not isinstance(other, Shape):
            return NotImplemented
        return self.data == other.data

    def __ne__(self, other):
        """Return true if the two shapes do not have equal dimensions,
        otherwise false
        """
        if not isinstance(other, Shape):
            return NotImplemented
        return self.data != other.data

    def __len__(self):
        """Return literal 2"""
        return 2

    def __getitem__(self, key):
        """Return the dimension corresponding to `key`"""
        key = operator.index(key)
        return self.data[key]

    def __setitem__(self, key, value):
        """Set the dimension corresponding to `key` with `value`"""
        key = operator.index(key)
        self.data[key] = value

    def __iter__(self):
        """Return an iterator over the dimensions of the shape"""
        yield from iter(self.data)

    def __reversed__(self):
        """Return a reversed iterator over the dimensions of the shape"""
        yield from reversed(self.data)

    def __contains__(self, value):
        """Return true if the shape contains `value`, otherwise false"""
        return value in self.data

    def __deepcopy__(self, memo=None):
        """Return a copy of the shape"""
        return Shape(*self.data)  # Our components are immutable

    __copy__ = __deepcopy__

    @property
    def nrows(self):
        """The first dimension of the shape"""
        return self.data[0]

    @nrows.setter
    def nrows(self, value):
        self.data[0] = value

    @property
    def ncols(self):
        """The second dimension of the shape"""
        return self.data[1]

    @ncols.setter
    def ncols(self, value):
        self.data[1] = value

    @property
    def size(self):
        """The product of the shape's dimensions"""
        nrows, ncols = self.data
        return nrows * ncols

    def compatible(self, other, *, error=False):
        """Return true if the shape is compatible with `other`, otherwise false

        If `error` is true, `IncompatibilityError` will be raised if the two
        shapes are found to be incompatible. See the `IncompatibilityError`
        documentation for more details.
        """
        res = self == other
        if not res:
            res = self.size == other.size and 1 in self and 1 in other
            if error and not res:
                raise IncompatibilityError(f"operating matrix of shape {self} is incompatible with operand of shape {other}")
        return res

    def copy(self):
        """Return a copy of the shape"""
        return Shape(*self.data)

    def reverse(self):
        """Reverse the shape's dimensions in place"""
        self.data.reverse()
        return self

    def resolve_index(self, key, *, by=Rule.ROW):
        """Return an index `key` as an equivalent integer, respective to a rule

        Raises `IndexError` if the key is out of range.
        """
        n = self.data[by]

        res = operator.index(key)
        res = res + (n * (res < 0))

        if res < 0 or res >= n:
            name = by.true_name
            raise IndexError(f"there are {n} {name}s but index is {key}")

        return res

    def resolve_slice(self, key, *, by=Rule.ROW):
        """Return a slice `key` as an equivalent sequence of indices,
        respective to a rule
        """
        n = self.data[by]
        return range(*key.indices(n))


def logical_and(a, b, /): return not not (a and b)
def logical_or(a, b, /): return not not (a or b)
def logical_xor(a, b, /): return (not not a) is not (not not b)
def logical_not(a, /): return not a


class GenericMatrix(Sequence):
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

        Can be used as a kind of "cast" to a different matrix subclass if
        `other` goes unused.
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
        name = type(self).__name__
        return f"{name}({self.data!r}, nrows={self.nrows!r}, ncols={self.ncols!r})"

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
        items = iter(self)
        h = self.shape

        max_width = 10
        res = StringIO()

        if h.size:
            for _ in range(h.nrows):
                res.write("| ")

                for _ in range(h.ncols):
                    chars = str(next(items))
                    if len(chars) > max_width:
                        res.write(f"{chars[:max_width - 1]}…")
                    else:
                        res.write(chars.rjust(max_width))
                    res.write(" ")

                res.write("|\n")
        else:
            res.write("Empty matrix ")

        res.write(f"({h})")

        return res.getvalue()

    def unary_operator(self, basis, *, out=None):
        out = out or GenericMatrix
        h = self.shape
        return out.wrap(list(map(basis, self)), shape=h.copy())

    def binary_operator(self, basis, other, *, exp=None, out=None, reverse=False):
        exp, out = (exp or GenericMatrix, out or GenericMatrix)
        if not isinstance(other, exp):
            return NotImplemented
        h = self.shape
        h.compatible(other.shape, error=True)
        if reverse:
            data = list(map(basis, other, self))
        else:
            data = list(map(basis, self, other))
        return out.wrap(data, shape=h.copy())

    def __eq__(self, other):
        """Element-wise `__eq__()`"""
        return self.binary_operator(operator.eq, other)

    def __ne__(self, other):
        """Element-wise `__ne__()`"""
        return self.binary_operator(operator.ne, other)

    def __and__(self, other):
        """Element-wise logical AND"""
        return self.binary_operator(
            logical_and,
            other,
            out=numeric.IntegralMatrix,
        )

    __rand__ = __and__

    def __xor__(self, other):
        """Element-wise logical XOR"""
        return self.binary_operator(
            logical_xor,
            other,
            out=numeric.IntegralMatrix,
        )

    __rxor__ = __xor__

    def __or__(self, other):
        """Element-wise logical OR"""
        return self.binary_operator(
            logical_or,
            other,
            out=numeric.IntegralMatrix,
        )

    __ror__ = __or__

    def __invert__(self):
        """Element-wise logical NOT"""
        return self.unary_operator(
            logical_not,
            out=numeric.IntegralMatrix,
        )

    def __getitem__(self, key):
        """Return the element or sub-matrix corresponding to `key`

        If `key` is an integer or slice, it is treated as if it is indexing the
        flattened matrix, returning the corresponding value(s). A slice will
        always return a wrap matrix of shape `(1, N)`, where `N` is the length
        of the slice's range.

        If `key` is a tuple, the first index is applied against the rows, while
        the second is applied against the columns. A tuple of two integers,
        `(i, j)`, will return the element at row `i`, column `j`. All other
        tuple variations will return a wrap sub-matrix of shape `(M, N)`, where
        `M` is the length of the first slice's range, and `N` is the length of
        the second slice's range - integers are treated as length 1 slices if
        mixed with at least one other slice.
        """
        data = self.data
        h = self.shape

        if isinstance(key, tuple):

            def get(keys, k):
                return type(self).wrap(
                    [data[i * h[1] + j] for i, j in keys],
                    shape=k,
                )

            rowkey, colkey = key

            if isinstance(rowkey, slice):
                ix = h.resolve_slice(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = h.resolve_slice(colkey, by=Rule.COL)
                    other = get(itertools.product(ix, jx), k=Shape(len(ix), len(jx)))

                else:
                    j = h.resolve_index(colkey, by=Rule.COL)
                    other = get(zip(ix, itertools.repeat(j)), k=Shape(len(ix), 1))

            else:
                i = h.resolve_index(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = h.resolve_slice(colkey, by=Rule.COL)
                    other = get(zip(itertools.repeat(i), jx), k=Shape(1, len(jx)))

                else:
                    j = h.resolve_index(colkey, by=Rule.COL)
                    other = data[i * h[1] + j]

            return other

        if isinstance(key, slice):

            def get(keys, k):
                return type(self).wrap(
                    [data[i] for i in keys],
                    shape=k,
                )

            ix = range(*key.indices(h.size))
            other = get(ix, k=Shape(1, len(ix)))

            return other

        try:
            other = data[key]
        except IndexError:
            raise IndexError("index out of range") from None
        else:
            return other

    def __setitem__(self, key, other):
        """Overwrite the element or sub-matrix corresponding to `key`

        If `key` is an integer or slice, it is treated as if it is indexing the
        flattened matrix, overwriting the corresponding value(s).

        If `key` is a tuple, the first index is applied against the rows, while
        the second is applied against the columns. A tuple of two integers,
        `(i, j)`, will overwrite the element at row `i`, column `j`. All other
        tuple variations will overwrite a sub-matrix of shape `(M, N)`, where
        `M` is the length of the first slice's range, and `N` is the length of
        the second slice's range - integers are treated as length 1 slices if
        mixed with at least one other slice.
        """
        data = self.data
        h = self.shape

        if isinstance(key, tuple):

            def set(keys, k):
                k.compatible(other.shape, error=True)
                for (i, j), x in zip(keys, other):
                    data[i * h[1] + j] = x

            rowkey, colkey = key

            if isinstance(rowkey, slice):
                ix = h.resolve_slice(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = h.resolve_slice(colkey, by=Rule.COL)
                    set(itertools.product(ix, jx), k=Shape(len(ix), len(jx)))

                else:
                    j = h.resolve_index(colkey, by=Rule.COL)
                    set(zip(ix, itertools.repeat(j)), k=Shape(len(ix), 1))

            else:
                i = h.resolve_index(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = h.resolve_slice(colkey, by=Rule.COL)
                    set(zip(itertools.repeat(i), jx), k=Shape(1, len(jx)))

                else:
                    j = h.resolve_index(colkey, by=Rule.COL)
                    data[i * h[1] + j] = other

            return

        if isinstance(key, slice):

            def set(keys, k):
                k.compatible(other.shape, error=True)
                for i, x in zip(keys, other):
                    data[i] = x

            ix = range(*key.indices(h.size))
            set(ix, k=Shape(1, len(ix)))

            return

        try:
            data[key] = other
        except IndexError:
            raise IndexError("index out of range") from None
        else:
            return

    def __len__(self):
        """Return the matrix's size"""
        return len(self.data)

    def __iter__(self):
        """Return an iterator over the elements of the matrix"""
        yield from iter(self.data)

    def __reversed__(self):
        """Return a reverse iterator over the elements of the matrix"""
        yield from reversed(self.data)

    def __contains__(self, value):
        """Return true if the matrix contains `value`, otherwise false"""
        return value in self.data

    def __copy__(self):
        """Return a shallow copy of the matrix"""
        data = self.data
        h = self.shape
        return type(self).wrap(copy.copy(data), shape=h.copy())

    def __deepcopy__(self, memo=None):
        """Return a deep copy of the matrix"""
        data = self.data
        h = self.shape
        return type(self).wrap(copy.deepcopy(data, memo), shape=h.copy())

    @property
    def size(self):
        """The product of the number of rows and columns"""
        return self.shape.size

    @property
    def nrows(self):
        """The matrix's number of rows"""
        return self.shape[0]

    @property
    def ncols(self):
        """The matrix's number of columns"""
        return self.shape[1]

    def index(self, value, start=0, stop=None):
        """Return the index of the first element equal to `value`

        Raises `ValueError` if the value could not be found in the matrix.
        """
        try:
            index = super().index(value, start, stop)
        except ValueError:
            raise ValueError("value not found") from None
        else:
            return index

    def count(self, value):
        """Return the number of times `value` appears in the matrix"""
        return super().count(value)

    def bool(self):
        """Element-wise boolean conversion"""
        return self.unary_operator(bool, out=numeric.IntegralMatrix)

    def reshape(self, nrows, ncols):
        """Re-interpret the matrix's shape

        Raises `ValueError` if any of the given dimensions are negative, or if
        their product does not equal the matrix's current size.
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("dimensions must be non-negative")
        h = self.shape
        if (n := h.size) != (nrows * ncols):
            raise ValueError(f"cannot re-shape size {n} matrix as shape {nrows} × {ncols}")
        h[0], h[1] = nrows, ncols
        return self

    def slices(self, *, by=Rule.ROW):
        """Return an iterator that yields shallow copies of each row or column"""
        cls = type(self)
        h = self.shape
        k = by.subshape(h)
        for i in range(h[by]):
            data = self.data[by.slice(i, h)]
            yield cls.wrap(data, shape=k.copy())

    def mask(self, selector, null):
        """Replace the elements who have a true parallel value in `selector`
        with `null`

        Raises `ValueError` if the selector differs in size.
        """
        data = self.data
        h = self.shape
        h.compatible(selector.shape, error=True)
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
        ix = (i for i, x in enumerate(self) if x is old or x == old)
        for i in itertools.islice(ix, times):
            data[i] = new
        return self

    def reverse(self):
        """Reverse the matrix's elements in place"""
        data = self.data
        data.reverse()
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
        for ii, jj in zip(by.range(i, h), by.range(j, h)):
            data[ii], data[jj] = data[jj], data[ii]
        return self

    def flip(self, *, by=Rule.ROW):
        """Reverse the matrix's rows or columns in place"""
        data = self.data
        h = self.shape
        n = h[by]
        for i in range(n // 2):
            j = n - i - 1
            for ii, jj in zip(by.range(i, h), by.range(j, h)):
                data[ii], data[jj] = data[jj], data[ii]
        return self

    def flatten(self, *, by=Rule.ROW):
        """Re-shape the matrix to a row or column vector

        If flattened to a column vector, the elements are arranged into
        column-major order.
        """
        if by is Rule.COL: self.transpose()  # For column-major order
        h = self.shape
        h[~by] = h.size
        h[by] = 1
        return self

    def transpose(self):
        """Transpose the matrix in place"""
        h = self.shape
        if (m := h[0]) > 1 and (n := h[1]) > 1:
            ix, jx = range(m), range(n)
            data = self.data
            data[:] = (data[i * n + j] for j in jx for i in ix)
        h.reverse()
        return self

    def stack(self, other, *, by=Rule.ROW):
        """Stack a matrix along the rows or columns

        Raises `ValueError` if the inverse dimension differs between the
        operand matrices.
        """
        if self is other: other = other.copy()

        data = self.data
        h, k = self.shape, other.shape

        dy = ~by
        if (m := h[dy]) != (n := k[dy]):
            name = dy.true_name
            raise ValueError(f"operating matrix has {m} {name}s but operand has {n}")

        (m, n), (_, q) = (h, k)

        if by is Rule.COL and m > 1:
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
        h = self.shape
        s = by.slice(h.resolve_index(key, by=by), h)

        data = self.data[s]
        del self.data[s]

        h[by] -= 1

        return type(self).wrap(data, shape=by.subshape(h))

    def copy(self, *, deep=False):
        """Return a shallow or deep copy of the matrix"""
        return copy.deepcopy(self) if deep else copy.copy(self)
