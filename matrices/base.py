import copy
import itertools
import operator
import reprlib
from collections.abc import Sequence
from enum import Enum
from io import StringIO

__all__ = ["Rule", "Shape", "BaseMatrix"]


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


def binary_reverse(func):
    """Return a wrapper of a given binary function that reverses its incoming
    arguments
    """
    def wrapper(x, y):
        return func(y, x)
    return wrapper


def logical_and(a, b, /): return not not (a and b)
def logical_or(a, b, /): return not not (a or b)
def logical_xor(a, b, /): return (not not a) is not (not not b)
def logical_not(a, /): return not a


class BaseMatrix(Sequence):
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
        if (n := len(data)) != shape.size:
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
        return f"Matrix({self.data!r}, nrows={self.nrows!r}, ncols={self.ncols!r})"

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
        shape = self.shape
        items = iter(self)

        max_width = 10
        res = StringIO()

        if shape.size:
            for _ in range(shape.nrows):
                res.write("| ")

                for _ in range(shape.ncols):
                    chars = str(next(items))
                    if len(chars) > max_width:
                        res.write(f"{chars[:max_width - 1]}…")
                    else:
                        res.write(chars.rjust(max_width))
                    res.write(" ")

                res.write("|\n")
        else:
            res.write("Empty matrix ")

        res.write(f"({shape})")

        return res.getvalue()

    def map(self, func, other=None):
        if other is not None:
            if not isinstance(other, BaseMatrix):
                return NotImplemented
            if (m := self.size) != (n := other.size):
                raise ValueError(f"operating matrix has size {m} but operand has size {n}")
            values = map(func, self, other)
        else:
            values = map(func, self)
        return list(values)

    def __eq__(self, other):
        """Element-wise equals"""
        data = self.map(operator.eq, other)
        if data is NotImplemented:
            return data
        shape = self.shape
        return BaseMatrix.wrap(data, shape=shape.copy())

    def __ne__(self, other):
        """Element-wise not equals"""
        data = self.map(operator.ne, other)
        if data is NotImplemented:
            return data
        shape = self.shape
        return BaseMatrix.wrap(data, shape=shape.copy())

    def __len__(self):
        """Return the matrix's size"""
        return len(self.data)

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
        shape = self.shape

        if isinstance(key, tuple):
            rowkey, colkey = key

            w = shape.ncols
            def getitems(indices, nrows, ncols):
                data = [self.data[i * w + j] for i, j in indices]
                return type(self).wrap(data, shape=Shape(nrows, ncols))

            if isinstance(rowkey, slice):
                ix = shape.resolve_slice(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = shape.resolve_slice(colkey, by=Rule.COL)
                    return getitems(
                        indices=itertools.product(ix, jx),
                        nrows=len(ix),
                        ncols=len(jx),
                    )

                else:
                    j = shape.resolve_index(colkey, by=Rule.COL)
                    return getitems(
                        indices=zip(ix, itertools.repeat(j)),
                        nrows=len(ix),
                        ncols=1,
                    )

            else:
                i = shape.resolve_index(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = shape.resolve_slice(colkey, by=Rule.COL)
                    return getitems(
                        indices=zip(itertools.repeat(i), jx),
                        nrows=1,
                        ncols=len(jx),
                    )

                else:
                    j = shape.resolve_index(colkey, by=Rule.COL)
                    return self.data[i * w + j]

        if isinstance(key, slice):
            ix = range(*key.indices(shape.size))

            data = [self.data[i] for i in ix]
            return type(self).wrap(data, shape=Shape(1, len(ix)))

        try:
            value = self.data[key]
        except IndexError:
            raise IndexError("index out of range") from None
        else:
            return value

    def __setitem__(self, key, value):
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
        shape = self.shape

        if isinstance(key, tuple):
            rowkey, colkey = key

            w = shape.ncols
            def setitems(indices, nrows, ncols):
                if (m := nrows * ncols) != (n := len(value)):
                    raise ValueError(f"slice selected {m} items but sequence has length {n}")
                for (i, j), x in zip(indices, value):
                    self.data[i * w + j] = x

            if isinstance(rowkey, slice):
                ix = shape.resolve_slice(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = shape.resolve_slice(colkey, by=Rule.COL)
                    return setitems(
                        indices=itertools.product(ix, jx),
                        nrows=len(ix),
                        ncols=len(jx),
                    )

                else:
                    j = shape.resolve_index(colkey, by=Rule.COL)
                    return setitems(
                        indices=zip(ix, itertools.repeat(j)),
                        nrows=len(ix),
                        ncols=1,
                    )

            else:
                i = shape.resolve_index(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = shape.resolve_slice(colkey, by=Rule.COL)
                    return setitems(
                        indices=zip(itertools.repeat(i), jx),
                        nrows=1,
                        ncols=len(jx),
                    )

                else:
                    j = shape.resolve_index(colkey, by=Rule.COL)
                    self.data[i * w + j] = value
                    return

        if isinstance(key, slice):
            ix = range(*key.indices(shape.size))

            if (m := len(ix)) != (n := len(value)):
                raise ValueError(f"slice selected {m} items but sequence has length {n}")
            for i, x in zip(ix, value):
                self.data[i] = x

            return

        try:
            self.data[key] = value
        except IndexError:
            raise IndexError("index out of range") from None
        else:
            return

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
        data = copy.copy(self.data)
        shape = self.shape
        return type(self).wrap(data, shape=shape.copy())

    def __deepcopy__(self, memo=None):
        """Return a deep copy of the matrix"""
        data = copy.deepcopy(self.data, memo)
        shape = self.shape
        return type(self).wrap(data, shape=shape.copy())

    @property
    def size(self):
        """The product of the number of rows and columns"""
        return self.shape.size

    @property
    def nrows(self):
        """The matrix's number of rows"""
        return self.shape.nrows

    @property
    def ncols(self):
        """The matrix's number of columns"""
        return self.shape.ncols

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

    def reshape(self, nrows, ncols):
        """Re-interpret the matrix's shape

        Raises `ValueError` if any of the given dimensions are negative, or if
        their product does not equal the matrix's current size.
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("dimensions must be non-negative")
        shape = self.shape
        if (n := shape.size) != nrows * ncols:
            raise ValueError(f"cannot re-shape size {n} matrix as shape {nrows} × {ncols}")
        shape.nrows = nrows
        shape.ncols = ncols
        return self

    def slices(self, *, by=Rule.ROW):
        """Return an iterator that yields shallow copies of each row or column"""
        shape = self.shape
        subshape = by.subshape(shape)
        for i in range(shape[by]):
            data = self.data[by.slice(i, shape)]
            yield type(self).wrap(data, shape=subshape.copy())

    def mask(self, selector, null):
        """Replace the elements who have a true parallel value in `selector`
        with `null`

        Raises `ValueError` if the selector differs in size.
        """
        if (m := self.size) != (n := selector.size):
            raise ValueError(f"operating matrix has size {m} but selector has size {n}")
        for i, masked in enumerate(selector):
            if masked: self.data[i] = null
        return self

    def replace(self, old, wrap, *, times=None):
        """Replace elements equal to `old` with `wrap`

        If `times` is given, only the first `times` occurrences of `old` will
        be replaced.

        This method considers two objects equal if a comparison by identity or
        equality is satisfied, which can sometimes be helpful for replacing
        objects such as `math.nan`.
        """
        ix = (i for i, x in enumerate(self) if x is old or x == old)
        for i in itertools.islice(ix, times):
            self.data[i] = wrap
        return self

    def reverse(self):
        """Reverse the matrix's elements in place"""
        self.data.reverse()
        return self

    def swap(self, key1, key2, *, by=Rule.ROW):
        """Swap the two rows or columns beginning at the given indices

        Note that, due to how `Matrix` stores its data, swapping is performed
        in linear time with respect to the specified dimension.
        """
        data, shape = self.data, self.shape
        i = shape.resolve_index(key1, by=by)
        j = shape.resolve_index(key2, by=by)
        for h, k in zip(by.range(i, shape), by.range(j, shape)):
            data[h], data[k] = data[k], data[h]
        return self

    def flip(self, *, by=Rule.ROW):
        """Reverse the matrix's rows or columns in place"""
        data, shape = self.data, self.shape
        n = shape[by]
        for i in range(n // 2):
            j = n - i - 1
            for h, k in zip(by.range(i, shape), by.range(j, shape)):
                data[h], data[k] = data[k], data[h]
        return self

    def flatten(self, *, by=Rule.ROW):
        """Re-shape the matrix to a row or column vector

        If flattened to a column vector, the elements are arranged into
        column-major order.
        """
        if by is Rule.COL: self.transpose()  # For column-major order
        shape = self.shape
        shape[by.inverse] = shape.size
        shape[by] = 1
        return self

    def transpose(self):
        """Transpose the matrix in place"""
        shape = self.shape
        if (nrows := shape[0]) > 1 and (ncols := shape[1]) > 1:
            data = self.data
            ix, jx = range(nrows), range(ncols)
            data[:] = (data[i * ncols + j] for j in jx for i in ix)
        shape.reverse()
        return self

    def stack(self, other, *, by=Rule.ROW):
        """Stack a sequence or other matrix along the rows or columns

        If `other` is a sequence type, but not a matrix, it will be interpreted
        as a vector.

        Raises `ValueError` if the inverse dimension differs between the
        operand matrices.
        """
        if self is other: other = other.copy()  # type: ignore[attr-defined]

        dy = by.inverse

        shape = self.shape
        if isinstance(other, BaseMatrix):
            other_shape = other.shape
        else:
            other_shape = Shape()
            other_shape[by] = 1
            other_shape[dy] = len(other)

        if (m := shape[dy]) != (n := other_shape[dy]):
            name = dy.true_name
            raise ValueError(f"operating matrix has {m} {name}s but operand has {n}")

        (m, n), (_, q) = (shape, other_shape)

        if by is Rule.COL and m > 1:
            left, right = iter(self), iter(other)
            self.data[:] = (
                x
                for _ in range(m)
                for x in itertools.chain(itertools.islice(left, n), itertools.islice(right, q))
            )
        else:
            self.data.extend(other)

        shape[by] += other_shape[by]

        return self

    def pull(self, key=-1, *, by=Rule.ROW):
        """Remove and return the row or column corresponding to `key`

        Raises `IndexError` if the matrix is empty, or if the index is out of
        range.
        """
        shape = self.shape

        slice = by.slice(shape.resolve_index(key, by=by), shape)

        data = self.data[slice]
        del self.data[slice]

        shape[by] -= 1

        return type(self).wrap(data, shape=by.subshape(shape))

    def copy(self, *, deep=False):
        """Return a shallow or deep copy of the matrix"""
        return copy.deepcopy(self) if deep else copy.copy(self)
