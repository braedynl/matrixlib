import operator
from collections.abc import Collection

from .protocols import ShapeLike
from .rule import Rule

__all__ = ["Shape"]


class Shape(Collection):
    """A mutable collection type for storing matrix dimensions

    Instances of `Shape` support integer - but not slice - indexing. Negative
    values are accepted by writing operations, and is left up to the matrix
    implementation to consider.

    Shapes should not be written-to when exposed by a matrix object, unless
    stated otherwise by the matrix class' documentation.
    """

    __slots__ = ("data",)

    def __init__(self, nrows=0, ncols=0):
        """Construct a shape from its two dimensions"""
        self.data = [nrows, ncols]

    def __repr__(self):
        """Return a canonical representation of the shape"""
        nrows, ncols = self.data
        return f"Shape(nrows={nrows!r}, ncols={ncols!r})"

    def __str__(self):
        """Return a string representation of the shape"""
        nrows, ncols = self.data
        return f"{nrows} × {ncols}"

    def __eq__(self, other):
        """Return true if the two shapes are equal, otherwise false"""
        if not isinstance(other, ShapeLike):
            return NotImplemented
        nrows, ncols = self.data
        return nrows == other[0] and ncols == other[1]

    def __len__(self):
        """Return literal 2"""
        return 2

    def __getitem__(self, key):
        """Return the dimension corresponding to `key`"""
        key = operator.index(key)
        try:
            value = self.data[key]
        except IndexError:
            raise IndexError("index out of range") from None
        else:
            return value

    def __setitem__(self, key, value):
        """Set the dimension corresponding to `key` with `value`"""
        key = operator.index(key)
        try:
            self.data[key] = value
        except IndexError:
            raise IndexError("index out of range") from None

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
        return Shape(*self.data)  # Our components are (hopefully) immutable

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

    def copy(self):
        """Return a copy of the shape"""
        return Shape(*self.data)

    def reverse(self):
        """Reverse the shape's dimensions in place"""
        self.data.reverse()
        return self

    def subshape(self, *, by=Rule.ROW):
        """Return the shape of any sub-matrix in the given rule's form"""
        shape = self.copy()
        shape.data[by] = 1
        return shape

    def resolve_index(self, key, *, by=Rule.ROW):
        """Return an index `key` as an equivalent integer, respective to a rule

        Raises `IndexError` if the key is out of range.
        """
        n = self.data[by]
        i = operator.index(key)
        i += n * (i < 0)
        if i < 0 or i >= n:
            raise IndexError(f"there are {n} {by.handle}s but index is {key}")
        return i

    def resolve_slice(self, key, *, by=Rule.ROW):
        """Return a slice `key` as an equivalent sequence of indices,
        respective to a rule
        """
        n = self.data[by]
        return range(*key.indices(n))

    def sequence(self, index, *, by=Rule.ROW):
        """Return the start, stop, and step values required to create a range
        or slice object of the given rule's shape beginning at `index`

        The input `index` must be positive - negative indices may produce
        unexpected results. This requirement is not checked for.
        """
        data = self.data
        dy = not by

        major = data[by]
        minor = data[dy]

        major_step = by * major + dy
        minor_step = dy * minor + by

        start = minor_step * index
        stop  = major_step * minor + start

        return start, stop, major_step

    def range(self, index, *, by=Rule.ROW):
        """Return a range of indices that can be used to construct a sub-matrix
        of the rule's shape beginning at `index`
        """
        return range(*self.sequence(index, by=by))

    def slice(self, index, *, by=Rule.ROW):
        """Return a slice that can be used to construct a sub-matrix of the
        rule's shape beginning at `index`
        """
        return slice(*self.sequence(index, by=by))
