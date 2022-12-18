import copy
import operator
from typing import TypeVar

from ..utilities import Rule
from .abstract import ShapeLike

__all__ = ["Shape"]

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


class Shape(ShapeLike[M, N]):
    """A mutable collection type for storing matrix dimensions"""

    __slots__ = ("_data",)

    def __init__(self, nrows, ncols):
        """Construct a shape from its two dimensions

        Raises `ValueError` if either of the two dimensions are negative.
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("dimensions must be non-negative")
        self._data = [nrows, ncols]

    def __repr__(self):
        """Return a canonical representation of the shape"""
        nrows, ncols = self
        return f"{self.__class__.__name__}(nrows={nrows!r}, ncols={ncols!r})"

    def __str__(self):
        """Return a string representation of the shape"""
        nrows, ncols = self
        return f"{nrows} × {ncols}"

    def __getitem__(self, key):
        try:
            value = self._data[key]
        except IndexError as error:
            raise IndexError("index out of range") from error
        else:
            return value

    def __setitem__(self, key, value):
        """Set the dimension corresponding to `key` with `value`"""
        if value < 0:
            raise ValueError("dimensions must be non-negative")
        try:
            self._data[key] = value
        except IndexError as error:
            raise IndexError("index out of range") from error

    def __iter__(self):
        yield from self._data

    def __reversed__(self):
        yield from reversed(self._data)

    def __contains__(self, value):
        return value in self._data

    def __deepcopy__(self, memo=None):
        """Return a copy of the shape"""
        cls = self.__class__

        copy = cls.__new__(cls)
        copy._data = self._data.copy()  # Our components are (hopefully) immutable

        return copy

    __copy__ = __deepcopy__

    @property
    def nrows(self):
        return self[0]

    @nrows.setter
    def nrows(self, value):
        self[0] = value

    @property
    def ncols(self):
        return self[1]

    @ncols.setter
    def ncols(self, value):
        self[1] = value

    def reverse(self):
        """Reverse the shape's dimensions in place"""
        self._data.reverse()
        return self

    def copy(self):
        """Return a copy of the shape"""
        return copy.deepcopy(self)

    def subshape(self, *, by=Rule.ROW):
        """Return the shape of any sub-matrix in the given rule's form"""
        shape = self.copy()
        shape[by.value] = 1
        return shape

    def resolve_index(self, key, *, by=Rule.ROW):
        """Return an index `key` as an equivalent integer, respective to a rule

        Raises `IndexError` if the key is out of range.
        """
        n = self[by.value]
        i = operator.index(key)
        i += n * (i < 0)
        if i < 0 or i >= n:
            handle = by.handle()
            raise IndexError(f"there are {n} {handle}s but index is {key}")
        return i

    def resolve_slice(self, key, *, by=Rule.ROW):
        """Return a slice `key` as an equivalent sequence of indices,
        respective to a rule
        """
        n = self[by.value]
        return range(*key.indices(n))

    def sequence(self, index, *, by=Rule.ROW):
        """Return the start, stop, and step values required to create a range
        or slice object of the given rule's shape beginning at `index`

        The input `index` must be positive - negative indices may produce
        unexpected results. This requirement is not checked for.
        """
        dy = by.invert()

        i = by.value
        j = dy.value

        major = self[i]
        minor = self[j]

        major_step = i * major + j
        minor_step = j * minor + i

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
