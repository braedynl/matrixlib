from typing import TypeVar

from ..utilities import Rule
from .abc import *

__all__ = ["ShapeLike", "Shape"]

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


class Shape(ShapeLike[M, N]):
    """A mutable `ShapeLike` intended for storing and manipulating dimensions
    within a matrix class implementation
    """

    # Implementation notes:
    # This is essentially just a wrapper around a built-in `list` instance. The
    # methods `__iter__()`, `__reversed__()`, and `__contains__()` all call the
    # parallel method of the composed `list` instance to save on time costs (as
    # opposed to going through its own `__getitem__()` implementation).

    __slots__ = ("_array",)

    def __init__(self, nrows, ncols):
        """Construct a shape from its two dimensions

        Raises `ValueError` if either of the two dimensions are negative.
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("dimensions must be non-negative")
        self._array = [nrows, ncols]

    def __repr__(self):
        """Return a canonical representation of the shape"""
        return f"{self.__class__.__name__}(nrows={self._array[0]!r}, ncols={self._array[1]!r})"

    def __getitem__(self, key):
        try:
            value = self._array[key]
        except IndexError as error:
            raise IndexError("index out of range") from error
        else:
            return value

    def __setitem__(self, key, value):
        """Set the dimension corresponding to `key` with `value`"""
        if value < 0:
            raise ValueError("dimensions must be non-negative")
        try:
            self._array[key] = value
        except IndexError as error:
            raise IndexError("index out of range") from error

    def __iter__(self):
        yield from iter(self._array)

    def __reversed__(self):
        yield from reversed(self._array)

    def __contains__(self, value):
        return value in self._array

    def __deepcopy__(self, memo=None):
        """Return a copy of the shape"""
        cls = self.__class__

        copy = cls.__new__(cls)
        copy._array = self._array.copy()

        return copy

    copy = __copy__ = __deepcopy__

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

    def subshape(self, *, by=Rule.ROW):
        """Return the shape of any sub-matrix in the given rule's form"""
        shape = self.copy()
        shape[by.value] = 1
        return shape

    def sequence(self, index, *, by=Rule.ROW):
        """Return the start, stop, and step values required to create a range
        or slice object of the given rule's shape beginning at `index`

        The input `index` must be positive - negative indices may produce
        unexpected results. This requirement is not checked for.
        """
        dy = ~by

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
