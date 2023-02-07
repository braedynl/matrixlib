import copy
from array import array as Array
from typing import TypeVar

from ..rule import Rule
from .abc import ShapeLike

__all__ = ["Shape"]

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


class Shape(ShapeLike[M, N]):
    """A mutable ``ShapeLike`` intended for storing and manipulating dimensions
    within a matrix class implementation
    """

    # Implementation notes:
    #
    # This is essentially just a wrapper around an `array.array` instance. The
    # methods `__iter__()`, `__reversed__()`, and `__contains__()` all call the
    # parallel method of the composed instance to save on time costs (as
    # opposed to going through its own `__getitem__()` implementation).
    #
    # An array is used for two reasons:
    # - Arrays consume less memory
    # - Checks for negative values are performed faster when constrained to an
    #   unsigned integer type (since the check happens in C)
    #
    # The biggest drawback to using an array is slower access time, since the
    # `int` instance has to be recreated when asked for. The difference is
    # considered to be marginal, however.
    #
    # We use an unsigned long long (typecode "Q"), since the API doesn't
    # provide a size type. It's a bit overkill, but still consumes less memory
    # than an equivalent `list` instance.

    __slots__ = ("_array",)

    def __init__(self, nrows, ncols):
        """Construct a shape from its dimensions"""
        self._array = Array("Q", (nrows, ncols))

    def __repr__(self):
        """Return a canonical representation of the shape"""
        array = self._array
        return f"{self.__class__.__name__}(nrows={array[0]!r}, ncols={array[1]!r})"

    def __getitem__(self, key):
        try:
            value = self._array[key]
        except IndexError as error:
            raise IndexError("shape index out of range") from error
        else:
            return value

    def __setitem__(self, key, value):
        """Set the dimension corresponding to ``key`` with ``value``"""
        try:
            self._array[key] = value
        except IndexError as error:
            raise IndexError("shape index out of range") from error

    def __iter__(self):
        yield from iter(self._array)

    def __reversed__(self):
        yield from reversed(self._array)

    def __contains__(self, value):
        return value in self._array

    def __deepcopy__(self, memo=None):
        """Return a copy of the shape"""
        cls = self.__class__

        new = cls.__new__(cls)
        new._array = copy.copy(self._array)  # We're holding ints - no need to deep copy

        return new

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
