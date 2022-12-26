import operator
from typing import TypeVar

from .abstract import MatrixLike
from .utilities import Rule

__all__ = ["MatrixView"]

T = TypeVar("T")
M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


class MatrixView(MatrixLike[T, M, N]):

    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rowkey, colkey = key
            if isinstance(rowkey, slice):
                if isinstance(colkey, slice):
                    result = self._target[key]
                else:
                    colkey = operator.index(colkey)
                    result = self._target[rowkey, colkey]
            else:
                if isinstance(colkey, slice):
                    rowkey = operator.index(rowkey)
                    result = self._target[rowkey, colkey]
                else:
                    rowkey = operator.index(rowkey)
                    colkey = operator.index(colkey)
                    result = self._target[rowkey, colkey]
        elif isinstance(key, slice):
            result = self._target[key]
        else:
            key = operator.index(key)
            result = self._target[key]
        return result

    def __iter__(self):
        yield from self._target

    def __reversed__(self):
        yield from reversed(self._target)

    def __contains__(self, value):
        return value in self._target

    def __add__(self, other):
        return self._target + other

    def __sub__(self, other):
        return self._target - other

    def __mul__(self, other):
        return self._target * other

    def __truediv__(self, other):
        return self._target / other

    def __floordiv__(self, other):
        return self._target // other

    def __mod__(self, other):
        return self._target % other

    def __divmod__(self, other):
        return divmod(self._target, other)

    def __pow__(self, other):
        return self._target ** other

    def __lshift__(self, other):
        return self._target << other

    def __rshift__(self, other):
        return self._target >> other

    def __and__(self, other):
        return self._target & other

    def __xor__(self, other):
        return self._target ^ other

    def __or__(self, other):
        return self._target | other

    def __matmul__(self, other):
        return self._target @ other

    def __neg__(self):
        return -self._target

    def __pos__(self):
        return +self._target

    def __abs__(self):
        return abs(self._target)

    def __invert__(self):
        return ~self._target

    @property
    def shape(self):
        return self._target.shape

    @property
    def nrows(self):
        return self._target.nrows

    @property
    def ncols(self):
        return self._target.ncols

    @property
    def size(self):
        return self._target.size

    def equal(self, other):
        return self._target.equal(other)

    def not_equal(self, other):
        return self._target.not_equal(other)

    def lesser(self, other):
        return self._target.lesser(other)

    def lesser_equal(self, other):
        return self._target.lesser_equal(other)

    def greater(self, other):
        return self._target.greater(other)

    def greater_equal(self, other):
        return self._target.greater_equal(other)

    def logical_and(self, other):
        return self._target.logical_and(other)

    def logical_or(self, other):
        return self._target.logical_or(other)

    def logical_not(self):
        return self._target.logical_not()

    def slices(self, *, by=Rule.ROW):
        yield from self._target.slices(by=by)

    def transpose(self):
        return self._target.transpose()
