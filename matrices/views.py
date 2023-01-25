import operator
from reprlib import recursive_repr
from typing import TypeVar

from . import FrozenMatrix
from .abc import MatrixLike
from .shapes import Shape
from .utilities import COL, ROW, Rule, checked_map

__all__ = ["MatrixView"]

T = TypeVar("T")
M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


class MatrixView(MatrixLike[T, M, N]):

    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    @recursive_repr("...")
    def __repr__(self):
        """Return a canonical representation of the view"""
        return f"{self.__class__.__name__}(target={self._target!r})"

    def __getitem__(self, key):
        return self._target[key]

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

    def conjugate(self):
        return self._target.conjugate()

    def transpose(self):  # TODO
        pass


# class MatrixTranspose(MatrixView[T, M, N]):

#     __slots__ = ()

#     def __getitem__(self, key):
#         target = self._target
#         shape  = self.shape

#         if isinstance(key, tuple):
#             row_key, col_key = key

#             if isinstance(row_key, slice):
#                 ix = shape.resolve_slice(row_key, by=ROW)

#                 if isinstance(col_key, slice):
#                     jx = shape.resolve_slice(col_key, by=COL)
#                     n  = shape.nrows
#                     result = FrozenMatrix.wrap(
#                         [target[j * n + i] for i in ix for j in jx],
#                         shape=Shape(len(ix), len(jx)),
#                     )

#                 else:
#                     j = shape.resolve_index(col_key, by=COL)
#                     n = shape.nrows
#                     result = FrozenMatrix.wrap(
#                         [target[j * n + i] for i in ix],
#                         shape=Shape(len(ix), 1),
#                     )

#             else:
#                 i = shape.resolve_index(row_key, by=ROW)

#                 if isinstance(col_key, slice):
#                     jx = shape.resolve_slice(col_key, by=COL)
#                     n  = shape.nrows
#                     result = FrozenMatrix.wrap(
#                         [target[j * n + i] for j in jx],
#                         shape=Shape(1, len(jx)),
#                     )

#                 else:
#                     j = shape.resolve_index(col_key, by=COL)
#                     n = shape.nrows
#                     result = target[j * n + i]

#         elif isinstance(key, slice):
#             ix = range(*key.indices(shape.nrows * shape.ncols))

#             result = FrozenMatrix.wrap(
#                 [target[self.permute_index(i)] for i in ix],
#                 shape=Shape(1, len(ix)),
#             )

#         else:
#             result = target[self.permute_index(key)]

#         return result

#     def __iter__(self, *, iter=iter):
#         target = self._target
#         nrows, ncols = target.shape
#         ix = range(nrows)
#         jx = range(ncols)
#         for j in iter(jx):
#             for i in iter(ix):
#                 yield target[i * ncols + j]

#     def __reversed__(self):
#         yield from self.__iter__(iter=reversed)

#     @property
#     def shape(self):
#         target = self._target
#         return Shape(
#             nrows=target.ncols,
#             ncols=target.nrows,
#         )

#     @property
#     def nrows(self):
#         return self._target.ncols

#     @property
#     def ncols(self):
#         return self._target.nrows

#     def transpose(self):
#         return MatrixView(self._target)

#     def permute_index(self, key):
#         """Return an index `key` as its transposed equivalent with respect to
#         the target matrix

#         Raises `IndexError` if the key is out of range.
#         """
#         nrows, ncols = self._target.shape
#         n = nrows * ncols
#         i = operator.index(key)
#         i += n * (i < 0)
#         if i < 0 or i >= n:
#             raise IndexError(f"there are {n} items but index is {key}")
#         j = n - 1
#         return i if i == j else (i * ncols) % j
