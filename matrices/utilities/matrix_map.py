from typing import TypeVar

from ..abc import ShapedCollection

__all__ = ["MatrixMap"]

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class MatrixMap(ShapedCollection[T_co, M_co, N_co]):
    """A ``ShapedCollection`` wrapper for mapping processes on ``MatrixLike``
    objects

    Note that elements of the collection are computed lazily, and without
    caching. ``__iter__()`` (or any of its dependents) will compute/re-compute
    the mapping for each call.
    """

    __slots__ = ("_func", "_args", "_shape")

    def __init__(self, func, matrix1, /, *matrices):
        shape1 = matrix1.shape
        for matrix2 in matrices:
            shape2 = matrix2.shape
            if shape1 != shape2:
                raise ValueError(f"incompatible shapes {shape1}, {shape2}")
        self._func  = func
        self._args  = (matrix1, *matrices)
        self._shape = shape1

    def __iter__(self):
        yield from map(self._func, *self._args)

    @property
    def shape(self):
        return self._shape
