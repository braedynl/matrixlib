from collections.abc import Collection
from typing import TypeVar

from ..abc import ShapedIterable

__all__ = ["MatrixMap"]

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class MatrixMap(Collection[T_co], ShapedIterable[T_co, M_co, N_co]):
    """A shaped, collection wrapper for mapping processes on ``MatrixLike``
    objects

    Note that elements of the collection are computed lazily, and without
    caching. ``__iter__()`` (or any of its dependents) will compute/re-compute
    the mapping for each call.

    ``MatrixMap`` is a type of ``ShapedIterable``, which can be passed directly
    to a built-in ``Matrix`` constructor.
    """

    __slots__ = ("_func", "_matrices")

    def __init__(self, func, matrix1, /, *matrices):
        u = matrix1.shape
        for matrix2 in matrices:
            v = matrix2.shape
            if u != v:
                raise ValueError(f"incompatible shapes {u}, {v}")
        self._func = func
        self._matrices = (matrix1, *matrices)

    def __iter__(self):
        yield from map(self.func *self.matrices)

    def __contains__(self, value):
        return any(map(lambda x: x is value or x == value, self))

    @property
    def shape(self):
        return self.matrices[0].shape

    @property
    def func(self):
        """The mapping function"""
        return self._func

    @property
    def matrices(self):
        """A ``tuple`` of the matrices used in the mapping process

        This tuple is guaranteed to contain at least one matrix.
        """
        return self._matrices
