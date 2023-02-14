from collections.abc import Collection
from typing import TypeVar

from ..abc import ShapedIterable

__all__ = ["CheckedMap", "vectorize"]

T = TypeVar("T")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class CheckedMap(Collection[T_co], ShapedIterable[T_co, M_co, N_co]):

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


def vectorize():
    """Convert a scalar-based function into a matrix-based one

    For a function, ``f()``, whose signature is ``f(x: T, ...) -> S``,
    ``vectorize()`` returns a wrapper of ``f()`` whose signature is
    ``f(x: MatrixLike[T, M, N], ...) -> CheckedMap[S, M, N]``.

    Vectorization will only be applied to positional function arguments.
    Keyword arguments are stripped, and will no longer be passable.

    The returned object is a ``CheckedMap``, which can be fed directly to the
    built-in ``Matrix`` constructor (and any of its sub-classes, of course).
    """

    def vectorize_decorator(func, /):

        def vectorize_wrapper(matrix1, /, *matrices):
            return CheckedMap(func, matrix1, *matrices)

        return vectorize_wrapper

    return vectorize_decorator
