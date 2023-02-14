import functools
from collections.abc import Iterator
from typing import Generic, TypeVar

__all__ = ["checked_map", "vectorize"]

T = TypeVar("T")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class checked_map(Iterator[T], Generic[T, M, N]):
    """Return a shaped ``map``-like iterator over one or more matrices

    Raises ``ValueError`` if not all of the input matrices are of equal shape.
    """

    __slots__ = ("_items", "_shape")

    def __init__(self, func, matrix1, /, *matrices):
        u = matrix1.shape
        for matrix2 in matrices:
            v = matrix2.shape
            if u != v:
                raise ValueError(f"incompatible shapes {u}, {v}")
        self._shape = u
        self._items = map(func, matrix1, *matrices)

    def __next__(self):
        return next(self._items)

    def __iter__(self):
        return self

    @property
    def shape(self):
        """The intended shape"""
        return self._shape

    @property
    def nrows(self):
        """The intended number of rows"""
        return self.shape[0]

    @property
    def ncols(self):
        """The intended number of columns"""
        return self.shape[1]

    @property
    def size(self):
        """The intended size"""
        shape = self.shape
        return shape[0] * shape[1]


def vectorize():
    """Convert a scalar-based function into a matrix-based one

    For a function, ``f()``, whose signature is ``f(x: T, ...) -> S``,
    ``vectorize()`` returns a wrapper of ``f()`` whose signature is
    ``f(x: MatrixLike[T, M, N], ...) -> checked_map[S, M, N]``.

    Vectorization only applies to positional arguments. Keyword arguments are
    preserved as they appear in the decorated function, though this is not
    properly type-hinted due to some current limitations of the typing system.

    The returned object is a ``checked_map`` iterator, which can be directly
    fed into the built-in ``Matrix`` constructor (and all of its sub-classes,
    of course).
    """

    def vectorize_decorator(func, /):

        def vectorize_wrapper(matrix1, /, *matrices, **kwargs):
            return checked_map(
                functools.partial(func, **kwargs) if kwargs else func,
                matrix1,
                *matrices,
            )

        return vectorize_wrapper

    return vectorize_decorator
