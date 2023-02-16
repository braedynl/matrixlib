import functools
import itertools
import operator
from typing import TypeVar

from ..abc import ShapedCollection

__all__ = ["MatrixProduct"]

ComplexT_co = TypeVar("ComplexT_co", covariant=True, bound=complex)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)
P_co = TypeVar("P_co", covariant=True, bound=int)


class MatrixProduct(ShapedCollection[ComplexT_co, M_co, P_co]):
    """A ``ShapedCollection`` wrapper for matrix products on complex-valued
    ``MatrixLike`` objects

    Note that elements of the collection are computed lazily, and without
    caching. ``__iter__()`` (or any of its dependents) will compute/re-compute
    the mapping for each call.
    """

    __slots__ = ("_shape", "_matrix1", "_matrix2")

    def __init__(self, matrix1, matrix2):
        (m, n), (p, q) = (u, v) = (matrix1.shape, matrix2.shape)
        if n != p:
            raise ValueError(f"incompatible shapes {u}, {v} by inner dimensions")
        self._shape = (m, q)
        self._matrix1 = matrix1
        self._matrix2 = matrix2

    def __iter__(self):
        a = self._matrix1
        b = self._matrix2

        m = a.nrows
        n = a.ncols
        q = b.ncols

        if not n:
            yield from itertools.repeat(0, times=m * q)
            return

        ix = range(m)
        jx = range(q)
        kx = range(n)

        for i in ix:
            for j in jx:
                yield functools.reduce(
                    operator.add,
                    map(lambda k: a[i, k] * b[k, j], kx),
                )

    @property
    def shape(self):
        return self._shape