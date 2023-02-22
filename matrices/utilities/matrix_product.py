import itertools
import operator
from collections.abc import Iterable, Iterator
from typing import TypeVar

from ..abc import ShapedCollection, ShapedIndexable

__all__ = ["MatrixProduct"]

ComplexT = TypeVar("ComplexT", bound=complex)
ComplexT_co = TypeVar("ComplexT_co", covariant=True, bound=complex)
ComplexT_contra = TypeVar("ComplexT_contra", contravariant=True, bound=complex)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)
P_co = TypeVar("P_co", covariant=True, bound=int)


def sum_products(a: Iterable[ComplexT], b: Iterable[ComplexT], /) -> ComplexT:
    """Return the sum of products between two iterables, ``a`` and ``b``"""
    return sum(map(operator.mul, a, b))


class MatrixProduct(ShapedCollection[ComplexT_co, M_co, P_co]):
    """A ``ShapedCollection`` wrapper for matrix products on complex-valued
    ``MatrixLike`` objects

    Note that elements of the collection are computed lazily, and without
    caching. ``__iter__()`` (or any of its dependents) will compute/re-compute
    the mapping for each call.
    """

    __slots__ = ("_args", "_shape")

    def __init__(
        self,
        a: ShapedIndexable[ComplexT_co, M_co, N_co],
        b: ShapedIndexable[ComplexT_co, N_co, P_co],
        /,
    ) -> None:
        (m, n), (p, q) = (u, v) = (a.shape, b.shape)
        if n != p:
            raise ValueError(f"incompatible shapes {u}, {v} by inner dimensions")
        self._args  = (a, b)
        self._shape = (m, q)

    def __iter__(self) -> Iterator[ComplexT_co]:
        a, b = self._args

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
                yield sum_products(
                    (a[i * n + k] for k in kx),
                    (b[k * q + j] for k in kx),
                )

    @property
    def shape(self) -> tuple[M_co, P_co]:
        return self._shape
