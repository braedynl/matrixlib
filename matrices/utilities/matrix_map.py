from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, TypeVar, overload

from ..abc import ShapedCollection, ShapedIterable

__all__ = ["MatrixMap"]

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

S_co = TypeVar("S_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class MatrixMap(ShapedCollection[S_co, M_co, N_co]):
    """A ``ShapedCollection`` wrapper for mapping processes on ``MatrixLike``
    objects

    Note that elements of the collection are computed lazily, and without
    caching. ``__iter__()`` (or any of its dependents) will compute/re-compute
    the mapping for each call.
    """

    __slots__ = ("_func", "_args", "_shape")

    @overload
    def __init__(
        self,
        f: Callable[[T1], S_co],
        a: ShapedIterable[T1, M_co, N_co],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        f: Callable[[T1, T2], S_co],
        a: ShapedIterable[T1, M_co, N_co],
        b: ShapedIterable[T2, M_co, N_co],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        f: Callable[[T1, T2, T3], S_co],
        a: ShapedIterable[T1, M_co, N_co],
        b: ShapedIterable[T2, M_co, N_co],
        c: ShapedIterable[T3, M_co, N_co],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        f: Callable[[T1, T2, T3, T4], S_co],
        a: ShapedIterable[T1, M_co, N_co],
        b: ShapedIterable[T2, M_co, N_co],
        c: ShapedIterable[T3, M_co, N_co],
        d: ShapedIterable[T4, M_co, N_co],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        f: Callable[[T1, T2, T3, T4, T5], S_co],
        a: ShapedIterable[T1, M_co, N_co],
        b: ShapedIterable[T2, M_co, N_co],
        c: ShapedIterable[T3, M_co, N_co],
        d: ShapedIterable[T4, M_co, N_co],
        e: ShapedIterable[T5, M_co, N_co],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        f: Callable[..., S_co],
        a: ShapedIterable[Any, M_co, N_co],
        /,
        *bs: ShapedIterable[Any, M_co, N_co],
    ) -> None: ...

    def __init__(self, f, a, /, *bs):
        u = a.shape
        for b in bs:
            v = b.shape
            if u != v:
                raise ValueError(f"incompatible shapes {u}, {v}")
        self._func  = f
        self._args  = a, *bs
        self._shape = u

    def __iter__(self) -> Iterator[S_co]:
        yield from map(self._func, *self._args)

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self._shape
