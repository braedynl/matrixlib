from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, final, overload

from ..abc import ShapedIterable
from .matrix_map import MatrixMap

__all__ = ["vectorize"]

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

S = TypeVar("S")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


@final
class vectorize:
    """Convert a scalar-based function into a matrix-based one

    For a function, ``f()``, whose signature is ``f(x: T, ...) -> S``,
    ``vectorize()`` returns a wrapper of ``f()`` whose signature is
    ``f(x: ShapedIterable[T, M, N], ...) -> MatrixMap[S, M, N]``. Equivalent to
    constructing ``MatrixMap(f, x, ...)``.

    Vectorization will only be applied to positional function arguments.
    Keyword arguments are stripped, and will no longer be passable.
    """

    __slots__ = ()

    @overload
    def __call__(
        self,
        f: Callable[[T1], S],
        /,
    ) -> Callable[[ShapedIterable[T1, M, N]], MatrixMap[S, M, N]]: ...
    @overload
    def __call__(
        self,
        f: Callable[[T1, T2], S],
        /,
    ) -> Callable[[ShapedIterable[T1, M, N], ShapedIterable[T2, M, N]], MatrixMap[S, M, N]]: ...
    @overload
    def __call__(
        self,
        f: Callable[[T1, T2, T3], S],
        /,
    ) -> Callable[[ShapedIterable[T1, M, N], ShapedIterable[T2, M, N], ShapedIterable[T3, M, N]], MatrixMap[S, M, N]]: ...
    @overload
    def __call__(
        self,
        f: Callable[[T1, T2, T3, T4], S],
        /,
    ) -> Callable[[ShapedIterable[T1, M, N], ShapedIterable[T2, M, N], ShapedIterable[T3, M, N], ShapedIterable[T4, M, N]], MatrixMap[S, M, N]]: ...
    @overload
    def __call__(
        self,
        f: Callable[[T1, T2, T3, T4, T5], S],
        /,
    ) -> Callable[[ShapedIterable[T1, M, N], ShapedIterable[T2, M, N], ShapedIterable[T3, M, N], ShapedIterable[T4, M, N], ShapedIterable[T5, M, N]], MatrixMap[S, M, N]]: ...
    @overload
    def __call__(
        self,
        f: Callable[..., S],
        /,
    ) -> Callable[..., MatrixMap[S, M, N]]: ...

    def __call__(self, f: Callable[..., S], /) -> Callable[..., MatrixMap[S, M, N]]:

        def vectorize_wrapper(a: ShapedIterable[Any, M, N], /, *bx: ShapedIterable[Any, M, N]) -> MatrixMap[S, M, N]:
            return MatrixMap(f, a, *bx)

        vectorize_wrapper.__module__   = f.__module__  # We don't use functools.wraps() since it
        vectorize_wrapper.__name__     = f.__name__    # copies annotations
        vectorize_wrapper.__qualname__ = f.__qualname__
        vectorize_wrapper.__doc__      = f.__doc__

        return vectorize_wrapper
