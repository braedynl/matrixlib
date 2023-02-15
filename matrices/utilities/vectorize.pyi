from collections.abc import Callable
from typing import TypeVar, overload

from ..abc import MatrixLike
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


class _vectorize_dispatch:

    # NOTE: this type does *not* exist in the implementation file - this is
    # solely to help vectorize() give correct overloading information.

    @overload
    @staticmethod
    def __call__(
        func: Callable[[T1], S],
        /,
    ) -> Callable[[MatrixLike[T1, M, N]], MatrixMap[S, M, N]]: ...
    @overload
    @staticmethod
    def __call__(
        func: Callable[[T1, T2], S],
        /,
    ) -> Callable[[MatrixLike[T1, M, N], MatrixLike[T2, M, N]], MatrixMap[S, M, N]]: ...
    @overload
    @staticmethod
    def __call__(
        func: Callable[[T1, T2, T3], S],
        /,
    ) -> Callable[[MatrixLike[T1, M, N], MatrixLike[T2, M, N], MatrixLike[T3, M, N]], MatrixMap[S, M, N]]: ...
    @overload
    @staticmethod
    def __call__(
        func: Callable[[T1, T2, T3, T4], S],
        /,
    ) -> Callable[[MatrixLike[T1, M, N], MatrixLike[T2, M, N], MatrixLike[T3, M, N], MatrixLike[T4, M, N]], MatrixMap[S, M, N]]: ...
    @overload
    @staticmethod
    def __call__(
        func: Callable[[T1, T2, T3, T4, T5], S],
        /,
    ) -> Callable[[MatrixLike[T1, M, N], MatrixLike[T2, M, N], MatrixLike[T3, M, N], MatrixLike[T4, M, N], MatrixLike[T5, M, N]], MatrixMap[S, M, N]]: ...
    @overload
    @staticmethod
    def __call__(
        func: Callable[..., S],
        /,
    ) -> Callable[..., MatrixMap[S, M, N]]: ...


def vectorize() -> _vectorize_dispatch: ...
