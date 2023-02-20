from collections.abc import Callable, Iterator
from typing import Any, TypeVar, overload

from ..abc import MatrixLike, ShapedCollection

__all__ = ["MatrixMap"]

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class MatrixMap(ShapedCollection[T_co, M_co, N_co]):

    @overload
    def __init__(
        self,
        f: Callable[[T1], T_co],
        a: MatrixLike[T1, M_co, N_co],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        f: Callable[[T1, T2], T_co],
        a: MatrixLike[T1, M_co, N_co],
        b: MatrixLike[T2, M_co, N_co],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        f: Callable[[T1, T2, T3], T_co],
        a: MatrixLike[T1, M_co, N_co],
        b: MatrixLike[T2, M_co, N_co],
        c: MatrixLike[T3, M_co, N_co],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        f: Callable[[T1, T2, T3, T4], T_co],
        a: MatrixLike[T1, M_co, N_co],
        b: MatrixLike[T2, M_co, N_co],
        c: MatrixLike[T3, M_co, N_co],
        d: MatrixLike[T4, M_co, N_co],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        f: Callable[[T1, T2, T3, T4, T5], T_co],
        a: MatrixLike[T1, M_co, N_co],
        b: MatrixLike[T2, M_co, N_co],
        c: MatrixLike[T3, M_co, N_co],
        d: MatrixLike[T4, M_co, N_co],
        e: MatrixLike[T5, M_co, N_co],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        f: Callable[..., T_co],
        a: MatrixLike[Any, M_co, N_co],
        /,
        *bx: MatrixLike[Any, M_co, N_co],
    ) -> None: ...
    def __iter__(self) -> Iterator[T_co]: ...

    @property
    def shape(self) -> tuple[M_co, N_co]: ...
