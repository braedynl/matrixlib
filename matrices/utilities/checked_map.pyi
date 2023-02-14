from collections.abc import Callable, Collection, Iterator
from typing import Any, Literal, TypeVar, overload

from ..abc import MatrixLike, ShapedIterable

__all__ = ["CheckedMap", "vectorize"]

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

T = TypeVar("T")
S = TypeVar("S")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class CheckedMap(Collection[T_co], ShapedIterable[T_co, M_co, N_co]):

    __slots__: tuple[Literal["_func"], Literal["_matrices"]]

    @overload
    def __init__(
        self,
        func: Callable[[T1], T_co],
        matrix1: MatrixLike[T1, M_co, N_co],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        func: Callable[[T1, T2], T_co],
        matrix1: MatrixLike[T1, M_co, N_co],
        matrix2: MatrixLike[T2, M_co, N_co],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        func: Callable[[T1, T2, T3], T_co],
        matrix1: MatrixLike[T1, M_co, N_co],
        matrix2: MatrixLike[T2, M_co, N_co],
        matrix3: MatrixLike[T3, M_co, N_co],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        func: Callable[[T1, T2, T3, T4], T_co],
        matrix1: MatrixLike[T1, M_co, N_co],
        matrix2: MatrixLike[T2, M_co, N_co],
        matrix3: MatrixLike[T3, M_co, N_co],
        matrix4: MatrixLike[T4, M_co, N_co],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        func: Callable[[T1, T2, T3, T4, T5], T_co],
        matrix1: MatrixLike[T1, M_co, N_co],
        matrix2: MatrixLike[T2, M_co, N_co],
        matrix3: MatrixLike[T3, M_co, N_co],
        matrix4: MatrixLike[T4, M_co, N_co],
        matrix5: MatrixLike[T5, M_co, N_co],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        func: Callable[..., T_co],
        matrix1: MatrixLike[Any, M_co, N_co],
        /,
        *matrices: MatrixLike[Any, M_co, N_co],
    ) -> None: ...
    def __iter__(self) -> Iterator[T_co]: ...
    def __contains__(self, value: Any) -> bool: ...

    @property
    def shape(self) -> tuple[M_co, N_co]: ...
    @property
    def func(self) -> Callable[..., T_co]: ...
    @property
    def matrices(self) -> tuple[MatrixLike[Any, M_co, N_co], ...]: ...


class vectorize_dispatch:

    # HACK: this type does *not* exist in the implementation file - this is
    # solely to help vectorize() give correct overloading information.

    @overload
    @staticmethod
    def __call__(
        func: Callable[[T1], S],
        /,
    ) -> Callable[[MatrixLike[T1, M, N]], CheckedMap[S, M, N]]: ...
    @overload
    @staticmethod
    def __call__(
        func: Callable[[T1, T2], S],
        /,
    ) -> Callable[[MatrixLike[T1, M, N], MatrixLike[T2, M, N]], CheckedMap[S, M, N]]: ...
    @overload
    @staticmethod
    def __call__(
        func: Callable[[T1, T2, T3], S],
        /,
    ) -> Callable[[MatrixLike[T1, M, N], MatrixLike[T2, M, N], MatrixLike[T3, M, N]], CheckedMap[S, M, N]]: ...
    @overload
    @staticmethod
    def __call__(
        func: Callable[[T1, T2, T3, T4], S],
        /,
    ) -> Callable[[MatrixLike[T1, M, N], MatrixLike[T2, M, N], MatrixLike[T3, M, N], MatrixLike[T4, M, N]], CheckedMap[S, M, N]]: ...
    @overload
    @staticmethod
    def __call__(
        func: Callable[[T1, T2, T3, T4, T5], S],
        /,
    ) -> Callable[[MatrixLike[T1, M, N], MatrixLike[T2, M, N], MatrixLike[T3, M, N], MatrixLike[T4, M, N], MatrixLike[T5, M, N]], CheckedMap[S, M, N]]: ...
    @overload
    @staticmethod
    def __call__(
        func: Callable[..., S],
        /,
    ) -> Callable[..., CheckedMap[S, M, N]]: ...


def vectorize() -> vectorize_dispatch: ...
