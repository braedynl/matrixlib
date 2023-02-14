from collections.abc import Callable, Iterator
from typing import Any, Generic, Literal, TypeVar, overload

from ..abc import MatrixLike

__all__ = ["checked_map", "vectorize"]

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

T = TypeVar("T")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)

Self = TypeVar("Self", bound=checked_map)


class checked_map(Iterator[T_co], Generic[T_co, M_co, N_co]):

    __slots__: tuple[Literal["_items"], Literal["_shape"]]

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

    def __next__(self) -> T_co: ...
    def __iter__(self: Self) -> Self: ...

    @property
    def shape(self) -> tuple[M_co, N_co]: ...


@overload
def vectorize(
    func: Callable[[T1], T],
    /,
) -> Callable[[MatrixLike[T1, M, N]], checked_map[T, M, N]]: ...
@overload
def vectorize(
    func: Callable[[T1, T2], T],
    /,
) -> Callable[[MatrixLike[T1, M, N], MatrixLike[T2, M, N]], checked_map[T, M, N]]: ...
@overload
def vectorize(
    func: Callable[[T1, T2, T3], T],
    /,
) -> Callable[[MatrixLike[T1, M, N], MatrixLike[T2, M, N], MatrixLike[T3, M, N]], checked_map[T, M, N]]: ...
@overload
def vectorize(
    func: Callable[[T1, T2, T3, T4], T],
    /,
) -> Callable[[MatrixLike[T1, M, N], MatrixLike[T2, M, N], MatrixLike[T3, M, N], MatrixLike[T4, M, N]], checked_map[T, M, N]]: ...
@overload
def vectorize(
    func: Callable[[T1, T2, T3, T4, T5], T],
    /,
) -> Callable[[MatrixLike[T1, M, N], MatrixLike[T2, M, N], MatrixLike[T3, M, N], MatrixLike[T4, M, N], MatrixLike[T5, M, N]], checked_map[T, M, N]]: ...
@overload
def vectorize(
    func: Callable[..., T],
    /,
) -> Callable[..., checked_map[T, M, N]]: ...
