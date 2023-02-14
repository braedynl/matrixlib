from collections.abc import Callable, Collection, Iterator
from typing import Any, Literal, TypeVar, overload

from ..abc import MatrixLike, ShapedIterable

__all__ = ["MatrixMap"]

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class MatrixMap(Collection[T_co], ShapedIterable[T_co, M_co, N_co]):

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
