from collections.abc import Callable, Iterator
from typing import Any, TypeVar, overload

from ..abc import MatrixLike

__all__ = ["checked_map", "checked_rmap"]

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

T = TypeVar("T")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


@overload
def checked_map(
    func: Callable[[T1], T],
    matrix1: MatrixLike[T1, M, N],
    /,
) -> Iterator[T]: ...
@overload
def checked_map(
    func: Callable[[T1, T2], T],
    matrix1: MatrixLike[T1, M, N],
    matrix2: MatrixLike[T2, M, N],
    /,
) -> Iterator[T]: ...
@overload
def checked_map(
    func: Callable[[T1, T2, T3], T],
    matrix1: MatrixLike[T1, M, N],
    matrix2: MatrixLike[T2, M, N],
    matrix3: MatrixLike[T3, M, N],
    /,
) -> Iterator[T]: ...
@overload
def checked_map(
    func: Callable[[T1, T2, T3, T4], T],
    matrix1: MatrixLike[T1, M, N],
    matrix2: MatrixLike[T2, M, N],
    matrix3: MatrixLike[T3, M, N],
    matrix4: MatrixLike[T4, M, N],
    /,
) -> Iterator[T]: ...
@overload
def checked_map(
    func: Callable[[T1, T2, T3, T4, T5], T],
    matrix1: MatrixLike[T1, M, N],
    matrix2: MatrixLike[T2, M, N],
    matrix3: MatrixLike[T3, M, N],
    matrix4: MatrixLike[T4, M, N],
    matrix5: MatrixLike[T5, M, N],
    /,
) -> Iterator[T]: ...
@overload
def checked_map(
    func: Callable[..., T],
    matrix: MatrixLike[Any, M, N],
    /,
    *matrices: MatrixLike[Any, M, N],
) -> Iterator[T]: ...

@overload
def checked_rmap(
    func: Callable[[T1], T],
    matrix1: MatrixLike[T1, M, N],
    /,
) -> Iterator[T]: ...
@overload
def checked_rmap(
    func: Callable[[T1, T2], T],
    matrix1: MatrixLike[T2, M, N],
    matrix2: MatrixLike[T1, M, N],
    /,
) -> Iterator[T]: ...
@overload
def checked_rmap(
    func: Callable[[T1, T2, T3], T],
    matrix1: MatrixLike[T3, M, N],
    matrix2: MatrixLike[T2, M, N],
    matrix3: MatrixLike[T1, M, N],
    /,
) -> Iterator[T]: ...
@overload
def checked_rmap(
    func: Callable[[T1, T2, T3, T4], T],
    matrix1: MatrixLike[T4, M, N],
    matrix2: MatrixLike[T3, M, N],
    matrix3: MatrixLike[T2, M, N],
    matrix4: MatrixLike[T1, M, N],
    /,
) -> Iterator[T]: ...
@overload
def checked_rmap(
    func: Callable[[T1, T2, T3, T4, T5], T],
    matrix1: MatrixLike[T5, M, N],
    matrix2: MatrixLike[T4, M, N],
    matrix3: MatrixLike[T3, M, N],
    matrix4: MatrixLike[T2, M, N],
    matrix5: MatrixLike[T1, M, N],
    /,
) -> Iterator[T]: ...
@overload
def checked_rmap(
    func: Callable[..., T],
    matrix: MatrixLike[Any, M, N],
    /,
    *matrices: MatrixLike[Any, M, N],
) -> Iterator[T]: ...
