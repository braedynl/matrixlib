from collections.abc import Callable, Iterator
from typing import Any, TypeVar, overload

from ..abc import MatrixLike
from .rule import *

__all__ = [
    "Rule",
    "ROW",
    "COL",
    "checked_map",
    "logical_and",
    "logical_or",
    "logical_not",
]

T = TypeVar("T")

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


@overload
def checked_map(func: Callable[[T1], T], a: MatrixLike[T1, M, N]) -> Iterator[T]: ...
@overload
def checked_map(func: Callable[[T1, T2], T], a: MatrixLike[T1, M, N], b: MatrixLike[T2, M, N]) -> Iterator[T]: ...
@overload
def checked_map(func: Callable[[T1, T2, T3], T], a: MatrixLike[T1, M, N], b: MatrixLike[T2, M, N], c: MatrixLike[T3, M, N]) -> Iterator[T]: ...
@overload
def checked_map(func: Callable[[T1, T2, T3, T4], T], a: MatrixLike[T1, M, N], b: MatrixLike[T2, M, N], c: MatrixLike[T3, M, N], d: MatrixLike[T4, M, N]) -> Iterator[T]: ...
@overload
def checked_map(func: Callable[[T1, T2, T3, T4, T5], T], a: MatrixLike[T1, M, N], b: MatrixLike[T2, M, N], c: MatrixLike[T3, M, N], d: MatrixLike[T4, M, N], e: MatrixLike[T5, M, N]) -> Iterator[T]: ...
@overload
def checked_map(func: Callable[..., T], a: MatrixLike[Any, M, N], *bx: MatrixLike[Any, M, N]) -> Iterator[T]: ...
def logical_and(a: Any, b: Any, /) -> bool: ...
def logical_or(a: Any, b: Any, /) -> bool: ...
def logical_not(a: Any, /) -> bool: ...
