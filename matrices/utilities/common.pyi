from collections.abc import Callable, Iterator
from typing import Literal, TypeVar, overload

from ..abstract import MatrixLike
from ..typeshed import (SupportsClosedAdd, SupportsDotProduct,
                        SupportsRDotProduct)

__all__ = ["checked_map", "compare", "multiply"]

T = TypeVar("T")

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
P = TypeVar("P", bound=int)

SupportsClosedAddT = TypeVar("SupportsClosedAddT", bound=SupportsClosedAdd)

@overload
def checked_map(func: Callable[[T1], T], a: MatrixLike[T1, M, N]) -> Iterator[T]: ...
@overload
def checked_map(func: Callable[[T1, T2], T], a: MatrixLike[T1, M, N], b: MatrixLike[T2, M, N]) -> Iterator[T]: ...
@overload
def checked_map(func: Callable[[T1, T2, T3], T], a: MatrixLike[T1, M, N], b: MatrixLike[T2, M, N], c: MatrixLike[T3, M, N]) -> Iterator[T]: ...
@overload
def checked_map(func: Callable[[T1, T2, T3, T4], T], a: MatrixLike[T1, M, N], b: MatrixLike[T2, M, N], c: MatrixLike[T3, M, N], d: MatrixLike[T4, M, N]) -> Iterator[T]: ...
def compare(a: MatrixLike[T, M, N], b: MatrixLike[T, M, N]) -> Literal[-1, 0, 1]: ...
@overload
def multiply(a: MatrixLike[SupportsDotProduct[T, SupportsClosedAddT], M, N], b: MatrixLike[T, N, P]) -> Iterator[SupportsClosedAddT]: ...
@overload
def multiply(a: MatrixLike[T, M, N], b: MatrixLike[SupportsRDotProduct[T, SupportsClosedAddT], N, P]) -> Iterator[SupportsClosedAddT]: ...
