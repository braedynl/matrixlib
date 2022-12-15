import enum
from collections.abc import Callable
from enum import Flag
from typing import Any, Literal, TypeVar, overload

from .matrices import MatrixLike
from .typeshed import SupportsConjugate

__all__ = [
    "Ordering",
    "matrix_order",
    "matrix_map",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "conjugate",
]

T = TypeVar("T")
S = TypeVar("S")

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

NRowsT = TypeVar("NRowsT", bound=int)
NColsT = TypeVar("NColsT", bound=int)


class Ordering(Flag):
    LESSER  = enum.auto()
    EQUAL   = enum.auto()
    GREATER = enum.auto()


def matrix_order(a: MatrixLike[T, NRowsT, NColsT], b: MatrixLike[T, NRowsT, NColsT]) -> Literal[Ordering.LESSER, Ordering.EQUAL, Ordering.GREATER]: ...
@overload
def matrix_map(func: Callable[[T1], S], a: MatrixLike[T1, NRowsT, NColsT]) -> map[S]: ...
@overload
def matrix_map(func: Callable[[T1, T2], S], a: MatrixLike[T1, NRowsT, NColsT], b: MatrixLike[T2, NRowsT, NColsT]) -> map[S]: ...
@overload
def matrix_map(func: Callable[[T1, T2, T3], S], a: MatrixLike[T1, NRowsT, NColsT], b: MatrixLike[T2, NRowsT, NColsT], c: MatrixLike[T3, NRowsT, NColsT]) -> map[S]: ...
@overload
def matrix_map(func: Callable[[T1, T2, T3, T4], S], a: MatrixLike[T1, NRowsT, NColsT], b: MatrixLike[T2, NRowsT, NColsT], c: MatrixLike[T3, NRowsT, NColsT], d: MatrixLike[T4, NRowsT, NColsT]) -> map[S]: ...
@overload
def matrix_map(func: Callable[[T1, T2, T3, T4, T5], S], a: MatrixLike[T1, NRowsT, NColsT], b: MatrixLike[T2, NRowsT, NColsT], c: MatrixLike[T3, NRowsT, NColsT], d: MatrixLike[T4, NRowsT, NColsT], e: MatrixLike[T5, NRowsT, NColsT]) -> map[S]: ...
@overload
def matrix_map(func: Callable[..., S], a: MatrixLike[Any, NRowsT, NColsT], b: MatrixLike[Any, NRowsT, NColsT], c: MatrixLike[Any, NRowsT, NColsT], d: MatrixLike[Any, NRowsT, NColsT], e: MatrixLike[Any, NRowsT, NColsT], *fx: MatrixLike[Any, NRowsT, NColsT]) -> map[S]: ...

def logical_and(a: Any, b: Any, /) -> bool: ...
def logical_or(a: Any, b: Any, /) -> bool: ...
def logical_xor(a: Any, b: Any, /) -> bool: ...
def logical_not(a: Any, /) -> bool: ...
def conjugate(x: SupportsConjugate[T], /) -> T: ...
