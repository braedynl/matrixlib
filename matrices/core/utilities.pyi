import enum
from collections.abc import Callable
from enum import Flag
from typing import Any, Literal, TypeVar, overload

from .matrices import MatrixLike
from .typeshed import SupportsConjugate, SupportsGT, SupportsLT

__all__ = [
    "Ordering",
    "order",
    "apply",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "conjugate",
]

T  = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")

NRowsT = TypeVar("NRowsT", bound=int)
NColsT = TypeVar("NColsT", bound=int)


class Ordering(Flag):
    LESSER  = enum.auto()
    EQUAL   = enum.auto()
    GREATER = enum.auto()


@overload
def order(a: MatrixLike[SupportsLT[T], NRowsT, NColsT], b: MatrixLike[T, NRowsT, NColsT], /) -> Literal[Ordering.LESSER, Ordering.EQUAL, Ordering.GREATER]: ...
@overload
def order(a: MatrixLike[SupportsGT[T], NRowsT, NColsT], b: MatrixLike[T, NRowsT, NColsT], /) -> Literal[Ordering.LESSER, Ordering.EQUAL, Ordering.GREATER]: ...
@overload
def order(a: MatrixLike[T, NRowsT, NColsT], b: MatrixLike[SupportsLT[T], NRowsT, NColsT], /) -> Literal[Ordering.LESSER, Ordering.EQUAL, Ordering.GREATER]: ...
@overload
def order(a: MatrixLike[T, NRowsT, NColsT], b: MatrixLike[SupportsGT[T], NRowsT, NColsT], /) -> Literal[Ordering.LESSER, Ordering.EQUAL, Ordering.GREATER]: ...
@overload
def apply(func: Callable[[T1], T2], a: MatrixLike[T1, NRowsT, NColsT], /) -> map[T2]: ...
@overload
def apply(func: Callable[[T1, T2], T3], a: MatrixLike[T1, NRowsT, NColsT], b: MatrixLike[T2, NRowsT, NColsT], /) -> map[T3]: ...
def logical_and(a: Any, b: Any, /) -> bool: ...
def logical_or(a: Any, b: Any, /) -> bool: ...
def logical_xor(a: Any, b: Any, /) -> bool: ...
def logical_not(a: Any, /) -> bool: ...
def conjugate(x: SupportsConjugate[T], /) -> T: ...
