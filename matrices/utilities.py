from typing import Any, TypeVar

from .types import SupportsConjugate

__all__ = [
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "conjugate",
]

T = TypeVar("T")


def logical_and(a: Any, b: Any, /) -> bool:
    """Return the logical AND of two objects"""
    return not not (a and b)


def logical_or(a: Any, b: Any, /) -> bool:
    """Return the logical OR of two objects"""
    return not not (a or b)


def logical_xor(a: Any, b: Any, /) -> bool:
    """Return the logical XOR of two objects"""
    return (not not a) is not (not not b)


def logical_not(a: Any, /) -> bool:
    """Return the logical NOT of an object"""
    return not a


def conjugate(x: SupportsConjugate[T], /) -> T:
    """Return the conjugate of an object"""
    return x.conjugate()
