import enum
from enum import Flag

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


class Ordering(Flag):
    LESSER  = enum.auto()
    EQUAL   = enum.auto()
    GREATER = enum.auto()


def order(a, b, /):
    if (u := a.shape) != (v := b.shape):
        raise ValueError(f"matrix of shape {u} is incompatible with operand shape {v}")
    for x, y in zip(a, b):
        if x < y:
            return Ordering.LESSER
        if x > y:
            return Ordering.GREATER
    return Ordering.EQUAL


def apply(func, a, b, /):
    if (u := a.shape) != (v := b.shape):
        raise ValueError(f"matrix of shape {u} is incompatible with operand shape {v}")
    return map(func, a, b)


def logical_and(a, b, /):
    """Return the logical AND of `a` and `b`"""
    return not not (a and b)


def logical_or(a, b, /):
    """Return the logical OR of `a` and `b`"""
    return not not (a or b)


def logical_xor(a, b, /):
    """Return the logical XOR of `a` and `b`"""
    return (not not a) is not (not not b)


def logical_not(a, /):
    """Return the logical NOT of `a`"""
    return not a


def conjugate(x, /):
    """Return the object's conjugate"""
    return x.conjugate()
