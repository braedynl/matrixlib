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


def checked_map(func, a, *bx):
    """Return an iterator that computes a function from the arguments of
    equally-shaped matrices

    Raises `ValueError` if there exists a matrix whose shape differs from the
    others.
    """
    if not bx:
        return map(func, a)
    u = a.shape
    for b in bx:
        if u != (v := b.shape):
            raise ValueError(f"incompatible shapes {u}, {v}")
    return map(func, a, *bx)


def logical_and(a, b, /):
    """Return the logical AND of `a` and `b`"""
    return not not (a and b)


def logical_or(a, b, /):
    """Return the logical OR of `a` and `b`"""
    return not not (a or b)


def logical_not(a, /):
    """Return the logical NOT of `a`"""
    return not a
