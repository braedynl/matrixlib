from __future__ import annotations

__all__ = [
    "logical_and",
    "logical_xor",
    "logical_or",
    "logical_not",
]


def logical_and(a: object, b: object, /) -> bool:
    """Return the logical ``and`` of two objects."""
    return not not (a and b)


def logical_xor(a: object, b: object, /) -> bool:
    """Return the logical exclusive-or of two objects."""
    return (not not a) != (not not b)


def logical_or(a: object, b: object, /) -> bool:
    """Return the logical ``or`` of two objects."""
    return not not (a or b)


def logical_not(a: object, /) -> bool:
    """Return the logical ``not`` of an object."""
    return not a
