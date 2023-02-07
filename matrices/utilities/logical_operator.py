
__all__ = [
    "logical_and",
    "logical_or",
    "logical_not",
]


def logical_and(a, b, /):
    """Return the logical AND of ``a`` and ``b``"""
    return not not (a and b)


def logical_or(a, b, /):
    """Return the logical OR of ``a`` and ``b``"""
    return not not (a or b)


def logical_not(a, /):
    """Return the logical NOT of ``a``"""
    return not a
