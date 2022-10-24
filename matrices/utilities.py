import itertools

from .protocols import MatrixLike

__all__ = [
    "shaped",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "conjugate",
]


def shaped(obj, shape):
    """Return an iterator of `obj` as if it were a matrix of `shape`

    Raises `ValueError` under the following conditions:
    - The shape of `obj` does not equal the input `shape` when `obj` is a
      `MatrixLike`.
    - The input `shape` has size 0 when `obj` is not a `MatrixLike`.

    Reason for the latter condition being that an object who fills a matrix
    cannot be an empty matrix.
    """
    h = shape
    if isinstance(obj, MatrixLike):
        if h != (k := obj.shape):
            raise ValueError(f"shape {h} is incompatible with operand of shape {k}")
        it = iter(obj)
    else:
        if 0 in h:
            raise ValueError(f"shape {h} is incompatible with operand of non-zero size")
        it = itertools.repeat(obj, times=shape.size)
    return it


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
