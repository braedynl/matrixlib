import itertools

from .protocols import MatrixLike

__all__ = [
    "likewise",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "conjugate",
]


def likewise(object, shape):
    """Return an iterator of `object` as if it were a matrix of `shape`

    Raises `ValueError` under the following conditions:
    - The shape of `object` does not equal the input `shape` when `object` is a
      `MatrixLike`.
    - The input `shape` has a product of 0 when `object` is not a `MatrixLike`.

    Non-`MatrixLike` objects conflict with product 0 shapes, since they are
    treated as being equivalent to a matrix filled solely by the object (which
    cannot have a size of 0).
    """
    u = shape
    if isinstance(object, MatrixLike):
        if u != (v := object.shape):
            raise ValueError(f"shape {u} is incompatible with operand shape {v}")
        it = iter(object)
    else:
        if 0 in u:
            raise ValueError(f"shape {u} is incompatible with operand of non-zero size")
        it = itertools.repeat(object, times=u.nrows * u.ncols)
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
