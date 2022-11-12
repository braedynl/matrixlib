import itertools

from .protocols import MatrixLike

__all__ = [
    "only",
    "likewise",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "conjugate",
]


def only(x):
    """Return the only contained object of a length 1 collection

    Raises `ValueError` if the collection does not have length 1.
    """
    if (n := len(x)) != 1:
        raise ValueError(f"cannot demote size {n} collection to singular object")
    return x[0]


def likewise(x, shape):
    """Return an iterator of `x` as if it were a matrix of `shape`

    Raises `ValueError` under the following conditions:
    - The shape of `x` does not equal the input `shape` when `x` is a
      `MatrixLike`.
    - The input `shape` has a product of 0 when `x` is not a `MatrixLike`.

    Non-`MatrixLike` objects conflict with product 0 shapes, since they are
    treated as being equivalent to a matrix filled solely by the object (which
    cannot have a size of 0).
    """
    u = shape
    if isinstance(x, MatrixLike):
        if u != (v := x.shape):
            raise ValueError(f"shape {u} is incompatible with operand shape {v}")
        it = iter(x)
    else:
        if 0 in u:
            raise ValueError(f"shape {u} is incompatible with operand of non-zero size")
        it = itertools.repeat(x, times=u.nrows * u.ncols)
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
