import itertools
from collections.abc import Iterator
from typing import Any, Protocol, TypeVar

from .protocols import MatrixLike, ShapeLike

__all__ = [
    "only",
    "likewise",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "conjugate",
]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class SupportsConjugate(Protocol[T_co]):
    def conjugate(self) -> T_co: ...

class SupportsLenAndGetItem(Protocol[T_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, key: int) -> T_co: ...


def only(x: SupportsLenAndGetItem[T]) -> T:
    """Return the only contained object of a length 1 collection

    Raises `ValueError` if the collection does not have length 1.
    """
    if (n := len(x)) != 1:
        raise ValueError(f"cannot demote size {n} collection to singular object")
    return x[0]


def likewise(x: MatrixLike[T] | T, shape: ShapeLike) -> Iterator[T]:
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


def logical_and(a: Any, b: Any, /) -> bool:
    """Return the logical AND of `a` and `b`"""
    return not not (a and b)


def logical_or(a: Any, b: Any, /) -> bool:
    """Return the logical OR of `a` and `b`"""
    return not not (a or b)


def logical_xor(a: Any, b: Any, /) -> bool:
    """Return the logical XOR of `a` and `b`"""
    return (not not a) is not (not not b)


def logical_not(a: Any, /) -> bool:
    """Return the logical NOT of `a`"""
    return not a


def conjugate(x: SupportsConjugate[T], /) -> T:
    """Return the object's conjugate"""
    return x.conjugate()
