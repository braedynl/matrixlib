from collections.abc import Iterator
from typing import Any, Protocol, TypeVar

from .protocols import MatrixLike, ShapeLike

__all__ = [
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


def likewise(object: MatrixLike[T] | T, shape: ShapeLike) -> Iterator[T]: ...

def logical_and(a: Any, b: Any, /) -> bool: ...
def logical_or(a: Any, b: Any, /) -> bool: ...
def logical_xor(a: Any, b: Any, /) -> bool: ...
def logical_not(a: Any, /) -> bool: ...

def conjugate(x: SupportsConjugate[T], /) -> T: ...
