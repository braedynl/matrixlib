from typing import Any

__all__ = [
    "logical_and",
    "logical_or",
    "logical_not",
]


def logical_and(a: Any, b: Any, /) -> bool: ...
def logical_or(a: Any, b: Any, /) -> bool: ...
def logical_not(a: Any, /) -> bool: ...
