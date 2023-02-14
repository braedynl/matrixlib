from collections.abc import Iterator
from typing import Generic, TypeVar

__all__ = ["checked_map", "vectorize"]

T = TypeVar("T")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


class checked_map(Iterator[T], Generic[T, M, N]):

    __slots__ = ("_items", "_shape")

    def __init__(self, func, matrix1, /, *matrices):
        u = matrix1.shape
        for matrix2 in matrices:
            v = matrix2.shape
            if u != v:
                raise ValueError(f"incompatible shapes {u}, {v}")
        self._shape = u
        self._items = iter(map(func, matrix1, *matrices))

    def __next__(self):
        return next(self._items)

    def __iter__(self):
        return self

    @property
    def shape(self):
        return self._shape


def vectorize(func, /):

    def vectorize_wrapper(matrix1, /, *matrices):
        return checked_map(func, matrix1, *matrices)

    return vectorize_wrapper
