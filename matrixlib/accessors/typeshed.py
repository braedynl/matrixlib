from __future__ import annotations

__all__ = [
    "Shaped",
    "SizedIterable",
    "ShapedIterable",
]

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Sized
from typing import Protocol, override, runtime_checkable


@runtime_checkable
class Shaped[M: int, N: int](Protocol, metaclass=ABCMeta):

    @property
    @abstractmethod
    def shape(self) -> tuple[M, N]:
        raise NotImplementedError


@runtime_checkable
class SizedIterable[T](Iterable[T], Sized, Protocol, metaclass=ABCMeta):
    ...


@runtime_checkable
class ShapedIterable[M: int, N: int, T](SizedIterable[T], Shaped[M, N], Protocol, metaclass=ABCMeta):

    @override
    def __len__(self) -> int:
        shape = self.shape
        return shape[0] * shape[1]
