from abc import abstractmethod
from collections.abc import Iterable, Sized
from typing import Protocol, TypeVar, runtime_checkable

__all__ = ["Shaped", "ShapedIterable"]

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


@runtime_checkable
class Shaped(Sized, Protocol[M_co, N_co]):

    def __len__(self) -> int:
        shape = self.shape
        return shape[0] * shape[1]

    @property
    @abstractmethod
    def shape(self) -> tuple[M_co, N_co]:
        pass


@runtime_checkable
class ShapedIterable(Shaped[M_co, N_co], Iterable[T_co], Protocol[T_co, M_co, N_co]):
    pass
