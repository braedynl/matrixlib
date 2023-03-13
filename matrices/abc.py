from abc import abstractmethod
from collections.abc import Sized
from typing import Protocol, TypeVar

__all__ = ["Shaped"]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class Shaped(Sized, Protocol[M_co, N_co]):

    def __len__(self) -> int:
        return self.nrows * self.ncols

    @property
    @abstractmethod
    def shape(self) -> tuple[M_co, N_co]:
        raise NotImplementedError

    @property
    def nrows(self) -> M_co:
        return self.shape[0]

    @property
    def ncols(self) -> N_co:
        return self.shape[1]
