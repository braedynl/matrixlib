from __future__ import annotations

import operator
from abc import abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class Mesh(Sequence[T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, Mesh):
            return (
                self.shape == other.shape
                and
                all(map(operator.eq, self, other))
            )
        return NotImplemented

    def __len__(self) -> int:
        return self.nrows * self.ncols

    @property
    @abstractmethod
    def array(self) -> Sequence[T_co]:
        raise NotImplementedError

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
