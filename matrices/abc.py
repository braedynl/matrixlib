from abc import ABCMeta, abstractmethod
from collections.abc import Sized
from typing import Generic, TypeVar

__all__ = ["Shaped"]

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class Shaped(Sized, Generic[M_co, N_co], metaclass=ABCMeta):
    """ABC for classes that provide the ``__len__()`` method, alongside the
    ``shape``, ``nrows``, and ``ncols`` properties.
    """

    __slots__ = ()

    def __len__(self) -> int:
        """Return the shape's product"""
        return self.nrows * self.ncols

    @property
    @abstractmethod
    def shape(self) -> tuple[M_co, N_co]:
        """The number of rows and columns as a ``tuple``"""
        raise NotImplementedError

    @property
    def nrows(self) -> M_co:
        """The number of rows"""
        return self.shape[0]

    @property
    def ncols(self) -> N_co:
        """The number of columns"""
        return self.shape[1]
