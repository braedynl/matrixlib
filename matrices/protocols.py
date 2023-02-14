from abc import abstractmethod
from collections.abc import Iterable
from typing import Protocol, TypeVar, runtime_checkable

__all__ = ["Shaped", "ShapedIterable"]

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


@runtime_checkable
class Shaped(Protocol[M_co, N_co]):
    """Protocol class indicating the presence of a ``shape`` property"""

    @property
    @abstractmethod
    def shape(self) -> tuple[M_co, N_co]:
        """A ``tuple`` containing the number of rows and columns, respectively"""
        pass


@runtime_checkable
class ShapedIterable(Shaped[M_co, N_co], Iterable[T_co], Protocol[T_co, M_co, N_co]):
    """Protocol class indicating the presence of a ``shape`` property and
    an ``__iter__()`` method

    It is expected that the number of items emitted by ``__iter__()`` is equal
    to the product of ``shape``.
    """
    pass
