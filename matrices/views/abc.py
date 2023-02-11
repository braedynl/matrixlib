from abc import ABCMeta
from typing import TypeVar

from views import SequenceViewLike

from ..abc import MatrixLike

__all__ = ["MatrixViewLike"]

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class MatrixViewLike(SequenceViewLike[T_co], MatrixLike[T_co, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()
