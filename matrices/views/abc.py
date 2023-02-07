from abc import ABCMeta
from reprlib import recursive_repr
from typing import TypeVar

from ..abc import MatrixLike

__all__ = ["MatrixLikeView"]

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", bound=int, covariant=True)
N_co = TypeVar("N_co", bound=int, covariant=True)


class MatrixLikeView(MatrixLike[T_co, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    @recursive_repr("...")
    def __repr__(self):
        """Return a canonical representation of the view"""
        return f"{self.__class__.__name__}(target={self._target!r})"

    def __deepcopy__(self, memo=None):
        """Return the view"""
        return self

    __copy__ = __deepcopy__
