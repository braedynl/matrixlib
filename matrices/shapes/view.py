import operator
from typing import TypeVar

from .abstract import ShapeLike

__all__ = ["ShapeView"]

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class ShapeView(ShapeLike[M_co, N_co]):

    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    def __repr__(self):
        """Return a canonical representation of the view"""
        return f"{self.__class__.__name__}(target={self._target!r})"

    def __getitem__(self, key):
        return self._target[operator.index(key)]

    def __iter__(self):
        yield from self._target

    def __reversed__(self):
        yield from reversed(self._target)

    def __contains__(self, value):
        return value in self._target

    def __deepcopy__(self, memo=None):
        """Return the view"""
        return self

    __copy__ = __deepcopy__
