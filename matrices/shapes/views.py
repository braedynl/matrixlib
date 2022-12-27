import operator
from typing import TypeVar

from .abc import ShapeLike

__all__ = ["ShapeView"]

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


class ShapeView(ShapeLike[M, N]):

    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    def __repr__(self):
        """Return a canonical representation of the view"""
        return f"{self.__class__.__name__}(target={self._target!r})"

    def __getitem__(self, key):
        key = operator.index(key)
        return self._target[key]

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
