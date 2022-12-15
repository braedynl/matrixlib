import operator
from typing import TypeVar

from .abstract import ShapeLike

__all__ = ["ShapeView"]

NRowsT_co = TypeVar("NRowsT_co", covariant=True, bound=int)
NColsT_co = TypeVar("NColsT_co", covariant=True, bound=int)


class ShapeView(ShapeLike[NRowsT_co, NColsT_co]):

    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    def __repr__(self):
        """Return a canonical representation of the view"""
        return f"{self.__class__.__name__}(target={self._target!r})"

    __str__ = __repr__

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
