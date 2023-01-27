from abc import ABCMeta, abstractmethod
from collections.abc import Collection
from typing import Generic, TypeVar

__all__ = ["ShapeLike"]

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class ShapeLike(Collection[M_co | N_co], Generic[M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()
    __match_args__ = ("nrows", "ncols")

    def __str__(self):
        """Return a string representation of the shape"""
        return f"{self[0]} × {self[1]}"

    def __eq__(self, other):
        """Return true if the two shapes are equal, otherwise false"""
        if self is other:
            return True
        if isinstance(other, ShapeLike):
            return (
                self[0] == other[0]
                and
                self[1] == other[1]
            )
        return NotImplemented

    def __ne__(self, other):
        """Return true if the two shapes are not equal, otherwise false"""
        if self is other:
            return False
        if isinstance(other, ShapeLike):
            return (
                self[0] != other[0]
                or
                self[1] != other[1]
            )
        return NotImplemented

    def __len__(self):
        """Return literal 2"""
        return 2

    @abstractmethod
    def __getitem__(self, key):
        """Return the dimension corresponding to `key`"""
        pass

    def __iter__(self):
        """Return an iterator over the dimensions of the shape"""
        yield self[0]
        yield self[1]

    def __reversed__(self):
        """Return a reversed iterator over the dimensions of the shape"""
        yield self[1]
        yield self[0]

    def __contains__(self, value):
        """Return true if the shape contains `value`, otherwise false"""
        return self[0] == value or self[1] == value

    @abstractmethod
    def __deepcopy__(self, memo=None):
        """Return a copy of the shape"""
        pass

    def __copy__(self):
        """Return a copy of the shape"""
        return self.__deepcopy__()

    @property
    def nrows(self):
        """The first dimension of the shape"""
        return self[0]

    @property
    def ncols(self):
        """The second dimension of the shape"""
        return self[1]

    def copy(self):
        """Return a copy of the shape"""
        return self.__deepcopy__()

    @abstractmethod
    def reverse(self):
        """Return the shape reversed"""
        pass
