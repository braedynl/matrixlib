from abc import ABCMeta, abstractmethod
from collections.abc import Collection
from typing import Generic, TypeVar, Union

__all__ = ["ShapeLike"]

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class ShapeLike(Collection[Union[M_co, N_co]], Generic[M_co, N_co], metaclass=ABCMeta):
    """Abstract base class for shape-like objects

    A shape is a two-item collection of positive integers that represent a
    matrix's row and column count. The size of a matrix can be evaluated by
    taking the product of its shape.
    """

    __slots__ = ()
    __match_args__ = ("nrows", "ncols")

    def __str__(self):
        """Return a string representation of the shape"""
        return f"{self[0]} × {self[1]}"

    def __lt__(self, other):
        """Return true if lexicographic ``a < b``, otherwise false"""
        if isinstance(other, ShapeLike):
            return self.compare(other) < 0
        return NotImplemented

    def __le__(self, other):
        """Return true if lexicographic ``a <= b``, otherwise false"""
        if isinstance(other, ShapeLike):
            return self.compare(other) <= 0
        return NotImplemented

    def __eq__(self, other):
        """Return true if lexicographic ``a == b``, otherwise false"""
        if isinstance(other, ShapeLike):
            return self.compare(other) == 0
        return NotImplemented

    def __ne__(self, other):
        """Return true if lexicographic ``a != b``, otherwise false"""
        if isinstance(other, ShapeLike):
            return self.compare(other) != 0
        return NotImplemented

    def __gt__(self, other):
        """Return true if lexicographic ``a > b``, otherwise false"""
        if isinstance(other, ShapeLike):
            return self.compare(other) > 0
        return NotImplemented

    def __ge__(self, other):
        """Return true if lexicographic ``a >= b``, otherwise false"""
        if isinstance(other, ShapeLike):
            return self.compare(other) >= 0
        return NotImplemented

    def __len__(self):
        """Return literal 2"""
        return 2

    @abstractmethod
    def __getitem__(self, key):
        """Return the dimension corresponding to ``key``"""
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
        """Return true if the shape contains ``value``, otherwise false"""
        return self[0] == value or self[1] == value

    @property
    def nrows(self):
        """The number of rows"""
        return self[0]

    @property
    def ncols(self):
        """The number of columns"""
        return self[1]

    def compare(self, other):
        """Return literal -1, 0, or 1 if lexicographic ``a < b``, ``a == b``,
        or ``a > b``, respectively
        """
        if self is other:
            return 0
        for m, n in zip(self, other):
            if m == n:
                continue
            if m < n:
                return -1
            if m > n:
                return 1
            raise RuntimeError  # Unreachable
        return 0
