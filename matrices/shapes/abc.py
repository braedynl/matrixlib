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

    def __lt__(self, other):
        """Return true if lexicographic `a < b`, otherwise false"""
        if isinstance(other, ShapeLike):
            return self.compare(other) < 0
        return NotImplemented

    def __le__(self, other):
        """Return true if lexicographic `a <= b`, otherwise false"""
        if isinstance(other, ShapeLike):
            return self.compare(other) <= 0
        return NotImplemented

    def __eq__(self, other):
        """Return true if lexicographic `a == b`, otherwise false"""
        if isinstance(other, ShapeLike):
            return self.compare(other) == 0
        return NotImplemented

    def __ne__(self, other):
        """Return true if lexicographic `a != b`, otherwise false"""
        if isinstance(other, ShapeLike):
            return self.compare(other) != 0
        return NotImplemented

    def __gt__(self, other):
        """Return true if lexicographic `a > b`, otherwise false"""
        if isinstance(other, ShapeLike):
            return self.compare(other) > 0
        return NotImplemented

    def __ge__(self, other):
        """Return true if lexicographic `a >= b`, otherwise false"""
        if isinstance(other, ShapeLike):
            return self.compare(other) >= 0
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

    @property
    def nrows(self):
        """The first dimension of the shape"""
        return self[0]

    @property
    def ncols(self):
        """The second dimension of the shape"""
        return self[1]

    # A reversal operation is required at the base level for use by matrix
    # views - specifically `MatrixTranspose` and its sub-classes.

    # Its implementation may be in-place or out-of-place. If in-place, ensure
    # that a mutable reference to a composed shape *is not* exposed by the
    # `shape` property. Doing otherwise could make views mutate the target
    # shape unexpectedly.

    @abstractmethod
    def reverse(self):
        """Return the shape reversed"""
        pass

    def compare(self, other):
        """Return literal -1, 0, or 1 if lexicographic `a < b`, `a == b`, or
        `a > b`, respectively
        """
        if self is other:
            return 0
        for x, y in zip(self, other):
            if x == y:
                continue
            if x < y:
                return -1
            if x > y:
                return 1
            raise RuntimeError  # Unreachable
        return 0
