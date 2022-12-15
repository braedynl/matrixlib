from abc import ABCMeta, abstractmethod
from collections.abc import Collection
from typing import Generic, Literal, TypeVar

__all__ = [
    "ShapeLike",
    "AnyShape",
    "AnyRowVectorShape",
    "AnyColVectorShape",
    "AnyVectorShape",
]

NRowsT_co = TypeVar("NRowsT_co", covariant=True, bound=int)
NColsT_co = TypeVar("NColsT_co", covariant=True, bound=int)


class ShapeLike(Collection[NRowsT_co | NColsT_co], Generic[NRowsT_co, NColsT_co], metaclass=ABCMeta):

    __match_args__ = ("nrows", "ncols")

    def __eq__(self, other):
        """Return true if the two shapes are equal, otherwise false"""
        if isinstance(other, ShapeLike):
            return (
                self[0] == other[0]
                and
                self[1] == other[1]
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

    @property
    def nrows(self):
        """The first dimension of the shape"""
        return self[0]

    @property
    def ncols(self):
        """The second dimension of the shape"""
        return self[1]


AnyShape = ShapeLike[int, int]
AnyRowVectorShape = ShapeLike[Literal[1], int]
AnyColVectorShape = ShapeLike[int, Literal[1]]
AnyVectorShape = AnyRowVectorShape | AnyColVectorShape
