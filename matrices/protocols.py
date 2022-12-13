import copy
from abc import abstractmethod
from collections.abc import Iterator
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from .rule import Rule

__all__ = [
    "ShapeLike",
    "MatrixLike",
]

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class ShapeLike(Protocol):
    """Protocol of operations defined for shape-like objects

    Note that this protocol can match with matrix types through `isinstance()`,
    though such a match is considered invalid.
    """

    def __eq__(self, other: Any) -> bool:
        """Return true if the two shapes are equal, otherwise false"""
        if isinstance(other, ShapeLike):
            return (
                self[0] == other[0]
                and
                self[1] == other[1]
            )
        return NotImplemented

    def __len__(self) -> Literal[2]:
        """Return literal 2"""
        return 2

    @abstractmethod
    def __getitem__(self, key: int) -> int:
        """Return the dimension corresponding to `key`"""
        pass

    def __iter__(self) -> Iterator[int]:
        """Return an iterator over the dimensions of the shape"""
        yield self[0]
        yield self[1]

    def __reversed__(self) -> Iterator[int]:
        """Return a reversed iterator over the dimensions of the shape"""
        yield self[1]
        yield self[0]

    def __contains__(self, value: Any) -> bool:
        """Return true if the shape contains `value`, otherwise false"""
        return self[0] == value or self[1] == value

    @property
    def nrows(self) -> int:
        """The first dimension of the shape"""
        return self[0]

    @property
    def ncols(self) -> int:
        """The second dimension of the shape"""
        return self[1]


@runtime_checkable
class MatrixLike(Protocol[T_co]):
    """Protocol of operations defined for matrix-like objects"""

    @abstractmethod
    def __eq__(self, other):
        """Return true if element-wise `a == b` is true for all element pairs,
        otherwise false

        For a matrix of each comparison result, use the `eq()` method.
        """
        pass

    def __len__(self):
        """Return the matrix's size"""
        return self.size

    @abstractmethod
    def __getitem__(self, key):
        """Return the element or sub-matrix corresponding to `key`"""
        pass

    def __iter__(self):
        """Return an iterator over the values of the matrix in row-major order"""
        for i in range(self.size):
            yield self[i]

    def __reversed__(self):
        """Return an iterator over the values of the matrix in reverse
        row-major order
        """
        for i in reversed(range(self.size)):
            yield self[i]

    def __contains__(self, value):
        """Return true if the matrix contains `value`, otherwise false"""
        return any(map(lambda x: x is value or x == value), self)

    @abstractmethod
    def __copy__(self):
        """Return a shallow copy of the matrix"""
        pass

    @abstractmethod
    def __and__(self, other):
        """Return element-wise `logical_and(a, b)`"""
        pass

    @abstractmethod
    def __rand__(self, other):
        """Return element-wise `logical_and(b, a)`"""
        pass

    @abstractmethod
    def __or__(self, other):
        """Return element-wise `logical_or(a, b)`"""
        pass

    @abstractmethod
    def __ror__(self, other):
        """Return element-wise `logical_or(b, a)`"""
        pass

    @abstractmethod
    def __xor__(self, other):
        """Return element-wise `logical_xor(a, b)`"""
        pass

    @abstractmethod
    def __rxor__(self, other):
        """Return element-wise `logical_xor(b, a)`"""
        pass

    @abstractmethod
    def __invert__(self):
        """Return element-wise `logical_not(a)`"""
        pass

    @property
    @abstractmethod
    def shape(self):
        """A collection of the matrix's dimensions"""
        pass

    @property
    def nrows(self):
        """The matrix's number of rows"""
        return self.shape.nrows

    @property
    def ncols(self):
        """The matrix's number of columns"""
        return self.shape.ncols

    @property
    def size(self):
        """The product of the matrix's number of rows and columns"""
        nrows, ncols = self.shape
        return nrows * ncols

    # XXX: For vectorized operations like eq() and ne(), a new matrix composed
    # of the results from the mapped operation should be returned. ValueError
    # should be raised if two matrices have unequal shapes. If the right-hand
    # side is not a matrix, the object should be repeated for each operation
    # performed on a matrix entry (essentially becoming a matrix of the same
    # shape, filled entirely by the object).

    @abstractmethod
    def eq(self, other):
        """Return element-wise `a == b`"""
        pass

    @abstractmethod
    def ne(self, other):
        """Return element-wise `a != b`"""
        pass

    # XXX: By convention, methods that can be interpreted for either direction
    # take a keyword argument, "by", that switches the method's interpretation
    # between row and column-wise.
    # 0 is used to dictate row-wise, 1 is used to dictate column-wise. All
    # methods that can be interpreted in this manner should default to
    # row-wise.

    @abstractmethod
    def slices(self, *, by=Rule.ROW):
        """Return an iterator that yields shallow copies of each row or column"""
        pass

    def copy(self):
        """Return a shallow copy of the matrix"""
        return copy.copy(self)
