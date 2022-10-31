import copy
from abc import abstractmethod
from typing import Protocol, TypeVar, runtime_checkable

from .rule import Rule

__all__ = [
    "ComplexLike",
    "RealLike",
    "IntegralLike",
    "ShapeLike",
    "MatrixLike",
    "ComplexMatrixLike",
    "RealMatrixLike",
    "IntegralMatrixLike",
]


# XXX: Notice for implementors
#
# All arithmetic binary operators (except / and **, more on these two later)
# should return instances of themselves or some instance of the subclassed
# protocol when the operand is an instance of the subclassed protocol. For
# example, a subclass of RealLike should implement an __add__() and __radd__()
# that takes an instance of RealLike, and produces a RealLike as a result
# (either a new instance of the enclosing class, or a "built-in real" such as
# float or int).
#
# True division should follow the rule above in all subclasses up to (but not
# including) IntegralLike. Subclasses of IntegralLike may produce a RealLike,
# but are free to be more specific.
#
# Exponentiation may produce a ComplexLike in all subclasses, and is free to be
# more specific. The numeric type as a result of exponentiation changes with
# value - not type - and so instances of ComplexMatrix will always return a
# ComplexMatrix in their __pow__()/__rpow__() overloads to be type-safe without
# spending time evaluating the contents of the matrices.
#
# __abs__() should return a RealLike in subclasses that are specifically
# complex-likes. If real-like or narrower, it should return an instance of the
# subclassed protocol (a complex number's absolute value is its real distance,
# real numbers' absolute value can be a new real number).
#
# All sub-classes should implement a built-in numeric conversion. At minimum,
# complex-likes should implement __complex__(), real-likes should implement
# __float__(), and integral-likes should implement __int__()


# These protocols are implemented as a hierarchy, but note that the concrete
# numeric matrix implementations are not - there is a lot of ongoing debate on
# whether numeric types should be implemented hierarchically, since it often
# requires violating the Liskov Substitution Principle.
# Implementors are free to choose how to structure their subclasses, but do
# bear in mind the potential consequences in creating a concrete hierarchy
# rather than individual concrete implementations.


@runtime_checkable
class ComplexLike(Protocol):
    """Protocol of operations defined for complex-like objects

    Acts as the root of the numeric-like protocol tower.
    """

    @abstractmethod
    def __eq__(self, other):
        """Return `a == b`"""
        pass

    @abstractmethod
    def __add__(self, other):
        """Return `a + b`"""
        pass

    @abstractmethod
    def __radd__(self, other):
        """Return `b + a`"""
        pass

    def __sub__(self, other):
        """Return `a - b`"""
        return self + -other

    def __rsub__(self, other):
        """Return `b - a`"""
        return other + -self

    @abstractmethod
    def __mul__(self, other):
        """Return `a * b`"""
        pass

    @abstractmethod
    def __rmul__(self, other):
        """Return `b * a`"""
        pass

    @abstractmethod
    def __truediv__(self, other):
        """Return `a / b`"""
        pass

    @abstractmethod
    def __rtruediv__(self, other):
        """Return `b / a`"""
        pass

    @abstractmethod
    def __pow__(self, other):
        """Return `a ** b`"""
        pass

    @abstractmethod
    def __rpow__(self, other):
        """Return `b ** a`"""
        pass

    @abstractmethod
    def __neg__(self):
        """Return `-a`"""
        pass

    @abstractmethod
    def __pos__(self):
        """Return `+a`"""
        pass

    @abstractmethod
    def __abs__(self):
        """Return `abs(a)`"""
        pass

    @abstractmethod
    def conjugate(self):
        """Return `conjugate(a)`"""
        pass


@runtime_checkable
class RealLike(ComplexLike, Protocol):
    """Protocol of operations defined for real-like objects

    Derives from `ComplexLike`.
    """

    @abstractmethod
    def __lt__(self, other):
        """Return `a < b`"""
        pass

    @abstractmethod
    def __le__(self, other):
        """Return `a <= b`"""
        pass

    @abstractmethod
    def __floordiv__(self, other):
        """Return `a // b`"""
        pass

    @abstractmethod
    def __rfloordiv__(self, other):
        """Return `b // a`"""
        pass

    @abstractmethod
    def __mod__(self, other):
        """Return `a % b`"""
        pass

    @abstractmethod
    def __rmod__(self, other):
        """Return `b % a`"""
        pass


@runtime_checkable
class IntegralLike(RealLike, Protocol):
    """Protocol of operations defined for integral-like objects

    Derives from `RealLike`.
    """

    @abstractmethod
    def __index__(self):
        """Return an equivalent `int` instance, losslessly"""
        pass


@runtime_checkable
class ShapeLike(Protocol):
    """Protocol of operations defined for shape-like objects

    Note that this protocol can match with matrix types through `isinstance()`,
    though such a match is considered invalid.
    """

    def __eq__(self, other):
        """Return true if the two shapes are equal, otherwise false"""
        if not isinstance(other, ShapeLike):
            return NotImplemented
        return self[0] == other[0] and self[1] == other[1]

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


T_co = TypeVar("T_co", covariant=True)

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


ComplexLikeT_co = TypeVar("ComplexLikeT_co", bound=ComplexLike, covariant=True)

@runtime_checkable
class ComplexMatrixLike(MatrixLike[ComplexLikeT_co], Protocol[ComplexLikeT_co]):
    """Protocol of operations defined for matrix-like objects that contain
    complex-like values

    Derives from `MatrixLike`.
    """

    @abstractmethod
    def __add__(self, other):
        """Return element-wise `a + b`"""
        pass

    @abstractmethod
    def __radd__(self, other):
        """Return element-wise `b + a`"""
        pass

    def __sub__(self, other):
        """Return element-wise `a - b`"""
        return self + -other

    def __rsub__(self, other):
        """Return element-wise `b - a`"""
        return other + -self

    @abstractmethod
    def __mul__(self, other):
        """Return element-wise `a * b`"""
        pass

    @abstractmethod
    def __rmul__(self, other):
        """Return element-wise `b * a`"""
        pass

    @abstractmethod
    def __truediv__(self, other):
        """Return element-wise `a / b`"""
        pass

    @abstractmethod
    def __rtruediv__(self, other):
        """Return element-wise `b / a`"""
        pass

    @abstractmethod
    def __pow__(self, other):
        """Return element-wise `a ** b`"""
        pass

    @abstractmethod
    def __rpow__(self, other):
        """Return element-wise `b ** a`"""
        pass

    @abstractmethod
    def __matmul__(self, other):
        """Return the matrix product `a @ b`"""
        pass

    # XXX: No abstract reverse equivalent for __matmul__(), since it should be
    # monomorphic - can be provided if necessary.

    @abstractmethod
    def __neg__(self):
        """Return element-wise `-a`"""
        pass

    @abstractmethod
    def __pos__(self):
        """Return element-wise `+a`"""
        pass

    @abstractmethod
    def __abs__(self):
        """Return element-wise `abs(a)`"""
        pass

    @abstractmethod
    def conjugate(self):
        """Return element-wise `conjugate(a)`"""
        pass

    @abstractmethod
    def complex(self):
        """Return element-wise `complex(a)`"""
        pass


RealLikeT_co = TypeVar("RealLikeT_co", bound=RealLike, covariant=True)

@runtime_checkable
class RealMatrixLike(ComplexMatrixLike[RealLikeT_co], Protocol[RealLikeT_co]):
    """Protocol of operations defined for matrix-like objects that contain
    real-like values

    Derives from `ComplexMatrixLike`.
    """

    @abstractmethod
    def __lt__(self, other):
        """Return true if lexicographic `a < b`, otherwise false

        For a matrix of each comparison result, use the `lt()` method.
        """
        pass

    @abstractmethod
    def __le__(self, other):
        """Return true if lexicographic `a <= b`, otherwise false

        For a matrix of each comparison result, use the `le()` method.
        """
        pass

    @abstractmethod
    def __gt__(self, other):
        """Return true if lexicographic `a > b`, otherwise false

        For a matrix of each comparison result, use the `gt()` method.
        """
        pass

    @abstractmethod
    def __ge__(self, other):
        """Return true if lexicographic `a >= b`, otherwise false

        For a matrix of each comparison result, use the `ge()` method.
        """
        pass

    @abstractmethod
    def __floordiv__(self, other):
        """Return element-wise `a // b`"""
        pass

    @abstractmethod
    def __rfloordiv__(self, other):
        """Return element-wise `b // a`"""
        pass

    @abstractmethod
    def __mod__(self, other):
        """Return element-wise `a % b`"""
        pass

    @abstractmethod
    def __rmod__(self, other):
        """Return element-wise `b % a`"""
        pass

    @abstractmethod
    def lt(self, other):
        """Return element-wise `a < b`"""
        pass

    @abstractmethod
    def le(self, other):
        """Return element-wise `a <= b`"""
        pass

    @abstractmethod
    def gt(self, other):
        """Return element-wise `a > b`"""
        pass

    @abstractmethod
    def ge(self, other):
        """Return element-wise `a >= b`"""
        pass

    @abstractmethod
    def float(self):
        """Return element-wise `float(a)`"""
        pass


IntegralLikeT_co = TypeVar("IntegralLikeT_co", bound=ComplexLike, covariant=True)

@runtime_checkable
class IntegralMatrixLike(RealMatrixLike[IntegralLikeT_co], Protocol[IntegralLikeT_co]):
    """Protocol of operations defined for matrix-like objects that contain
    integral-like values

    Derives from `RealMatrixLike`.
    """

    # XXX: Demotion to a scalar object can only occur if, and only if, the
    # matrix's size is 1

    @abstractmethod
    def __index__(self):
        """Return the matrix as an `int` instance, losslessly"""
        pass

    @abstractmethod
    def int(self):
        """Return element-wise `int(a)`"""
        pass
