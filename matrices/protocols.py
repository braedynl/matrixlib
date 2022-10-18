import copy
import sys
from abc import abstractmethod
from typing import Protocol, runtime_checkable

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
# The exception to this rule is true division and exponentiation.
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
# All comparison operators should return instances of IntegralLike. Numeric
# matrix types will return instances of IntegralMatrixLike.


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

    # XXX: this method is only defined for 3.11 or higher due to the built-in
    # complex type not having it in prior versions (thus making it fail
    # instance checks). Implementing this method is recommended.

    if sys.version_info >= (3, 11):

        @abstractmethod
        def __complex__(self):
            """Return an equivalent `complex` instance"""
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

    @abstractmethod
    def __float__(self):
        """Return an equivalent `float` instance"""
        pass


@runtime_checkable
class IntegralLike(RealLike, Protocol):
    """Protocol of operations defined for integral-like objects

    Derives from `RealLike`.
    """

    @abstractmethod
    def __int__(self):
        """Return an equivalent `int` instance"""
        pass

    def __index__(self):
        """Return an equivalent `int` instance, losslessly"""
        return int(self)


@runtime_checkable
class ShapeLike(Protocol):
    """Protocol of operations defined for shape-like objects

    Note that this protocol can match with matrix types, though such a match is
    considered invalid. A shape's length is always 2, and its size is the
    product of its elements - a matrix's length and size are identical, being
    the size of its shape.
    """

    def __eq__(self, other):
        """Return true if the two shapes are equal, otherwise false

        Under this protocol, shapes are considered equal if at least one of
        the following criteria is met:
        - The shapes are element-wise equivalent
        - The shapes' products are equivalent, and both contain at least one
          dimension equal to 1 (i.e., both could be represented
          one-dimensionally)

        For element-wise equivalence alone, use the `equal()` method.
        """
        if not isinstance(other, ShapeLike):
            return NotImplemented
        return self.equal(other) or (self.size == other.size and 1 in self and 1 in other)

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

    @property
    def size(self):
        """The product of the shape's dimensions"""
        nrows, ncols = self
        return nrows * ncols

    def equal(self, other):
        """Return true if the two shapes are element-wise equivalent, otherwise
        false
        """
        if self is other:
            return True
        return self[0] == other[0] and self[1] == other[1]


@runtime_checkable
class MatrixLike(Protocol):
    """Protocol of operations defined for matrix-like objects"""

    # XXX: For vectorized operations like __eq__() and __ne__(), a new matrix
    # composed of the results from the mapped operation should be returned,
    # with a shape equivalent to that of the left-hand side matrix (i.e.,
    # self).
    # ValueError should be raised if two matrices have unequal shapes -
    # equality being in terms of how the ShapeLike protocol defines shape
    # equality.
    # If the right-hand side is not a matrix, the object should be "dragged"
    # along the map. Example implementation:

    # def __eq__(self, other):
    #     if isinstance(other, MatrixLike):
    #         if self.shape != other.shape:
    #             raise ValueError
    #         it = iter(other)
    #     else:
    #         it = itertools.repeat(other)
    #     return MyMatrix(map(operator.eq, self, it), *self.shape)

    @abstractmethod
    def __eq__(self, other):
        """Return element-wise `a == b`"""
        pass

    @abstractmethod
    def __ne__(self, other):
        """Return element-wise `a != b`"""
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

    __rand__ = __and__

    @abstractmethod
    def __or__(self, other):
        """Return element-wise `logical_or(a, b)`"""
        pass

    __ror__ = __or__

    @abstractmethod
    def __xor__(self, other):
        """Return element-wise `logical_xor(a, b)`"""
        pass

    __rxor__ = __xor__

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
        return self.shape.size

    def equal(self, other):
        """Return true if the two matrices have an element-wise equivalent data
        buffer and shape, otherwise false
        """
        if self is other:
            return True

        h, k = self.shape, other.shape

        def equal(x, y):
            if x is y:
                return True
            n = isinstance(x, MatrixLike) + isinstance(y, MatrixLike)
            if n == 2:
                return x.equal(y)
            if n == 1:
                return False
            return x == y

        return h.equal(k) and all(map(equal, self, other))

    def copy(self):
        """Return a shallow copy of the matrix"""
        return copy.copy(self)

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


@runtime_checkable
class ComplexMatrixLike(MatrixLike, Protocol):
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

    @abstractmethod
    def __rmatmul__(self, other):
        """Return the reverse matrix product `b @ a`"""
        pass

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

    # XXX: Same version restriction reason as described by ComplexLike

    if sys.version_info >= (3, 11):

        @abstractmethod
        def __complex__(self):
            """Return an equivalent `complex` instance"""
            pass

    @abstractmethod
    def conjugate(self):
        """Return element-wise `conjugate(a)`"""
        pass

    @abstractmethod
    def complex(self):
        """Return a matrix of each value's equivalent `complex` instance"""
        pass


@runtime_checkable
class RealMatrixLike(ComplexMatrixLike, Protocol):
    """Protocol of operations defined for matrix-like objects that contain
    real-like values

    Derives from `ComplexMatrixLike`.
    """

    @abstractmethod
    def __lt__(self, other):
        """Return element-wise `a < b`"""
        pass

    @abstractmethod
    def __le__(self, other):
        """Return element-wise `a <= b`"""
        pass

    @abstractmethod
    def __gt__(self, other):
        """Return element-wise `a > b`"""
        pass

    @abstractmethod
    def __ge__(self, other):
        """Return element-wise `a >= b`"""
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
    def __float__(self):
        """Return an equivalent `float` instance"""
        pass

    @abstractmethod
    def float(self):
        """Return a matrix of each value's equivalent `float` instance"""
        pass


@runtime_checkable
class IntegralMatrixLike(RealMatrixLike, Protocol):
    """Protocol of operations defined for matrix-like objects that contain
    integral-like values

    Derives from `RealMatrixLike`.
    """

    @abstractmethod
    def __int__(self):
        """Return an equivalent `int` instance"""
        pass

    def __index__(self):
        """Return an equivalent `int` instance, losslessly"""
        return int(self)

    @abstractmethod
    def int(self):
        """Return a matrix of each value's equivalent `int` instance"""
        pass
