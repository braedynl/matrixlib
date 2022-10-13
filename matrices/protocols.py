import sys
from abc import abstractmethod
from typing import Protocol, runtime_checkable

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
# All binary operators (except / and **, more on these two later) should return
# instances of themselves or some instance of the subclassed protocol when the
# operand is an instance of the subclassed protocol. For example, a subclass of
# RealLike should implement an __add__() and __radd__() that takes an instance
# of RealLike, and produces a RealLike as a result (either a new instance of
# the enclosing class, or a "built-in real" such as float or int).
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


# These protocols are implemented as a hierarchy. The numeric matrix types,
# too, are implemented as a hierarchy - but, there is a lot of ongoing debate
# on whether numeric types should be implemented as a hierachy at all, since
# it often requires violating the Liskov Substitution Principle.
#
# Implementors are free to choose how to structure their subclasses - the
# numeric matrices are implemented hierarchically as a means to reduce some
# repetitious code, while advising users to perform instance checks with
# the protocols rather than with one of the concrete implementations (since
# their registration as subclasses may change in the future).


@runtime_checkable
class ComplexLike(Protocol):
    """Protocol of operations defined for complex-like objects

    Acts as the root of the numeric-like protocol tower.
    """

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

    Note that this protocol can match with matrix types. This is only ever
    valid if the matrix is of size 2.
    """

    def __len__(self):
        """Return literal 2"""
        return 2

    # XXX: Implementors should coerce keys to integers via operator.index()
    # so that enums like Rule can be used as keys

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
        return value == self[0] or value == self[1]

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


@runtime_checkable
class MatrixLike(Protocol):
    """Protocol of operations defined for matrix-like objects"""

    def __len__(self):
        """Return the matrix's size"""
        return self.size

    @abstractmethod
    def __getitem__(self, key):
        """Return the element or sub-matrix corresponding to `key`"""
        pass

    def __iter__(self):
        """Return an iterator over the values of the matrix in row-major order"""
        n = len(self)
        for i in range(n):
            yield self[i]

    def __reversed__(self):
        """Return an iterator over the values of the matrix in reverse
        row-major order
        """
        n = len(self)
        for i in reversed(range(n)):
            yield self[i]

    def __contains__(self, value):
        """Return true if the matrix contains `value`, otherwise false"""
        for x in self:
            if x is value or x == value:
                return True
        return False

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


@runtime_checkable
class ComplexMatrixLike(ComplexLike, MatrixLike, Protocol):
    """Protocol of operations defined for matrix-like objects that contain
    complex-like values

    Derives from `ComplexLike` and `MatrixLike`.
    """

    @abstractmethod
    def __matmul__(self, other):
        """Return `a @ b`"""
        pass

    @abstractmethod
    def __rmatmul__(self, other):
        """Return `b @ a`"""
        pass

    @abstractmethod
    def complex(self):
        """Return a matrix of each value's equivalent `complex` instance"""
        pass


@runtime_checkable
class RealMatrixLike(RealLike, ComplexMatrixLike, Protocol):
    """Protocol of operations defined for matrix-like objects that contain
    real-like values

    Derives from `RealLike` and `ComplexMatrixLike`.
    """

    @abstractmethod
    def __gt__(self, other):
        """Return `a > b`"""
        pass

    @abstractmethod
    def __ge__(self, other):
        """Return `a >= b`"""
        pass

    @abstractmethod
    def float(self):
        """Return a matrix of each value's equivalent `float` instance"""
        pass


@runtime_checkable
class IntegralMatrixLike(IntegralLike, RealMatrixLike, Protocol):
    """Protocol of operations defined for matrix-like objects that contain
    integral-like values

    Derives from `IntegralLike` and `RealMatrixLike`.
    """

    @abstractmethod
    def int(self):
        """Return a matrix of each value's equivalent `int` instance"""
        pass
