import functools
import math
import operator

from .generic import GenericMatrix
from .ordering import OrderingMatrix

__all__ = [
    "NumericMatrix",
    "ComplexMatrix",
    "RealMatrix",
    "RationalMatrix",
    "IntegralMatrix",
]

# This feels like it should be a standard function
conjugate = operator.methodcaller("conjugate")


class NumericMatrix(GenericMatrix):
    """Subclass of `GenericMatrix` that acts as the root of all matrices
    containing numeric objects

    This type does not implement any unique methods, but can be used to check
    for matrices that strictly contain numeric objects without caring for their
    specific classification:

    ```
    if isinstance(matrix, NumericMatrix):
        # matrix is either ComplexMatrix, RealMatrix, etc.
    ```
    """

    __slots__ = ()


class ComplexMatrix(NumericMatrix):
    """Subclass of `NumericMatrix` that adds operations defined for
    `numbers.Complex` types.
    """

    __slots__ = ()

    def complex(self):
        """Element-wise complex conversion"""
        return self.unary_operator(
            complex,
            out=ComplexMatrix,
        )

    def abs(self):
        """Element-wise real distance"""
        return self.unary_operator(
            abs,
            out=RealMatrix,
        )

    def conjugate(self):
        """Element-wise conjugation"""
        cls = type(self)
        return self.unary_operator(
            conjugate,
            out=cls,
        )

    def __add__(self, other):
        """Element-wise `__add__()`"""
        cls = type(self)
        return self.binary_operator(
            operator.add,
            other,
            exp=cls,
            out=cls,
        )

    def __radd__(self, other):
        """Element-wise `__add__()`"""
        cls = type(self)
        return self.binary_operator(
            operator.add,
            other,
            exp=cls,
            out=cls,
            reverse=True,
        )

    def __sub__(self, other):
        """Element-wise `__sub__()`"""
        cls = type(self)
        return self.binary_operator(
            operator.sub,
            other,
            exp=cls,
            out=cls,
        )

    def __rsub__(self, other):
        """Element-wise `__sub__()`"""
        cls = type(self)
        return self.binary_operator(
            operator.sub,
            other,
            exp=cls,
            out=cls,
            reverse=True,
        )

    def __mul__(self, other):
        """Element-wise `__mul__()`"""
        cls = type(self)
        return self.binary_operator(
            operator.mul,
            other,
            exp=cls,
            out=cls,
        )

    def __rmul__(self, other):
        """Element-wise `__mul__()`"""
        cls = type(self)
        return self.binary_operator(
            operator.mul,
            other,
            exp=cls,
            out=cls,
            reverse=True,
        )

    def __truediv__(self, other):
        """Element-wise `__truediv__()`"""
        cls = type(self)
        return self.binary_operator(
            operator.truediv,
            other,
            exp=cls,
            out=cls,
        )

    def __rtruediv__(self, other):
        """Element-wise `__truediv__()`"""
        cls = type(self)
        return self.binary_operator(
            operator.truediv,
            other,
            exp=cls,
            out=cls,
            reverse=True,
        )

    def __pow__(self, other):
        """Element-wise `__pow__()`"""
        cls = type(self)
        return self.binary_operator(
            operator.pow,
            other,
            exp=cls,
            out=cls,
        )

    def __rpow__(self, other):
        """Element-wise `__pow__()`"""
        cls = type(self)
        return self.binary_operator(
            operator.pow,
            other,
            exp=cls,
            out=cls,
            reverse=True,
        )

    def __neg__(self):
        """Element-wise `__neg__()`"""
        cls = type(self)
        return self.unary_operator(
            operator.neg,
            out=cls,
        )

    def __pos__(self):
        """Element-wise `__pos__()`"""
        cls = type(self)
        return self.unary_operator(
            operator.pos,
            out=cls,
        )


class RealMatrix(ComplexMatrix, OrderingMatrix):
    """Subclass of `ComplexMatrix` and `OrderingMatrix` that adds operations
    defined for `numbers.Real` types.
    """

    __slots__ = ()

    def abs(self):
        """Element-wise absolute value"""
        return type(self).refer(super().abs())

    def float(self):
        """Element-wise float conversion"""
        return self.unary_operator(
            float,
            out=RealMatrix,
        )

    def trunc(self):
        """Element-wise `math.trunc()`"""
        return self.unary_operator(
            math.trunc,
            out=IntegralMatrix,
        )

    def round(self, ndigits=None):
        """Element-wise `round()`"""
        cls = IntegralMatrix if ndigits is None else RealMatrix
        return self.unary_operator(
            functools.partial(round, ndigits),
            out=cls,
        )

    def floor(self):
        """Element-wise `math.floor()`"""
        return self.unary_operator(
            math.floor,
            out=IntegralMatrix,
        )

    def ceil(self):
        """Element-wise `math.ceil()`"""
        return self.unary_operator(
            math.ceil,
            out=IntegralMatrix,
        )

    def divmod(self):
        """Element-wise `divmod()`"""
        return self.unary_operator(
            divmod,
            out=GenericMatrix,
        )

    def __floordiv__(self, other):
        """Element-wise `__floordiv__()`"""
        cls = type(self)
        return self.binary_operator(
            operator.floordiv,
            other,
            exp=cls,
            out=cls,
        )

    def __rfloordiv__(self, other):
        """Element-wise `__floordiv__()`"""
        cls = type(self)
        return self.binary_operator(
            operator.floordiv,
            other,
            exp=cls,
            out=cls,
            reverse=True,
        )

    def __mod__(self, other):
        """Element-wise `__mod__()`"""
        cls = type(self)
        return self.binary_operator(
            operator.mod,
            other,
            exp=cls,
            out=cls,
        )

    def __rmod__(self, other):
        """Element-wise `__mod__()`"""
        cls = type(self)
        return self.binary_operator(
            operator.mod,
            other,
            exp=cls,
            out=cls,
            reverse=True,
        )


class RationalMatrix(RealMatrix):
    """Subclass of `RealMatrix` that adds operations defined for
    `numbers.Rational` types

    Currently, this type does not implement any unique methods.
    """

    __slots__ = ()


class IntegralMatrix(RationalMatrix):
    """Subclass of `RationalMatrix` that adds operations defined for
    `numbers.Integral` types
    """

    __slots__ = ()

    def int(self):
        """Element-wise integer conversion"""
        return self.unary_operator(
            int,
            out=IntegralMatrix,
        )
