import sys
import typing
from decimal import Decimal
from fractions import Fraction
from numbers import Complex, Integral, Number, Rational, Real
from typing import SupportsIndex, TypeVar, Union

from .generic import GenericMatrix
from .ordering import OrderingMatrix

# MyPy currently has some issues with the abstracts from numbers, so these
# union types are here as a rudimentary substitute (for now, that is - don't
# rely on these being here in future versions).

AnyIntegral = Union[Integral, int]
IntegralT = TypeVar("IntegralT", bound=AnyIntegral)

AnyRational = Union[AnyIntegral, Rational, Fraction]
RationalT = TypeVar("RationalT", bound=AnyRational)

AnyReal = Union[AnyRational, Real, float]
RealT = TypeVar("RealT", bound=AnyReal)

AnyComplex = Union[AnyReal, Complex, complex]
ComplexT = TypeVar("ComplexT", bound=AnyComplex)

AnyNumber = Union[AnyComplex, Number, Decimal]
NumericT = TypeVar("NumericT", bound=AnyNumber)


class NumericMatrix(GenericMatrix[NumericT]):
    pass


class ComplexMatrix(NumericMatrix[ComplexT]):
    if sys.version_info >= (3, 11):
        Self = typing.Self
    else:
        Self = TypeVar("Self", bound="ComplexMatrix")

    def complex(self: Self) -> ComplexMatrix[complex]: ...

    # The following methods should be overloaded in subclasses to type-hint the
    # enclosing class as its operand (if applicable) and return. Using Self
    # will capture the type variable, which may not always apply.

    # MyPy's override error can be safely ignored, as demotion to a matrix
    # that's higher in the tower will be captured by the reverse operator. The
    # implementation is purely monomorphic; returning NotImplemented if the
    # operand is not an instance of the enclosing class, allowing this behavior
    # to happen.

    def abs(self: Self) -> RealMatrix: ...
    def conjugate(self: Self) -> ComplexMatrix: ...

    def __add__(self: Self, other: ComplexMatrix) -> ComplexMatrix: ...
    def __sub__(self: Self, other: ComplexMatrix) -> ComplexMatrix: ...
    def __mul__(self: Self, other: ComplexMatrix) -> ComplexMatrix: ...
    def __truediv__(self: Self, other: ComplexMatrix) -> ComplexMatrix: ...
    def __pow__(self: Self, other: ComplexMatrix) -> ComplexMatrix: ...

    def __radd__(self: Self, other: ComplexMatrix) -> ComplexMatrix: ...
    def __rsub__(self: Self, other: ComplexMatrix) -> ComplexMatrix: ...
    def __rmul__(self: Self, other: ComplexMatrix) -> ComplexMatrix: ...
    def __rtruediv__(self: Self, other: ComplexMatrix) -> ComplexMatrix: ...
    def __rpow__(self: Self, other: ComplexMatrix) -> ComplexMatrix: ...

    def __neg__(self: Self) -> ComplexMatrix: ...
    def __pos__(self: Self) -> ComplexMatrix: ...


class RealMatrix(ComplexMatrix[RealT], OrderingMatrix[RealT]):
    if sys.version_info >= (3, 11):
        Self = typing.Self
    else:
        Self = TypeVar("Self", bound="RealMatrix")

    def float(self: Self) -> RealMatrix[float]: ...
    def trunc(self: Self) -> IntegralMatrix[int]: ...
    def floor(self: Self) -> IntegralMatrix[int]: ...
    def ceil(self: Self) -> IntegralMatrix[int]: ...
    def divmod(self: Self) -> GenericMatrix: ...  # Will often return GenericMatrix[tuple[Any, Any]], but not guaranteed

    @typing.overload
    def round(self: Self) -> IntegralMatrix[int]: ...
    @typing.overload
    def round(self: Self, ndigits: SupportsIndex) -> RealMatrix: ...

    # To ComplexMatrix, RealMatrix adds __floordiv__() and __mod__(). Both need
    # to be overloaded in the same manner.

    def abs(self: Self) -> RealMatrix: ...
    def conjugate(self: Self) -> RealMatrix: ...

    def __add__(self: Self, other: RealMatrix) -> RealMatrix: ...  # type: ignore[override]
    def __sub__(self: Self, other: RealMatrix) -> RealMatrix: ...  # type: ignore[override]
    def __mul__(self: Self, other: RealMatrix) -> RealMatrix: ...  # type: ignore[override]
    def __truediv__(self: Self, other: RealMatrix) -> RealMatrix: ...  # type: ignore[override]
    def __floordiv__(self: Self, other: RealMatrix) -> RealMatrix: ...  # type: ignore[override]
    def __mod__(self: Self, other: RealMatrix) -> RealMatrix: ...  # type: ignore[override]
    def __pow__(self: Self, other: RealMatrix) -> RealMatrix: ...  # type: ignore[override]

    def __radd__(self: Self, other: RealMatrix) -> RealMatrix: ...  # type: ignore[override]
    def __rsub__(self: Self, other: RealMatrix) -> RealMatrix: ...  # type: ignore[override]
    def __rmul__(self: Self, other: RealMatrix) -> RealMatrix: ...  # type: ignore[override]
    def __rtruediv__(self: Self, other: RealMatrix) -> RealMatrix: ...  # type: ignore[override]
    def __rfloordiv__(self: Self, other: RealMatrix) -> RealMatrix: ...  # type: ignore[override]
    def __rmod__(self: Self, other: RealMatrix) -> RealMatrix: ...  # type: ignore[override]
    def __rpow__(self: Self, other: RealMatrix) -> RealMatrix: ...  # type: ignore[override]

    def __neg__(self: Self) -> RealMatrix: ...
    def __pos__(self: Self) -> RealMatrix: ...


class RationalMatrix(RealMatrix[RationalT]):
    if sys.version_info >= (3, 11):
        Self = typing.Self
    else:
        Self = TypeVar("Self", bound="RationalMatrix")

    def abs(self: Self) -> RationalMatrix: ...
    def conjugate(self: Self) -> RationalMatrix: ...

    def __add__(self: Self, other: RationalMatrix) -> RationalMatrix: ...  # type: ignore[override]
    def __sub__(self: Self, other: RationalMatrix) -> RationalMatrix: ...  # type: ignore[override]
    def __mul__(self: Self, other: RationalMatrix) -> RationalMatrix: ...  # type: ignore[override]
    def __truediv__(self: Self, other: RationalMatrix) -> RationalMatrix: ...  # type: ignore[override]
    def __floordiv__(self: Self, other: RationalMatrix) -> RationalMatrix: ...  # type: ignore[override]
    def __mod__(self: Self, other: RationalMatrix) -> RationalMatrix: ...  # type: ignore[override]
    def __pow__(self: Self, other: RationalMatrix) -> RationalMatrix: ...  # type: ignore[override]

    def __radd__(self: Self, other: RationalMatrix) -> RationalMatrix: ...  # type: ignore[override]
    def __rsub__(self: Self, other: RationalMatrix) -> RationalMatrix: ...  # type: ignore[override]
    def __rmul__(self: Self, other: RationalMatrix) -> RationalMatrix: ...  # type: ignore[override]
    def __rtruediv__(self: Self, other: RationalMatrix) -> RationalMatrix: ...  # type: ignore[override]
    def __rfloordiv__(self: Self, other: RationalMatrix) -> RationalMatrix: ...  # type: ignore[override]
    def __rmod__(self: Self, other: RationalMatrix) -> RationalMatrix: ...  # type: ignore[override]
    def __rpow__(self: Self, other: RationalMatrix) -> RationalMatrix: ...  # type: ignore[override]

    def __neg__(self: Self) -> RationalMatrix: ...
    def __pos__(self: Self) -> RationalMatrix: ...


class IntegralMatrix(RationalMatrix[IntegralT]):
    if sys.version_info >= (3, 11):
        Self = typing.Self
    else:
        Self = TypeVar("Self", bound="IntegralMatrix")

    def int(self: Self) -> IntegralMatrix[int]: ...

    def abs(self: Self) -> IntegralMatrix: ...
    def conjugate(self: Self) -> IntegralMatrix: ...

    def __add__(self: Self, other: IntegralMatrix) -> IntegralMatrix: ...  # type: ignore[override]
    def __sub__(self: Self, other: IntegralMatrix) -> IntegralMatrix: ...  # type: ignore[override]
    def __mul__(self: Self, other: IntegralMatrix) -> IntegralMatrix: ...  # type: ignore[override]
    def __truediv__(self: Self, other: IntegralMatrix) -> IntegralMatrix: ...  # type: ignore[override]
    def __floordiv__(self: Self, other: IntegralMatrix) -> IntegralMatrix: ...  # type: ignore[override]
    def __mod__(self: Self, other: IntegralMatrix) -> IntegralMatrix: ...  # type: ignore[override]
    def __pow__(self: Self, other: IntegralMatrix) -> IntegralMatrix: ...  # type: ignore[override]

    def __radd__(self: Self, other: IntegralMatrix) -> IntegralMatrix: ...  # type: ignore[override]
    def __rsub__(self: Self, other: IntegralMatrix) -> IntegralMatrix: ...  # type: ignore[override]
    def __rmul__(self: Self, other: IntegralMatrix) -> IntegralMatrix: ...  # type: ignore[override]
    def __rtruediv__(self: Self, other: IntegralMatrix) -> IntegralMatrix: ...  # type: ignore[override]
    def __rfloordiv__(self: Self, other: IntegralMatrix) -> IntegralMatrix: ...  # type: ignore[override]
    def __rmod__(self: Self, other: IntegralMatrix) -> IntegralMatrix: ...  # type: ignore[override]
    def __rpow__(self: Self, other: IntegralMatrix) -> IntegralMatrix: ...  # type: ignore[override]

    def __neg__(self: Self) -> IntegralMatrix: ...
    def __pos__(self: Self) -> IntegralMatrix: ...
