from abc import abstractmethod
from typing import Any, Optional, Protocol, TypeVar, Union, overload

__all__ = [
    "SupportsDunderLTAndLE",
    "SupportsDunderGTAndGE",
    "SupportsComparison",
    "SupportsConjugate",
    "Complex",
    "Real",
]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


class SupportsDunderLTAndLE(Protocol):
    def __lt__(self, other: Any) -> bool: ...
    def __le__(self, other: Any) -> bool: ...

class SupportsDunderGTAndGE(Protocol):
    def __gt__(self, other: Any) -> bool: ...
    def __ge__(self, other: Any) -> bool: ...

SupportsComparison = Union[SupportsDunderLTAndLE, SupportsDunderGTAndGE]


class SupportsConjugate(Protocol[T_co]):
    def conjugate(self) -> T_co: ...


ComplexT = TypeVar("ComplexT", bound="Complex")

class Complex(Protocol):
    """Protocol for operations required of values in `ComplexMatrix` instances

    This protocol defines a subset of methods found in `numbers.Complex`.
    Conversion methods (e.g., `__complex__()`, `__bool__()`) are dropped, along
    with their `real` and `imag` components.

    For binary operations such as `__add__()` and `__mul__()`, the result
    should be complex if the operands are complex. This is not reflected by
    their signatures, as implementors may include additional behavior for types
    beyond complex values.
    """

    @abstractmethod
    def __eq__(self: ComplexT, other: Any) -> bool:
        """Return self == other"""
        pass

    @abstractmethod
    def __add__(self: ComplexT, other: Any) -> Any:
        """Return self + other"""
        pass

    def __sub__(self: ComplexT, other: Any) -> Any:
        """Return self - other"""
        return self + -other

    @abstractmethod
    def __mul__(self: ComplexT, other: Any) -> Any:
        """Return self * other"""
        pass

    @abstractmethod
    def __truediv__(self: ComplexT, other: Any) -> Any:
        """Return self / other"""
        pass

    @abstractmethod
    def __pow__(self: ComplexT, other: Any) -> Any:
        """Return self ** other"""
        pass

    @abstractmethod
    def __radd__(self: ComplexT, other: Any) -> Any:
        """Return other + self"""
        pass

    def __rsub__(self: ComplexT, other: Any) -> Any:
        """Return other - self"""
        return -self + other

    @abstractmethod
    def __rmul__(self: ComplexT, other: Any) -> Any:
        """Return other * self"""
        pass

    @abstractmethod
    def __rtruediv__(self: ComplexT, other: Any) -> Any:
        """Return other / self"""
        pass

    @abstractmethod
    def __rpow__(self: ComplexT, other: Any) -> Any:
        """Return other ** self"""
        pass

    @abstractmethod
    def __neg__(self: ComplexT) -> ComplexT:
        """Return -self"""
        pass

    @abstractmethod
    def __pos__(self: ComplexT) -> ComplexT:
        """Return +self"""
        pass

    @abstractmethod
    def __abs__(self: ComplexT) -> "Real":
        """Return the number's real distance from 0"""
        pass

    @abstractmethod
    def conjugate(self: ComplexT) -> ComplexT:
        """Return the number's complex conjugate"""
        pass


# decimal.Decimal will pass with this protocol, but note that Decimal instances
# *do not* interoperate with built-in floats (which is why it's not registered
# with numbers.Real) - an unfortunate but worthy sacrifice that's made for
# better typing precision.

RealT = TypeVar("RealT", bound="Real")

class Real(Complex, Protocol):
    """Protocol for operations required of values in `RealMatrix` instances

    To `Complex`, `Real` adds requirements for some methods found in
    `numbers.Real`. `__float__()` and the special methods for `divmod()` are
    dropped.

    Like complex values, binary operations should return new reals if their
    operands are real. Implementors are free to handle demotion to a complex
    result if the operands are non-real complex values - demotion and promotion
    are free to happen anywhere, so long as it happens somewhere.
    """

    @abstractmethod
    def __lt__(self: RealT, other: Any) -> bool:
        """Return self < other"""
        pass

    @abstractmethod
    def __le__(self: RealT, other: Any) -> bool:
        """Return self <= other"""
        pass

    @abstractmethod
    def __trunc__(self: RealT) -> int:
        """Return the number truncated to an integer"""
        pass

    @abstractmethod
    def __floor__(self: RealT) -> int:
        """Return the number floored to an integer"""
        pass

    @abstractmethod
    def __ceil__(self: RealT) -> int:
        """Return the number ceiled to an integer"""
        pass

    @abstractmethod
    @overload
    def __round__(self: RealT) -> int:
        pass

    @abstractmethod
    @overload
    def __round__(self: RealT, ndigits: int) -> RealT:  # Implementors may use SupportsIndex, some (like Fraction) narrow to int
        pass

    def __round__(self: RealT, ndigits: Optional[int] = None) -> Union[int, RealT]:
        """Return the number rounded to an integer or real"""
        pass

    @abstractmethod
    def __floordiv__(self: RealT, other: Any) -> Any:
        """Return self // other"""
        pass

    @abstractmethod
    def __rfloordiv__(self: RealT, other: Any) -> Any:
        """Return other // self"""
        pass

    @abstractmethod
    def __mod__(self: RealT, other: Any) -> Any:
        """Return self % other"""
        pass

    @abstractmethod
    def __rmod__(self: RealT, other: Any) -> Any:
        """Return other % self"""
        pass

    @abstractmethod
    def __abs__(self: RealT) -> RealT:
        """Return the number's absolute value"""
        pass

    def conjugate(self: RealT) -> RealT:
        """Return the number

        Complex conjugation is a no-op for reals.
        """
        return +self
