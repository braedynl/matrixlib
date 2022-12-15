from typing import Protocol, TypeVar

__all__ = [
    "SupportsLT",
    "SupportsLE",
    "SupportsGT",
    "SupportsGE",
    "SupportsLTAndGT",
    "SupportsLEAndGE",
    "SupportsComparison",
    "SupportsAdd",
    "SupportsSub",
    "SupportsMul",
    "SupportsTrueDiv",
    "SupportsFloorDiv",
    "SupportsMod",
    "SupportsPow",
    "SupportsRAdd",
    "SupportsRSub",
    "SupportsRMul",
    "SupportsRTrueDiv",
    "SupportsRFloorDiv",
    "SupportsRMod",
    "SupportsRPow",
    "SupportsNeg",
    "SupportsPos",
    "SupportsAbs",
    "SupportsConjugate",
]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


class SupportsLT(Protocol[T_contra]):
    def __lt__(self, other: T_contra) -> bool: ...
class SupportsLE(Protocol[T_contra]):
    def __le__(self, other: T_contra) -> bool: ...
class SupportsGT(Protocol[T_contra]):
    def __gt__(self, other: T_contra) -> bool: ...
class SupportsGE(Protocol[T_contra]):
    def __ge__(self, other: T_contra) -> bool: ...

class SupportsLTAndGT(SupportsLT[T_contra], SupportsGT[T_contra], Protocol[T_contra]): ...
class SupportsLEAndGE(SupportsLE[T_contra], SupportsGE[T_contra], Protocol[T_contra]): ...
class SupportsComparison(SupportsLTAndGT[T_contra], SupportsLEAndGE[T_contra], Protocol[T_contra]): ...

class SupportsAdd(Protocol[T_contra, T_co]):
    def __add__(self, other: T_contra) -> T_co: ...
class SupportsSub(Protocol[T_contra, T_co]):
    def __sub__(self, other: T_contra) -> T_co: ...
class SupportsMul(Protocol[T_contra, T_co]):
    def __mul__(self, other: T_contra) -> T_co: ...
class SupportsTrueDiv(Protocol[T_contra, T_co]):
    def __truediv__(self, other: T_contra) -> T_co: ...
class SupportsFloorDiv(Protocol[T_contra, T_co]):
    def __floordiv__(self, other: T_contra) -> T_co: ...
class SupportsMod(Protocol[T_contra, T_co]):
    def __mod__(self, other: T_contra) -> T_co: ...
class SupportsPow(Protocol[T_contra, T_co]):
    def __pow__(self, other: T_contra) -> T_co: ...

class SupportsRAdd(Protocol[T_contra, T_co]):
    def __radd__(self, other: T_contra) -> T_co: ...
class SupportsRSub(Protocol[T_contra, T_co]):
    def __rsub__(self, other: T_contra) -> T_co: ...
class SupportsRMul(Protocol[T_contra, T_co]):
    def __rmul__(self, other: T_contra) -> T_co: ...
class SupportsRTrueDiv(Protocol[T_contra, T_co]):
    def __rtruediv__(self, other: T_contra) -> T_co: ...
class SupportsRFloorDiv(Protocol[T_contra, T_co]):
    def __rfloordiv__(self, other: T_contra) -> T_co: ...
class SupportsRMod(Protocol[T_contra, T_co]):
    def __rmod__(self, other: T_contra) -> T_co: ...
class SupportsRPow(Protocol[T_contra, T_co]):
    def __rpow__(self, other: T_contra) -> T_co: ...

class SupportsNeg(Protocol[T_co]):
    def __neg__(self) -> T_co: ...
class SupportsPos(Protocol[T_co]):
    def __pos__(self) -> T_co: ...
class SupportsAbs(Protocol[T_co]):
    def __abs__(self) -> T_co: ...
class SupportsConjugate(Protocol[T_co]):
    def conjugate(self) -> T_co: ...
