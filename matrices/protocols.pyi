import sys
from abc import abstractmethod
from collections.abc import Iterator
from typing import Any, Literal, Protocol, TypeVar, overload, runtime_checkable

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

T_co = TypeVar("T_co", covariant=True)
ComplexLikeT_co = TypeVar("ComplexLikeT_co", bound=ComplexLike, covariant=True)
RealLikeT_co = TypeVar("RealLikeT_co", bound=RealLike, covariant=True)
IntegralLikeT_co = TypeVar("IntegralLikeT_co", bound=IntegralLike, covariant=True)

Self = TypeVar("Self")

Complex = complex  # Aliases for some built-ins, since these names may get used
Float = float      # for methods
Int = int
Slice = slice


@runtime_checkable
class ComplexLike(Protocol):

    @abstractmethod
    def __add__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __radd__(self: Self, other: Any) -> Any: ...
    def __sub__(self: Self, other: Any) -> Any: ...
    def __rsub__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __mul__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __rmul__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __truediv__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __rtruediv__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __pow__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __rpow__(self: Self, other: Any) -> Any: ...

    @abstractmethod
    def __neg__(self: Self) -> ComplexLike: ...
    @abstractmethod
    def __pos__(self: Self) -> ComplexLike: ...
    @abstractmethod
    def __abs__(self: Self) -> ComplexLike: ...

    if sys.version_info >= (3, 11):
        @abstractmethod
        def __complex__(self: Self) -> Complex: ...

    @abstractmethod
    def conjugate(self: Self) -> ComplexLike: ...


@runtime_checkable
class RealLike(ComplexLike, Protocol):

    @abstractmethod
    def __lt__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __le__(self: Self, other: Any) -> Any: ...

    @abstractmethod
    def __floordiv__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __rfloordiv__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __mod__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __rmod__(self: Self, other: Any) -> Any: ...

    @abstractmethod
    def __float__(self: Self) -> Float: ...


@runtime_checkable
class IntegralLike(RealLike, Protocol):

    @abstractmethod
    def __int__(self: Self) -> Int: ...
    def __index__(self: Self) -> Int: ...


@runtime_checkable
class ShapeLike(Protocol):

    def __len__(self: Self) -> Literal[2]: ...
    @abstractmethod
    def __getitem__(self: Self, key: Int) -> Int: ...
    def __iter__(self: Self) -> Iterator[Int]: ...
    def __reversed__(self: Self) -> Iterator[Int]: ...
    def __contains__(self: Self, value: Any) -> bool: ...

    @property
    def nrows(self: Self) -> Int: ...
    @property
    def ncols(self: Self) -> Int: ...
    @property
    def size(self: Self) -> Int: ...


@runtime_checkable
class MatrixLike(Protocol[T_co]):

    def __len__(self: Self) -> Int: ...
    @abstractmethod
    @overload
    def __getitem__(self: Self, key: Int) -> T_co: ...
    @abstractmethod
    @overload
    def __getitem__(self: Self, key: Slice) -> MatrixLike: ...
    @abstractmethod
    @overload
    def __getitem__(self: Self, key: tuple[Int, Int]) -> T_co: ...
    @abstractmethod
    @overload
    def __getitem__(self: Self, key: tuple[Int, Slice]) -> MatrixLike: ...
    @abstractmethod
    @overload
    def __getitem__(self: Self, key: tuple[Slice, Int]) -> MatrixLike: ...
    @abstractmethod
    @overload
    def __getitem__(self: Self, key: tuple[Slice, Slice]) -> MatrixLike: ...
    def __iter__(self: Self) -> Iterator[T_co]: ...
    def __reversed__(self: Self) -> Iterator[T_co]: ...
    def __contains__(self: Self, value: Any) -> bool: ...

    @property
    @abstractmethod
    def shape(self: Self) -> ShapeLike: ...
    @property
    def nrows(self: Self) -> Int: ...
    @property
    def ncols(self: Self) -> Int: ...
    @property
    def size(self: Self) -> Int: ...


@runtime_checkable
class ComplexMatrixLike(ComplexLike, MatrixLike[ComplexLikeT_co], Protocol[ComplexLikeT_co]):

    @abstractmethod
    def __matmul__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __rmatmul__(self: Self, other: Any) -> Any: ...

    @abstractmethod
    def complex(self: Self) -> ComplexMatrixLike[Complex]: ...


@runtime_checkable
class RealMatrixLike(RealLike, ComplexMatrixLike[RealLikeT_co], Protocol[RealLikeT_co]):

    @abstractmethod
    def __gt__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __ge__(self: Self, other: Any) -> Any: ...

    @abstractmethod
    def float(self: Self) -> RealMatrixLike[Float]: ...


@runtime_checkable
class IntegralMatrixLike(IntegralLike, RealMatrixLike[IntegralLikeT_co], Protocol[IntegralLikeT_co]):

    @abstractmethod
    def int(self: Self) -> IntegralMatrixLike[Int]: ...
