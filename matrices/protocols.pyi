import sys
from abc import abstractmethod
from collections.abc import Iterator
from typing import (Any, Literal, Protocol, SupportsIndex, TypeVar, overload,
                    runtime_checkable)

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
    def __abs__(self: Self) -> RealLike: ...

    if sys.version_info >= (3, 11):
        @abstractmethod
        def __complex__(self: Self) -> complex: ...

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
    def __float__(self: Self) -> float: ...


@runtime_checkable
class IntegralLike(RealLike, Protocol):

    @abstractmethod
    def __int__(self: Self) -> int: ...
    def __index__(self: Self) -> int: ...


@runtime_checkable
class ShapeLike(Protocol):

    def __eq__(self, other: Any) -> bool: ...

    def __len__(self: Self) -> Literal[2]: ...
    @abstractmethod
    def __getitem__(self: Self, key: SupportsIndex) -> int: ...
    def __iter__(self: Self) -> Iterator[int]: ...
    def __reversed__(self: Self) -> Iterator[int]: ...
    def __contains__(self: Self, value: Any) -> bool: ...

    @property
    def nrows(self: Self) -> int: ...
    @property
    def ncols(self: Self) -> int: ...
    @property
    def size(self: Self) -> int: ...

    def true_equals(self: Self, other: ShapeLike) -> bool: ...


@runtime_checkable
class MatrixLike(Protocol[T_co]):

    @abstractmethod
    def __eq__(self: Self, other: Any) -> Any: ...  # type: ignore[override]
    @abstractmethod
    def __ne__(self: Self, other: Any) -> Any: ...  # type: ignore[override]

    def __len__(self: Self) -> int: ...
    @abstractmethod
    @overload
    def __getitem__(self: Self, key: int) -> T_co: ...
    @abstractmethod
    @overload
    def __getitem__(self: Self, key: slice) -> MatrixLike: ...
    @abstractmethod
    @overload
    def __getitem__(self: Self, key: tuple[int, int]) -> T_co: ...
    @abstractmethod
    @overload
    def __getitem__(self: Self, key: tuple[int, slice]) -> MatrixLike: ...
    @abstractmethod
    @overload
    def __getitem__(self: Self, key: tuple[slice, int]) -> MatrixLike: ...
    @abstractmethod
    @overload
    def __getitem__(self: Self, key: tuple[slice, slice]) -> MatrixLike: ...
    def __iter__(self: Self) -> Iterator[T_co]: ...
    def __reversed__(self: Self) -> Iterator[T_co]: ...
    def __contains__(self: Self, value: Any) -> bool: ...
    @abstractmethod
    def __copy__(self: Self) -> MatrixLike[T_co]: ...

    @abstractmethod
    def __and__(self: Self, other: Any) -> Any: ...
    def __rand__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __or__(self: Self, other: Any) -> Any: ...
    def __ror__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __xor__(self: Self, other: Any) -> Any: ...
    def __rxor__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __invert__(self: Self) -> Any: ...

    @property
    @abstractmethod
    def shape(self: Self) -> ShapeLike: ...
    @property
    def nrows(self: Self) -> int: ...
    @property
    def ncols(self: Self) -> int: ...
    @property
    def size(self: Self) -> int: ...

    def true_equals(self: Self, other: MatrixLike) -> bool: ...
    def copy(self: Self) -> MatrixLike[T_co]: ...


@runtime_checkable
class ComplexMatrixLike(ComplexLike, MatrixLike[ComplexLikeT_co], Protocol[ComplexLikeT_co]):

    @abstractmethod
    def __matmul__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __rmatmul__(self: Self, other: Any) -> Any: ...

    @abstractmethod
    def complex(self: Self) -> ComplexMatrixLike[complex]: ...


@runtime_checkable
class RealMatrixLike(RealLike, ComplexMatrixLike[RealLikeT_co], Protocol[RealLikeT_co]):

    @abstractmethod
    def __gt__(self: Self, other: Any) -> Any: ...
    @abstractmethod
    def __ge__(self: Self, other: Any) -> Any: ...

    @abstractmethod
    def float(self: Self) -> RealMatrixLike[float]: ...


@runtime_checkable
class IntegralMatrixLike(IntegralLike, RealMatrixLike[IntegralLikeT_co], Protocol[IntegralLikeT_co]):

    @abstractmethod
    def int(self: Self) -> IntegralMatrixLike[int]: ...
