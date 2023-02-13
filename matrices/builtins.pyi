from collections.abc import Iterable, Sequence
from typing import Any, Literal, Optional, SupportsIndex, TypeVar, overload

from .abc import (ComplexMatrixLike, IntegralMatrixLike, MatrixLike,
                  RealMatrixLike)
from .rule import Rule

__all__ = [
    "Matrix",
    "ComplexMatrix",
    "RealMatrix",
    "IntegralMatrix",
]

T = TypeVar("T")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
P = TypeVar("P", bound=int)

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)
P_co = TypeVar("P_co", covariant=True, bound=int)

ComplexT_co = TypeVar("ComplexT_co", covariant=True, bound=complex)
RealT_co = TypeVar("RealT_co", covariant=True, bound=float)
IntegralT_co = TypeVar("IntegralT_co", covariant=True, bound=int)

MatrixT = TypeVar("MatrixT", bound=Matrix)


class Matrix(MatrixLike[T_co, M_co, N_co]):

    __slots__: tuple[Literal["_array"], Literal["_shape"]]

    @overload
    def __init__(self, array: MatrixLike[T_co, M_co, N_co]) -> None: ...
    @overload
    def __init__(self, array: Optional[Iterable[T_co]] = None, shape: Optional[tuple[Optional[M_co], Optional[N_co]]] = None) -> None: ...
    def __repr__(self) -> str: ...
    @overload
    def __getitem__(self, key: SupportsIndex) -> T_co: ...
    @overload
    def __getitem__(self, key: slice) -> MatrixLike[T_co, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, SupportsIndex]) -> T_co: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, slice]) -> MatrixLike[T_co, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, SupportsIndex]) -> MatrixLike[T_co, Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> MatrixLike[T_co, Any, Any]: ...
    def __deepcopy__(self: MatrixT, memo: Optional[dict[int, Any]] = None) -> MatrixT: ...
    def __copy__(self: MatrixT) -> MatrixT: ...

    @classmethod
    def from_raw_parts(cls: type[MatrixT], array: list[T_co], shape: tuple[M_co, N_co]) -> MatrixT: ...
    @classmethod
    def from_nested(cls: type[MatrixT], nested: Iterable[Iterable[T_co]]) -> MatrixT: ...

    @property
    def array(self) -> Sequence[T_co]: ...
    @property
    def shape(self) -> tuple[M_co, N_co]: ...

    def equal(self, other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    def not_equal(self, other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    def logical_and(self, other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    def logical_or(self, other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    def logical_not(self) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    def transpose(self) -> MatrixLike[T_co, N_co, M_co]: ...
    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T_co, M_co, N_co]: ...
    def reverse(self) -> MatrixLike[T_co, M_co, N_co]: ...


class ComplexMatrix(ComplexMatrixLike[ComplexT_co, M_co, N_co], Matrix[ComplexT_co, M_co, N_co]):

    __slots__: tuple[()]

    @overload
    def __getitem__(self, key: SupportsIndex) -> ComplexT_co: ...
    @overload
    def __getitem__(self, key: slice) -> ComplexMatrixLike[ComplexT_co, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, SupportsIndex]) -> ComplexT_co: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, slice]) -> ComplexMatrixLike[ComplexT_co, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, SupportsIndex]) -> ComplexMatrixLike[ComplexT_co, Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> ComplexMatrixLike[ComplexT_co, Any, Any]: ...
    @overload
    def __add__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __add__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __add__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __sub__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __sub__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __sub__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __mul__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __mul__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __mul__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __matmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, N_co, P_co]) -> ComplexMatrixLike[complex, M_co, P_co]: ...
    @overload
    def __matmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, N_co, P_co]) -> ComplexMatrixLike[complex, M_co, P_co]: ...
    @overload
    def __matmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, N_co, P_co]) -> ComplexMatrixLike[complex, M_co, P_co]: ...
    @overload
    def __truediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __truediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __truediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __radd__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __radd__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __radd__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __rsub__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __rsub__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __rsub__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __rmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __rmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __rmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __rmatmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, P_co, M_co]) -> ComplexMatrixLike[complex, P_co, N_co]: ...
    @overload
    def __rmatmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, P_co, M_co]) -> ComplexMatrixLike[complex, P_co, N_co]: ...
    @overload
    def __rmatmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, P_co, M_co]) -> ComplexMatrixLike[complex, P_co, N_co]: ...
    @overload
    def __rtruediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __rtruediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    def __rtruediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    def __neg__(self: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    def __pos__(self: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    def __abs__(self: ComplexMatrixLike[complex, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    def conjugate(self: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    def transpose(self) -> ComplexMatrixLike[ComplexT_co, N_co, M_co]: ...
    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[ComplexT_co, M_co, N_co]: ...
    def reverse(self) -> ComplexMatrixLike[ComplexT_co, M_co, N_co]: ...


class RealMatrix(RealMatrixLike[RealT_co, M_co, N_co], Matrix[RealT_co, M_co, N_co]):

    __slots__: tuple[()]

    @overload
    def __getitem__(self, key: SupportsIndex) -> RealT_co: ...
    @overload
    def __getitem__(self, key: slice) -> RealMatrixLike[RealT_co, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, SupportsIndex]) -> RealT_co: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, slice]) -> RealMatrixLike[RealT_co, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, SupportsIndex]) -> RealMatrixLike[RealT_co, Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> RealMatrixLike[RealT_co, Any, Any]: ...
    @overload
    def __add__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __add__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __sub__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __sub__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __mul__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __mul__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __matmul__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, N_co, P_co]) -> RealMatrixLike[float, M_co, P_co]: ...
    @overload
    def __matmul__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, N_co, P_co]) -> RealMatrixLike[float, M_co, P_co]: ...
    @overload
    def __truediv__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __truediv__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __floordiv__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __floordiv__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __mod__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __mod__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __divmod__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> MatrixLike[tuple[float, float], M_co, N_co]: ...
    @overload
    def __divmod__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> MatrixLike[tuple[float, float], M_co, N_co]: ...
    @overload
    def __radd__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __radd__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __rsub__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __rsub__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __rmul__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __rmul__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __rmatmul__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, P_co, M_co]) -> RealMatrixLike[float, P_co, N_co]: ...
    @overload
    def __rmatmul__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, P_co, M_co]) -> RealMatrixLike[float, P_co, N_co]: ...
    @overload
    def __rtruediv__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __rtruediv__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __rfloordiv__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __rfloordiv__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __rmod__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __rmod__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    def __rdivmod__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> MatrixLike[tuple[float, float], M_co, N_co]: ...
    @overload
    def __rdivmod__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> MatrixLike[tuple[float, float], M_co, N_co]: ...
    def __neg__(self: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    def __pos__(self: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    def __abs__(self: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    @overload
    def lesser(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    def lesser(self, other: RealMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    def lesser_equal(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    def lesser_equal(self, other: RealMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    def greater(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    def greater(self, other: RealMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    def greater_equal(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    def greater_equal(self, other: RealMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    def conjugate(self: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    def transpose(self) -> RealMatrixLike[RealT_co, N_co, M_co]: ...
    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[RealT_co, M_co, N_co]: ...
    def reverse(self) -> RealMatrixLike[RealT_co, M_co, N_co]: ...


class IntegralMatrix(IntegralMatrixLike[IntegralT_co, M_co, N_co], Matrix[IntegralT_co, M_co, N_co]):

    __slots__: tuple[()]

    @overload
    def __getitem__(self, key: SupportsIndex) -> IntegralT_co: ...
    @overload
    def __getitem__(self, key: slice) -> IntegralMatrixLike[IntegralT_co, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, SupportsIndex]) -> IntegralT_co: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, slice]) -> IntegralMatrixLike[IntegralT_co, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, SupportsIndex]) -> IntegralMatrixLike[IntegralT_co, Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> IntegralMatrixLike[IntegralT_co, Any, Any]: ...
    def __add__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __sub__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __mul__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __matmul__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, N_co, P_co]) -> IntegralMatrixLike[int, M_co, P_co]: ...
    def __truediv__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    def __floordiv__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __mod__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __divmod__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> MatrixLike[tuple[int, int], M_co, N_co]: ...
    def __lshift__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __rshift__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __and__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __xor__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __or__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __radd__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __rsub__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __rmul__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __rmatmul__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, P_co, M_co]) -> IntegralMatrixLike[int, P_co, N_co]: ...
    def __rtruediv__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    def __rfloordiv__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __rmod__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __rdivmod__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> MatrixLike[tuple[int, int], M_co, N_co]: ...
    def __rlshift__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __rrshift__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __rand__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __rxor__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __ror__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __neg__(self: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __pos__(self: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __abs__(self: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def __invert__(self: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...

    def lesser(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    def lesser_equal(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    def greater(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    def greater_equal(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    def conjugate(self: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    def transpose(self) -> IntegralMatrixLike[IntegralT_co, N_co, M_co]: ...
    def flip(self, *, by: Rule = Rule.ROW) -> IntegralMatrixLike[IntegralT_co, M_co, N_co]: ...
    def reverse(self) -> IntegralMatrixLike[IntegralT_co, M_co, N_co]: ...
