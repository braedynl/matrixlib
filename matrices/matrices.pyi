from collections.abc import Iterable, Sequence
from typing import (Any, Iterator, Literal, Optional, SupportsIndex, TypeVar,
                    Union, overload)

from .protocols import (ComplexLike, ComplexMatrixLike, IntegralLike,
                        IntegralMatrixLike, MatrixLike, RealLike,
                        RealMatrixLike)
from .rule import Rule
from .shape import Shape

__all__ = [
    "Matrix",
    "ComplexMatrix",
    "RealMatrix",
    "IntegralMatrix",
]

T = TypeVar("T")
ComplexLikeT = TypeVar("ComplexLikeT", bound=ComplexLike)
RealLikeT = TypeVar("RealLikeT", bound=RealLike)
IntegralLikeT = TypeVar("IntegralLikeT", bound=IntegralLike)

Self = TypeVar("Self")


class Matrix(Sequence[T]):

    __slots__: tuple[Literal["data"], Literal["shape"]]

    def __init__(self: Self, values: Iterable[T], *, nrows: int, ncols: int) -> None: ...
    def __repr__(self: Self) -> str: ...
    def __str__(self: Self) -> str: ...
    def __eq__(self: Self, other: Union[MatrixLike[Any], Any]) -> bool: ...
    def __ne__(self: Self, other: Union[MatrixLike[Any], Any]) -> bool: ...
    def __len__(self: Self) -> int: ...
    @overload
    def __getitem__(self: Self, key: SupportsIndex) -> T: ...
    @overload
    def __getitem__(self: Self, key: slice) -> Self: ...
    @overload
    def __getitem__(self: Self, key: tuple[SupportsIndex, SupportsIndex]) -> T: ...
    @overload
    def __getitem__(self: Self, key: tuple[SupportsIndex, slice]) -> Self: ...
    @overload
    def __getitem__(self: Self, key: tuple[slice, SupportsIndex]) -> Self: ...
    @overload
    def __getitem__(self: Self, key: tuple[slice, slice]) -> Self: ...
    @overload
    def __setitem__(self: Self, key: SupportsIndex, value: T) -> None: ...
    @overload
    def __setitem__(self: Self, key: slice, other: Union[MatrixLike[T], T]) -> None: ...
    @overload
    def __setitem__(self: Self, key: tuple[SupportsIndex, SupportsIndex], value: T) -> None: ...
    @overload
    def __setitem__(self: Self, key: tuple[SupportsIndex, slice], other: Union[MatrixLike[T], T]) -> None: ...
    @overload
    def __setitem__(self: Self, key: tuple[slice, SupportsIndex], other: Union[MatrixLike[T], T]) -> None: ...
    @overload
    def __setitem__(self: Self, key: tuple[slice, slice], other: Union[MatrixLike[T], T]) -> None: ...
    def __iter__(self: Self) -> Iterator[T]: ...
    def __reversed__(self: Self) -> Iterator[T]: ...
    def __contains__(self: Self, value: Any) -> bool: ...
    def __deepcopy__(self: Self, memo: Optional[dict[int, Any]] = None) -> Self: ...
    def __copy__(self: Self) -> Self: ...
    def __and__(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...
    def __rand__(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...
    def __or__(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...
    def __ror__(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...
    def __xor__(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...
    def __rxor__(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...
    def __invert__(self: Self) -> IntegralMatrix[bool]: ...

    @classmethod
    def wrap(cls: type[Self], data: list[T], shape: Shape) -> Self: ...
    @classmethod
    def fill(cls: type[Self], value: T, *, nrows: int, ncols: int) -> Self: ...
    @classmethod
    def infer(cls: type[Self], other: Iterable[Iterable[T]]) -> Self: ...

    @property
    def shape(self: Self) -> Shape: ...
    @property
    def nrows(self: Self) -> int: ...
    @property
    def ncols(self: Self) -> int: ...
    @property
    def size(self: Self) -> int: ...

    def index(self: Self, value: T, start: int = 0, stop: Optional[int] = None) -> int: ...
    def count(self: Self, value: T) -> int: ...
    def reverse(self: Self) -> Self: ...
    def eq(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...
    def ne(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...
    def reshape(self: Self, nrows: int, ncols: int) -> Self: ...
    def slices(self: Self, *, by: Rule = Rule.ROW) -> Iterator[Self]: ...
    def mask(self: Self, selector: MatrixLike[Any], null: T) -> Self: ...
    def replace(self: Self, old: T, new: T, *, times: Optional[int] = None) -> Self: ...
    def swap(self: Self, key1: SupportsIndex, key2: SupportsIndex, *, by: Rule = Rule.ROW) -> Self: ...
    def flip(self: Self, *, by: Rule = Rule.ROW) -> Self: ...
    def flatten(self: Self, *, by: Rule = Rule.ROW) -> Self: ...
    def transpose(self: Self) -> Self: ...
    def stack(self: Self, other: MatrixLike[T], *, by: Rule = Rule.ROW) -> Self: ...
    def pull(self: Self, key: SupportsIndex = -1, *, by: Rule = Rule.ROW) -> Self: ...
    def copy(self: Self, *, deep: bool = False) -> Self: ...


class ComplexMatrix(Matrix[ComplexLikeT]):

    __slots__: tuple[()]

    @overload
    def __add__(self: ComplexMatrix[complex], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __add__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __add__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __radd__(self: ComplexMatrix[complex], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __radd__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __radd__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __sub__(self: ComplexMatrix[complex], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __sub__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __sub__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rsub__(self: ComplexMatrix[complex], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __rsub__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __rsub__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __mul__(self: ComplexMatrix[complex], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __mul__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __mul__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rmul__(self: ComplexMatrix[complex], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __rmul__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __rmul__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __truediv__(self: ComplexMatrix[complex], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __truediv__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __truediv__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rtruediv__(self: ComplexMatrix[complex], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __rtruediv__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __rtruediv__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __pow__(self: ComplexMatrix[complex], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __pow__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __pow__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rpow__(self: ComplexMatrix[complex], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __rpow__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __rpow__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __matmul__(self: ComplexMatrix[complex], other: ComplexMatrixLike[complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __matmul__(self: Self, other: ComplexMatrixLike[Any]) -> ComplexMatrix[Any]: ...
    @overload
    def __matmul__(self: Self, other: MatrixLike[Any]) -> Matrix[Any]: ...

    @overload
    def __neg__(self: ComplexMatrix[complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __neg__(self: Self) -> ComplexMatrix[Any]: ...
    @overload
    def __pos__(self: ComplexMatrix[complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __pos__(self: Self) -> ComplexMatrix[Any]: ...
    @overload
    def __abs__(self: ComplexMatrix[complex]) -> RealMatrix[float]: ...
    @overload
    def __abs__(self: Self) -> RealMatrix[Any]: ...

    def __complex__(self: Self) -> complex: ...

    @overload
    def conjugate(self: ComplexMatrix[complex]) -> ComplexMatrix[complex]: ...
    @overload
    def conjugate(self: Self) -> ComplexMatrix[Any]: ...

    def complex(self: Self) -> ComplexMatrix[complex]: ...


class RealMatrix(Matrix[RealLikeT]):

    __slots__: tuple[()]

    def __lt__(self: Self, other: Union[MatrixLike[Any], Any]) -> bool: ...
    def __le__(self: Self, other: Union[MatrixLike[Any], Any]) -> bool: ...
    def __gt__(self: Self, other: Union[MatrixLike[Any], Any]) -> bool: ...
    def __ge__(self: Self, other: Union[MatrixLike[Any], Any]) -> bool: ...

    @overload
    def __add__(self: RealMatrix[float], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __add__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __add__(self: RealMatrix[float], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __add__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __add__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __radd__(self: RealMatrix[float], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: RealMatrix[float], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __radd__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __radd__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __sub__(self: RealMatrix[float], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: RealMatrix[float], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __sub__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __sub__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rsub__(self: RealMatrix[float], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: RealMatrix[float], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __rsub__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __rsub__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __mul__(self: RealMatrix[float], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: RealMatrix[float], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __mul__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __mul__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rmul__(self: RealMatrix[float], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: RealMatrix[float], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __rmul__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __rmul__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __truediv__(self: RealMatrix[float], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: RealMatrix[float], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __truediv__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __truediv__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rtruediv__(self: RealMatrix[float], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: RealMatrix[float], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __rtruediv__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __rtruediv__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __pow__(self: RealMatrix[float], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __pow__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __pow__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rpow__(self: RealMatrix[float], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __rpow__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __rpow__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __matmul__(self: RealMatrix[float], other: RealMatrixLike[float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: Self, other: RealMatrixLike[Any]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: RealMatrix[float], other: ComplexMatrixLike[complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __matmul__(self: Self, other: ComplexMatrixLike[Any]) -> ComplexMatrix[Any]: ...
    @overload
    def __matmul__(self: Self, other: MatrixLike[Any]) -> Matrix[Any]: ...
    @overload
    def __floordiv__(self: RealMatrix[float], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...
    @overload
    def __floordiv__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...
    @overload
    def __floordiv__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rfloordiv__(self: RealMatrix[float], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...
    @overload
    def __rfloordiv__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...
    @overload
    def __rfloordiv__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __mod__(self: RealMatrix[float], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...
    @overload
    def __mod__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...
    @overload
    def __mod__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rmod__(self: RealMatrix[float], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...
    @overload
    def __rmod__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...
    @overload
    def __rmod__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...

    @overload
    def __neg__(self: RealMatrix[float]) -> RealMatrix[float]: ...
    @overload
    def __neg__(self: Self) -> RealMatrix[Any]: ...
    @overload
    def __pos__(self: RealMatrix[float]) -> RealMatrix[float]: ...
    @overload
    def __pos__(self: Self) -> RealMatrix[Any]: ...
    @overload
    def __abs__(self: RealMatrix[float]) -> RealMatrix[float]: ...
    @overload
    def __abs__(self: Self) -> RealMatrix[Any]: ...

    def __complex__(self: Self) -> complex: ...
    def __float__(self: Self) -> float: ...

    def lt(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...
    def le(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...
    def gt(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...
    def ge(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...

    @overload
    def conjugate(self: RealMatrix[float]) -> RealMatrix[float]: ...
    @overload
    def conjugate(self: Self) -> RealMatrix[Any]: ...

    def complex(self: Self) -> ComplexMatrix[complex]: ...
    def float(self: Self) -> RealMatrix[float]: ...


class IntegralMatrix(Matrix[IntegralLikeT]):

    __slots__: tuple[()]

    def __lt__(self: Self, other: Union[MatrixLike[Any], Any]) -> bool: ...
    def __le__(self: Self, other: Union[MatrixLike[Any], Any]) -> bool: ...
    def __gt__(self: Self, other: Union[MatrixLike[Any], Any]) -> bool: ...
    def __ge__(self: Self, other: Union[MatrixLike[Any], Any]) -> bool: ...

    @overload
    def __add__(self: IntegralMatrix[int], other: Union[IntegralMatrixLike[int], int]) -> IntegralMatrix[int]: ...  # type: ignore[misc]
    @overload
    def __add__(self: Self, other: Union[IntegralMatrixLike[Any], IntegralLike]) -> IntegralMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __add__(self: IntegralMatrix[int], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __add__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __add__(self: IntegralMatrix[int], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __add__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __add__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __radd__(self: IntegralMatrix[int], other: Union[IntegralMatrixLike[int], int]) -> IntegralMatrix[int]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: Self, other: Union[IntegralMatrixLike[Any], IntegralLike]) -> IntegralMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: IntegralMatrix[int], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: IntegralMatrix[int], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __radd__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __radd__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __sub__(self: IntegralMatrix[int], other: Union[IntegralMatrixLike[int], int]) -> IntegralMatrix[int]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: Self, other: Union[IntegralMatrixLike[Any], IntegralLike]) -> IntegralMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: IntegralMatrix[int], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: IntegralMatrix[int], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __sub__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __sub__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rsub__(self: IntegralMatrix[int], other: Union[IntegralMatrixLike[int], int]) -> IntegralMatrix[int]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: Self, other: Union[IntegralMatrixLike[Any], IntegralLike]) -> IntegralMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: IntegralMatrix[int], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: IntegralMatrix[int], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __rsub__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __rsub__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __mul__(self: IntegralMatrix[int], other: Union[IntegralMatrixLike[int], int]) -> IntegralMatrix[int]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: Self, other: Union[IntegralMatrixLike[Any], IntegralLike]) -> IntegralMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: IntegralMatrix[int], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: IntegralMatrix[int], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __mul__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __mul__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rmul__(self: IntegralMatrix[int], other: Union[IntegralMatrixLike[int], int]) -> IntegralMatrix[int]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: Self, other: Union[IntegralMatrixLike[Any], IntegralLike]) -> IntegralMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: IntegralMatrix[int], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: IntegralMatrix[int], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __rmul__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __rmul__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __truediv__(self: IntegralMatrix[int], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: IntegralMatrix[int], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __truediv__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __truediv__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rtruediv__(self: IntegralMatrix[int], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: IntegralMatrix[int], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __rtruediv__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __rtruediv__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __pow__(self: IntegralMatrix[int], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __pow__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __pow__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rpow__(self: IntegralMatrix[int], other: Union[ComplexMatrixLike[complex], complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __rpow__(self: Self, other: Union[ComplexMatrixLike[Any], ComplexLike]) -> ComplexMatrix[Any]: ...
    @overload
    def __rpow__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __matmul__(self: IntegralMatrix[int], other: IntegralMatrixLike[int]) -> IntegralMatrix[int]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: Self, other: IntegralMatrixLike[Any]) -> IntegralMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: IntegralMatrix[int], other: RealMatrixLike[float]) -> RealMatrix[float]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: Self, other: RealMatrixLike[Any]) -> RealMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: IntegralMatrix[int], other: ComplexMatrixLike[complex]) -> ComplexMatrix[complex]: ...
    @overload
    def __matmul__(self: Self, other: ComplexMatrixLike[Any]) -> ComplexMatrix[Any]: ...
    @overload
    def __matmul__(self: Self, other: MatrixLike[Any]) -> Matrix[Any]: ...
    @overload
    def __floordiv__(self: IntegralMatrix[int], other: Union[IntegralMatrixLike[int], int]) -> IntegralMatrix[int]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: Self, other: Union[IntegralMatrixLike[Any], IntegralLike]) -> IntegralMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: IntegralMatrix[int], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...
    @overload
    def __floordiv__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...
    @overload
    def __floordiv__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rfloordiv__(self: IntegralMatrix[int], other: Union[IntegralMatrixLike[int], int]) -> IntegralMatrix[int]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: Self, other: Union[IntegralMatrixLike[Any], IntegralLike]) -> IntegralMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: IntegralMatrix[int], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...
    @overload
    def __rfloordiv__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...
    @overload
    def __rfloordiv__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __mod__(self: IntegralMatrix[int], other: Union[IntegralMatrixLike[int], int]) -> IntegralMatrix[int]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: Self, other: Union[IntegralMatrixLike[Any], IntegralLike]) -> IntegralMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: IntegralMatrix[int], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...
    @overload
    def __mod__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...
    @overload
    def __mod__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...
    @overload
    def __rmod__(self: IntegralMatrix[int], other: Union[IntegralMatrixLike[int], int]) -> IntegralMatrix[int]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: Self, other: Union[IntegralMatrixLike[Any], IntegralLike]) -> IntegralMatrix[Any]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: IntegralMatrix[int], other: Union[RealMatrixLike[float], float]) -> RealMatrix[float]: ...
    @overload
    def __rmod__(self: Self, other: Union[RealMatrixLike[Any], RealLike]) -> RealMatrix[Any]: ...
    @overload
    def __rmod__(self: Self, other: Union[MatrixLike[Any], Any]) -> Matrix[Any]: ...

    @overload
    def __neg__(self: IntegralMatrix[int]) -> IntegralMatrix[int]: ...
    @overload
    def __neg__(self: Self) -> IntegralMatrix[Any]: ...
    @overload
    def __pos__(self: IntegralMatrix[int]) -> IntegralMatrix[int]: ...
    @overload
    def __pos__(self: Self) -> IntegralMatrix[Any]: ...
    @overload
    def __abs__(self: IntegralMatrix[int]) -> IntegralMatrix[int]: ...
    @overload
    def __abs__(self: Self) -> IntegralMatrix[Any]: ...

    def __complex__(self: Self) -> complex: ...
    def __float__(self: Self) -> float: ...
    def __int__(self: Self) -> int: ...
    def __index__(self: Self) -> int: ...

    def lt(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...
    def le(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...
    def gt(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...
    def ge(self: Self, other: Union[MatrixLike[Any], Any]) -> IntegralMatrix[bool]: ...

    @overload
    def conjugate(self: IntegralMatrix[int]) -> IntegralMatrix[int]: ...
    @overload
    def conjugate(self: Self) -> IntegralMatrix[Any]: ...

    def complex(self: Self) -> ComplexMatrix[complex]: ...
    def float(self: Self) -> RealMatrix[float]: ...
    def int(self: Self) -> IntegralMatrix[int]: ...
