from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, Literal, TypeVar, overload

from ..abc import (ComplexMatrixLike, DatetimeMatrixLike, IntegralMatrixLike,
                   MatrixLike, RealMatrixLike, StringMatrixLike,
                   TimedeltaMatrixLike)
from ..rule import Rule
from .abc import MatrixViewLike

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)
P_co = TypeVar("P_co", covariant=True, bound=int)


class MatrixView(MatrixViewLike[T_co, M_co, N_co]):

    __slots__ = ("_target",)

    def __init__(self, target: MatrixLike[T_co, M_co, N_co]) -> None:
        self._target: MatrixLike[T_co, M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> T_co: ...
    @overload
    def __getitem__(self, key: slice) -> MatrixLike[T_co, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> MatrixLike[T_co, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> MatrixLike[T_co, Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> MatrixLike[T_co, Any, Any]: ...

    def __getitem__(self, key):
        return self.target.__getitem__(key)

    def __hash__(self) -> int:
        return hash(self.target)

    @property
    def array(self) -> Sequence[T_co]:
        return self.target.array

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self.target.shape

    @property
    def target(self) -> MatrixLike[T_co, M_co, N_co]:
        return self._target

    @overload
    def equal(self, other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def equal(self, other: Any) -> IntegralMatrixLike[M_co, N_co]: ...

    def equal(self, other):
        return self.target.equal(other)

    @overload
    def not_equal(self, other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def not_equal(self, other: Any) -> IntegralMatrixLike[M_co, N_co]: ...

    def not_equal(self, other):
        return self.target.not_equal(other)

    def transpose(self) -> MatrixLike[T_co, N_co, M_co]:
        return self.target.transpose()

    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T_co, M_co, N_co]:
        return self.target.flip(by=by)

    def reverse(self) -> MatrixLike[T_co, M_co, N_co]:
        return self.target.reverse()


class StringMatrixView(StringMatrixLike[M_co, N_co], MatrixView[str, M_co, N_co]):

    __slots__ = ()

    def __init__(self, target: StringMatrixLike[M_co, N_co]) -> None:
        self._target: StringMatrixLike[M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> str: ...
    @overload
    def __getitem__(self, key: slice) -> StringMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> str: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> StringMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> StringMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> StringMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        return self.target.__getitem__(key)

    @property
    def target(self) -> StringMatrixLike[M_co, N_co]:
        return self._target

    @overload
    def __add__(self, other: StringMatrixLike[M_co, N_co]) -> StringMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self, other: str) -> StringMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        return self.target.__add__(other)

    @overload
    def __mul__(self, other: IntegralMatrixLike[M_co, N_co]) -> StringMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: int) -> StringMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        return self.target.__mul__(other)

    def __radd__(self, other: str) -> StringMatrixLike[M_co, N_co]:
        return self.target.__radd__(other)

    @overload
    def lesser(self, other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self, other: str) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        return self.target.lesser(other)

    @overload
    def lesser_equal(self, other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self, other: str) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        return self.target.lesser_equal(other)

    @overload
    def greater(self, other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self, other: str) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        return self.target.greater(other)

    @overload
    def greater_equal(self, other: StringMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self, other: str) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        return self.target.greater_equal(other)


class ComplexMatrixView(ComplexMatrixLike[M_co, N_co], MatrixView[complex, M_co, N_co]):

    __slots__ = ()

    def __init__(self, target: ComplexMatrixLike[M_co, N_co]) -> None:
        self._target: ComplexMatrixLike[M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> complex: ...
    @overload
    def __getitem__(self, key: slice) -> ComplexMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> complex: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> ComplexMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> ComplexMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> ComplexMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        return self.target.__getitem__(key)

    @overload
    def __add__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self, other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self, other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self, other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        return self.target.__add__(other)

    @overload
    def __sub__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        return self.target.__sub__(other)

    @overload
    def __mul__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        return self.target.__mul__(other)

    @overload
    def __matmul__(self, other: IntegralMatrixLike[N_co, P_co]) -> ComplexMatrixLike[M_co, P_co]: ...
    @overload
    def __matmul__(self, other: RealMatrixLike[N_co, P_co]) -> ComplexMatrixLike[M_co, P_co]: ...
    @overload
    def __matmul__(self, other: ComplexMatrixLike[N_co, P_co]) -> ComplexMatrixLike[M_co, P_co]: ...

    def __matmul__(self, other):
        return self.target.__matmul__(other)

    @overload
    def __truediv__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: ComplexMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __truediv__(self, other):
        return self.target.__truediv__(other)

    @overload
    def __radd__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self, other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self, other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __radd__(self, other):
        return self.target.__radd__(other)

    @overload
    def __rsub__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self, other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self, other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rsub__(self, other):
        return self.target.__rsub__(other)

    @overload
    def __rmul__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self, other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self, other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rmul__(self, other):
        return self.__rmul__(other)

    @overload
    def __rmatmul__(self, other: IntegralMatrixLike[P_co, M_co]) -> ComplexMatrixLike[P_co, N_co]: ...
    @overload
    def __rmatmul__(self, other: RealMatrixLike[P_co, M_co]) -> ComplexMatrixLike[P_co, N_co]: ...

    def __rmatmul__(self, other):
        return self.target.__rmatmul__(other)

    @overload
    def __rtruediv__(self, other: IntegralMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self, other: int) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self, other: RealMatrixLike[M_co, N_co]) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self, other: float) -> ComplexMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rtruediv__(self, other):
        return self.target.__rtruediv__(other)

    def __neg__(self) -> ComplexMatrixLike[M_co, N_co]:
        return self.target.__neg__()

    def __abs__(self) -> RealMatrixLike[M_co, N_co]:
        return self.target.__abs__()

    @property
    def target(self) -> ComplexMatrixLike[M_co, N_co]:
        return self._target

    def transpose(self) -> ComplexMatrixLike[N_co, M_co]:
        return self.target.transpose()

    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[M_co, N_co]:
        return self.target.flip(by=by)

    def reverse(self) -> ComplexMatrixLike[M_co, N_co]:
        return self.target.reverse()

    def conjugate(self) -> ComplexMatrixLike[M_co, N_co]:
        return self.target.conjugate()


class RealMatrixView(RealMatrixLike[M_co, N_co], MatrixView[float, M_co, N_co]):

    __slots__ = ()

    def __init__(self, target: RealMatrixLike[M_co, N_co]) -> None:
        self._target: RealMatrixLike[M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> float: ...
    @overload
    def __getitem__(self, key: slice) -> RealMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> float: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> RealMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> RealMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> RealMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        return self.target.__getitem__(key)

    @overload
    def __add__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self, other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self, other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        return self.target.__add__(other)

    @overload
    def __sub__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        return self.target.__sub__(other)

    @overload
    def __mul__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        return self.target.__mul__(other)

    @overload
    def __matmul__(self, other: IntegralMatrixLike[N_co, P_co]) -> RealMatrixLike[M_co, P_co]: ...
    @overload
    def __matmul__(self, other: RealMatrixLike[N_co, P_co]) -> RealMatrixLike[M_co, P_co]: ...

    def __matmul__(self, other):
        return self.target.__matmul__(other)

    @overload
    def __truediv__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __truediv__(self, other):
        return self.target.__truediv__(other)

    @overload
    def __floordiv__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self, other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self, other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __floordiv__(self, other):
        return self.target.__floordiv__(other)

    @overload
    def __mod__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self, other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self, other: RealMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __mod__(self, other):
        return self.target.__mod__(other)

    @overload
    def __divmod__(self, other: IntegralMatrixLike[M_co, N_co]) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]: ...
    @overload
    def __divmod__(self, other: int) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]: ...
    @overload
    def __divmod__(self, other: RealMatrixLike[M_co, N_co]) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]: ...
    @overload
    def __divmod__(self, other: float) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]: ...

    def __divmod__(self, other):
        return self.target.__divmod__(other)

    @overload
    def __radd__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self, other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __radd__(self, other):
        return self.target.__radd__(other)

    @overload
    def __rsub__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self, other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rsub__(self, other):
        return self.target.__rsub__(other)

    @overload
    def __rmul__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self, other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rmul__(self, other):
        return self.target.__rmul__(other)

    def __rmatmul__(self, other: IntegralMatrixLike[P_co, M_co]) -> RealMatrixLike[P_co, N_co]:
        return self.target.__rmatmul__(other)

    @overload
    def __rtruediv__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self, other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rtruediv__(self, other):
        return self.target.__rtruediv__(other)

    @overload
    def __rfloordiv__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rfloordiv__(self, other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rfloordiv__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __rfloordiv__(self, other):
        return self.target.__rfloordiv__(other)

    @overload
    def __rmod__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmod__(self, other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmod__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __rmod__(self, other):
        return self.target.__rmod__(other)

    @overload
    def __rdivmod__(self, other: IntegralMatrixLike[M_co, N_co]) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]: ...
    @overload
    def __rdivmod__(self, other: int) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]: ...
    @overload
    def __rdivmod__(self, other: float) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]: ...

    def __rdivmod__(self, other):
        return self.target.__rdivmod__(other)

    def __neg__(self) -> RealMatrixLike[M_co, N_co]:
        return self.target.__neg__()

    def __abs__(self) -> RealMatrixLike[M_co, N_co]:
        return self.target.__abs__()

    @property
    def target(self) -> RealMatrixLike[M_co, N_co]:
        return self._target

    def transpose(self) -> RealMatrixLike[N_co, M_co]:
        return self.target.transpose()

    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[M_co, N_co]:
        return self.target.flip(by=by)

    def reverse(self) -> RealMatrixLike[M_co, N_co]:
        return self.target.reverse()

    @overload
    def lesser(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self, other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self, other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        return self.target.lesser(other)

    @overload
    def lesser_equal(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self, other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self, other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        return self.target.lesser_equal(other)

    @overload
    def greater(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self, other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self, other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        return self.target.greater(other)

    @overload
    def greater_equal(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self, other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self, other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        return self.target.greater_equal(other)


class IntegralMatrixView(IntegralMatrixLike[M_co, N_co], MatrixView[int, M_co, N_co]):

    __slots__ = ()

    def __init__(self, target: IntegralMatrixLike[M_co, N_co]) -> None:
        self._target: IntegralMatrixLike[M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> int: ...
    @overload
    def __getitem__(self, key: slice) -> IntegralMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> int: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> IntegralMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> IntegralMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> IntegralMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        return self.target.__getitem__(key)

    @overload
    def __add__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        return self.target.__add__(other)

    @overload
    def __sub__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        return self.target.__sub__(other)

    @overload
    def __mul__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        return self.target.__mul__(other)

    def __matmul__(self, other: IntegralMatrixLike[N_co, P_co]) -> IntegralMatrixLike[M_co, P_co]:
        return self.target.__matmul__(other)

    @overload
    def __truediv__(self, other: IntegralMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __truediv__(self, other):
        return self.target.__truediv__(other)

    @overload
    def __floordiv__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __floordiv__(self, other):
        return self.target.__floordiv__(other)

    @overload
    def __mod__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __mod__(self, other):
        return self.target.__mod__(other)

    @overload
    def __divmod__(self, other: IntegralMatrixLike[M_co, N_co]) -> tuple[IntegralMatrixLike[M_co, N_co], IntegralMatrixLike[M_co, N_co]]: ...
    @overload
    def __divmod__(self, other: int) -> tuple[IntegralMatrixLike[M_co, N_co], IntegralMatrixLike[M_co, N_co]]: ...
    @overload
    def __divmod__(self, other: float) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]: ...

    def __divmod__(self, other):
        return self.target.__divmod__(other)

    @overload
    def __lshift__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __lshift__(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...

    def __lshift__(self, other):
        return self.target.__lshift__(other)

    @overload
    def __rshift__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __rshift__(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...

    def __rshift__(self, other):
        return self.target.__rshift__(other)

    @overload
    def __and__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __and__(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...

    def __and__(self, other):
        return self.target.__and__(other)

    @overload
    def __xor__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __xor__(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...

    def __xor__(self, other):
        return self.target.__xor__(other)

    @overload
    def __or__(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __or__(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...

    def __or__(self, other):
        return self.target.__or__(other)

    @overload
    def __radd__(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __radd__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __radd__(self, other):
        return self.target.__radd__(other)

    @overload
    def __rsub__(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rsub__(self, other):
        return self.target.__rsub__(other)

    @overload
    def __rmul__(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rmul__(self, other):
        return self.target.__rmul__(other)

    @overload
    def __rtruediv__(self, other: int) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __rtruediv__(self, other: complex) -> ComplexMatrixLike[M_co, N_co]: ...

    def __rtruediv__(self, other):
        return self.target.__rtruediv__(other)

    @overload
    def __rfloordiv__(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __rfloordiv__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __rfloordiv__(self, other):
        return self.target.__rfloordiv__(other)

    @overload
    def __rmod__(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __rmod__(self, other: float) -> RealMatrixLike[M_co, N_co]: ...

    def __rmod__(self, other):
        return self.target.__rmod__(other)

    @overload
    def __rdivmod__(self, other: int) -> tuple[IntegralMatrixLike[M_co, N_co], IntegralMatrixLike[M_co, N_co]]: ...
    @overload
    def __rdivmod__(self, other: float) -> tuple[RealMatrixLike[M_co, N_co], RealMatrixLike[M_co, N_co]]: ...

    def __rdivmod__(self, other):
        return self.target.__rdivmod__(other)

    def __rlshift__(self, other: int) -> IntegralMatrixLike[M_co, N_co]:
        return self.target.__rlshift__(other)

    def __rrshift__(self, other: int) -> IntegralMatrixLike[M_co, N_co]:
        return self.target.__rrshift__(other)

    def __rand__(self, other: int) -> IntegralMatrixLike[M_co, N_co]:
        return self.target.__rand__(other)

    def __rxor__(self, other: int) -> IntegralMatrixLike[M_co, N_co]:
        return self.target.__rxor__(other)

    def __ror__(self, other: int) -> IntegralMatrixLike[M_co, N_co]:
        return self.target.__ror__(other)

    def __neg__(self) -> IntegralMatrixLike[M_co, N_co]:
        return self.target.__neg__()

    def __abs__(self) -> IntegralMatrixLike[M_co, N_co]:
        return self.target.__abs__()

    def __invert__(self) -> IntegralMatrixLike[M_co, N_co]:
        return self.target.__invert__()

    @property
    def target(self) -> IntegralMatrixLike[M_co, N_co]:
        return self._target

    def transpose(self) -> IntegralMatrixLike[N_co, M_co]:
        return self.target.transpose()

    def flip(self, *, by: Rule = Rule.ROW) -> IntegralMatrixLike[M_co, N_co]:
        return self.target.flip(by=by)

    def reverse(self) -> IntegralMatrixLike[M_co, N_co]:
        return self.target.reverse()

    @overload
    def lesser(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self, other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self, other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        return self.target.lesser(other)

    @overload
    def lesser_equal(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self, other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self, other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        return self.target.lesser_equal(other)

    @overload
    def greater(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self, other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self, other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        return self.target.greater(other)

    @overload
    def greater_equal(self, other: IntegralMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self, other: int) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self, other: RealMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self, other: float) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        return self.target.greater_equal(other)


class TimedeltaMatrixView(TimedeltaMatrixLike[M_co, N_co], MatrixView[timedelta, M_co, N_co]):

    __slots__ = ()

    def __init__(self, target: TimedeltaMatrixLike[M_co, N_co]) -> None:
        self._target: TimedeltaMatrixLike[M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> timedelta: ...
    @overload
    def __getitem__(self, key: slice) -> TimedeltaMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> timedelta: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> TimedeltaMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> TimedeltaMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> TimedeltaMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        return self.target.__getitem__(key)

    @overload
    def __add__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self, other: timedelta) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        return self.target.__add__(other)

    @overload
    def __sub__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: timedelta) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        return self.target.__sub__(other)

    @overload
    def __mul__(self, other: IntegralMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: int) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: RealMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __mul__(self, other: float) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __mul__(self, other):
        return self.target.__mul__(other)

    @overload
    def __truediv__(self, other: IntegralMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: int) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: RealMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: float) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> RealMatrixLike[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: timedelta) -> RealMatrixLike[M_co, N_co]: ...

    def __truediv__(self, other):
        return self.target.__truediv__(other)

    @overload
    def __floordiv__(self, other: IntegralMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self, other: int) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def __floordiv__(self, other: timedelta) -> IntegralMatrixLike[M_co, N_co]: ...

    def __floordiv__(self, other):
        return self.target.__floordiv__(other)

    @overload
    def __mod__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __mod__(self, other: timedelta) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __mod__(self, other):
        return self.target.__mod__(other)

    @overload
    def __divmod__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> tuple[IntegralMatrixLike[M_co, N_co], TimedeltaMatrixLike[M_co, N_co]]: ...
    @overload
    def __divmod__(self, other: timedelta) -> tuple[IntegralMatrixLike[M_co, N_co], TimedeltaMatrixLike[M_co, N_co]]: ...

    def __divmod__(self, other):
        return self.target.__divmod__(other)

    def __radd__(self, other: timedelta) -> TimedeltaMatrixLike[M_co, N_co]:
        return self.target.__radd__(other)

    def __rsub__(self, other: timedelta) -> TimedeltaMatrixLike[M_co, N_co]:
        return self.target.__rsub__(other)

    def __neg__(self) -> TimedeltaMatrixLike[M_co, N_co]:
        return self.target.__neg__()

    def __abs__(self) -> TimedeltaMatrixLike[M_co, N_co]:
        return self.target.__abs__()

    @property
    def target(self) -> TimedeltaMatrixLike[M_co, N_co]:
        return self._target

    @overload
    def lesser(self, other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self, other: timedelta) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        return self.target.lesser(other)

    @overload
    def lesser_equal(self, other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self, other: timedelta) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        return self.target.lesser_equal(other)

    @overload
    def greater(self, other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self, other: timedelta) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        return self.target.greater(other)

    @overload
    def greater_equal(self, other: TimedeltaMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self, other: timedelta) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        return self.target.greater_equal(other)


class DatetimeMatrixView(DatetimeMatrixLike[M_co, N_co], MatrixView[datetime, M_co, N_co]):

    __slots__ = ()

    def __init__(self, target: DatetimeMatrixLike[M_co, N_co]) -> None:
        self._target: DatetimeMatrixLike[M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> datetime: ...
    @overload
    def __getitem__(self, key: slice) -> DatetimeMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> datetime: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> DatetimeMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> DatetimeMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> DatetimeMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        return self.target.__getitem__(key)

    @overload
    def __add__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    def __add__(self, other: timedelta) -> DatetimeMatrixLike[M_co, N_co]: ...

    def __add__(self, other):
        return self.target.__add__(other)

    @overload
    def __sub__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: timedelta) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: DatetimeMatrixLike[M_co, N_co]) -> TimedeltaMatrixLike[M_co, N_co]: ...
    @overload
    def __sub__(self, other: datetime) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __sub__(self, other):
        return self.target.__sub__(other)

    @overload
    def __rsub__(self, other: TimedeltaMatrixLike[M_co, N_co]) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self, other: timedelta) -> DatetimeMatrixLike[M_co, N_co]: ...
    @overload
    def __rsub__(self, other: datetime) -> TimedeltaMatrixLike[M_co, N_co]: ...

    def __rsub__(self, other):
        return self.target.__rsub__(other)

    @property
    def target(self) -> DatetimeMatrixLike[M_co, N_co]:
        return self._target

    @overload
    def lesser(self, other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser(self, other: datetime) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser(self, other):
        return self.target.lesser(other)

    @overload
    def lesser_equal(self, other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def lesser_equal(self, other: datetime) -> IntegralMatrixLike[M_co, N_co]: ...

    def lesser_equal(self, other):
        return self.target.lesser_equal(other)

    @overload
    def greater(self, other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater(self, other: datetime) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater(self, other):
        return self.target.greater(other)

    @overload
    def greater_equal(self, other: DatetimeMatrixLike[M_co, N_co]) -> IntegralMatrixLike[M_co, N_co]: ...
    @overload
    def greater_equal(self, other: datetime) -> IntegralMatrixLike[M_co, N_co]: ...

    def greater_equal(self, other):
        return self.target.greater_equal(other)
