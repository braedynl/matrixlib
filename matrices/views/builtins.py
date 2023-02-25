from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal, Optional, TypeVar, overload

from ..abc import (ComplexMatrixLike, IntegralMatrixLike, MatrixLike,
                   MatrixLikeFlags, RealMatrixLike)
from ..builtins import (ComplexMatrix, ComplexMatrixOperatorsMixin,
                        IntegralMatrix, IntegralMatrixOperatorsMixin, Matrix,
                        MatrixOperatorsMixin, RealMatrix,
                        RealMatrixOperatorsMixin)
from ..rule import COL, ROW, Rule
from .abc import MatrixViewLike

__all__ = [
    # Basic viewers
    "MatrixView", "ComplexMatrixView", "RealMatrixView", "IntegralMatrixView",

    # Flags
    "MatrixTransformFlags", "MatrixTransposeFlags",

    # Base transforms
    "MatrixTransform", "ComplexMatrixTransform", "RealMatrixTransform",
    "IntegralMatrixTransform",

    # Transpose transforms
    "MatrixTranspose", "ComplexMatrixTranspose", "RealMatrixTranspose",
    "IntegralMatrixTranspose",

    # Row flip transforms
    "MatrixRowFlip", "ComplexMatrixRowFlip", "RealMatrixRowFlip",
    "IntegralMatrixRowFlip",

    # Column flip transforms
    "MatrixColFlip", "ComplexMatrixColFlip", "RealMatrixColFlip",
    "IntegralMatrixColFlip",

    # Reverse transforms
    "MatrixReverse", "ComplexMatrixReverse", "RealMatrixReverse",
    "IntegralMatrixReverse",
]

T = TypeVar("T")

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
P = TypeVar("P", bound=int)

MatrixViewT = TypeVar("MatrixViewT", bound="MatrixView")
MatrixTransformT = TypeVar("MatrixTransformT", bound="MatrixTransform")


class MatrixView(MatrixViewLike[T, M, N]):

    __slots__ = ("_target",)

    def __init__(self, target: MatrixLike[T, M, N]) -> None:
        self._target = target

    def __repr__(self) -> str:
        """Return a canonical representation of the view"""
        return f"{self.__class__.__name__}(target={self._target!r})"

    @overload
    def __getitem__(self, key: int) -> T: ...
    @overload
    def __getitem__(self, key: slice) -> MatrixLike[T, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> T: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> MatrixLike[T, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> MatrixLike[T, Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> MatrixLike[T, Any, Any]: ...

    def __getitem__(self, key):
        return self._target.__getitem__(key)

    def __deepcopy__(self: MatrixViewT, memo: Optional[dict[int, Any]] = None) -> MatrixViewT:
        """Return the view"""
        return self

    __copy__ = __deepcopy__

    @property
    def array(self) -> Sequence[T]:
        return self._target.array

    @property
    def shape(self) -> tuple[M, N]:
        return self._target.shape

    @property
    def flags(self) -> MatrixLikeFlags:
        return self._target.flags

    def equal(self, other: MatrixLike[Any, M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.equal(other)

    def not_equal(self, other: MatrixLike[Any, M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.not_equal(other)

    def transpose(self) -> MatrixLike[T, N, M]:
        return self._target.transpose()

    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T, M, N]:
        return self._target.flip(by=by)

    def reverse(self) -> MatrixLike[T, M, N]:
        return self._target.reverse()


class ComplexMatrixView(ComplexMatrixLike[M, N], MatrixView[complex, M, N]):

    __slots__ = ()

    def __init__(self, target: ComplexMatrixLike[M, N]) -> None:
        self._target: ComplexMatrixLike[M, N] = target

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
        return MatrixView.__getitem__(self, key)

    @overload
    def __add__(self, other: IntegralMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __add__(self, other: RealMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __add__(self, other: ComplexMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...

    def __add__(self, other):
        return self._target.__add__(other)

    @overload
    def __sub__(self, other: IntegralMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __sub__(self, other: RealMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __sub__(self, other: ComplexMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...

    def __sub__(self, other):
        return self._target.__sub__(other)

    @overload
    def __mul__(self, other: IntegralMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __mul__(self, other: RealMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __mul__(self, other: ComplexMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...

    def __mul__(self, other):
        return self._target.__mul__(other)

    @overload
    def __matmul__(self, other: IntegralMatrixLike[N, P]) -> ComplexMatrixLike[M, P]: ...
    @overload
    def __matmul__(self, other: RealMatrixLike[N, P]) -> ComplexMatrixLike[M, P]: ...
    @overload
    def __matmul__(self, other: ComplexMatrixLike[N, P]) -> ComplexMatrixLike[M, P]: ...

    def __matmul__(self, other):
        return self._target.__matmul__(other)

    @overload
    def __truediv__(self, other: IntegralMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __truediv__(self, other: RealMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __truediv__(self, other: ComplexMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...

    def __truediv__(self, other):
        return self._target.__truediv__(other)

    @overload
    def __radd__(self, other: IntegralMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __radd__(self, other: RealMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __radd__(self, other: ComplexMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...

    def __radd__(self, other):
        return self._target.__radd__(other)

    @overload
    def __rsub__(self, other: IntegralMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __rsub__(self, other: RealMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __rsub__(self, other: ComplexMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...

    def __rsub__(self, other):
        return self._target.__rsub__(other)

    @overload
    def __rmul__(self, other: IntegralMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __rmul__(self, other: RealMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __rmul__(self, other: ComplexMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...

    def __rmul__(self, other):
        return self._target.__rmul__(other)

    @overload
    def __rmatmul__(self, other: IntegralMatrixLike[P, M]) -> ComplexMatrixLike[P, N]: ...
    @overload
    def __rmatmul__(self, other: RealMatrixLike[P, M]) -> ComplexMatrixLike[P, N]: ...
    @overload
    def __rmatmul__(self, other: ComplexMatrixLike[P, M]) -> ComplexMatrixLike[P, N]: ...

    def __rmatmul__(self, other):
        return self._target.__rmatmul__(other)

    @overload
    def __rtruediv__(self, other: IntegralMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __rtruediv__(self, other: RealMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...
    @overload
    def __rtruediv__(self, other: ComplexMatrixLike[M, N]) -> ComplexMatrixLike[M, N]: ...

    def __rtruediv__(self, other):
        return self._target.__rtruediv__(other)

    def __neg__(self) -> ComplexMatrixLike[M, N]:
        return self._target.__neg__()

    def __abs__(self) -> RealMatrixLike[M, N]:
        return self._target.__abs__()

    def transpose(self) -> ComplexMatrixLike[N, M]:
        return self._target.transpose()

    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[M, N]:
        return self._target.flip(by=by)

    def reverse(self) -> ComplexMatrixLike[M, N]:
        return self._target.reverse()

    def conjugate(self) -> ComplexMatrixLike[M, N]:
        return self._target.conjugate()


class RealMatrixView(RealMatrixLike[M, N], MatrixView[float, M, N]):

    __slots__ = ()

    def __init__(self, target: RealMatrixLike[M, N]) -> None:
        self._target: RealMatrixLike[M, N] = target

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
        return MatrixView.__getitem__(self, key)

    @overload
    def __add__(self, other: IntegralMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...
    @overload
    def __add__(self, other: RealMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...

    def __add__(self, other):
        return self._target.__add__(other)

    @overload
    def __sub__(self, other: IntegralMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...
    @overload
    def __sub__(self, other: RealMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...

    def __sub__(self, other):
        return self._target.__sub__(other)

    @overload
    def __mul__(self, other: IntegralMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...
    @overload
    def __mul__(self, other: RealMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...

    def __mul__(self, other):
        return self._target.__mul__(other)

    @overload
    def __matmul__(self, other: IntegralMatrixLike[N, P]) -> RealMatrixLike[M, P]: ...
    @overload
    def __matmul__(self, other: RealMatrixLike[N, P]) -> RealMatrixLike[M, P]: ...

    def __matmul__(self, other):
        return self._target.__matmul__(other)

    @overload
    def __truediv__(self, other: IntegralMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...
    @overload
    def __truediv__(self, other: RealMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...

    def __truediv__(self, other):
        return self._target.__truediv__(other)

    @overload
    def __floordiv__(self, other: IntegralMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...
    @overload
    def __floordiv__(self, other: RealMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...

    def __floordiv__(self, other):
        return self._target.__floordiv__(other)

    @overload
    def __mod__(self, other: IntegralMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...
    @overload
    def __mod__(self, other: RealMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...

    def __mod__(self, other):
        return self._target.__mod__(other)

    @overload
    def __divmod__(self, other: IntegralMatrixLike[M, N]) -> tuple[RealMatrixLike[M, N], RealMatrixLike[M, N]]: ...
    @overload
    def __divmod__(self, other: RealMatrixLike[M, N]) -> tuple[RealMatrixLike[M, N], RealMatrixLike[M, N]]: ...

    def __divmod__(self, other):
        return self._target.__divmod__(other)

    @overload
    def __radd__(self, other: IntegralMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...
    @overload
    def __radd__(self, other: RealMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...

    def __radd__(self, other):
        return self._target.__radd__(other)

    @overload
    def __rsub__(self, other: IntegralMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...
    @overload
    def __rsub__(self, other: RealMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...

    def __rsub__(self, other):
        return self._target.__rsub__(other)

    @overload
    def __rmul__(self, other: IntegralMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...
    @overload
    def __rmul__(self, other: RealMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...

    def __rmul__(self, other):
        return self._target.__rmul__(other)

    @overload
    def __rmatmul__(self, other: IntegralMatrixLike[P, M]) -> RealMatrixLike[P, N]: ...
    @overload
    def __rmatmul__(self, other: RealMatrixLike[P, M]) -> RealMatrixLike[P, N]: ...

    def __rmatmul__(self, other):
        return self._target.__rmatmul__(other)

    @overload
    def __rtruediv__(self, other: IntegralMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...
    @overload
    def __rtruediv__(self, other: RealMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...

    def __rtruediv__(self, other):
        return self._target.__rtruediv__(other)

    @overload
    def __rfloordiv__(self, other: IntegralMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...
    @overload
    def __rfloordiv__(self, other: RealMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...

    def __rfloordiv__(self, other):
        return self._target.__rfloordiv__(other)

    @overload
    def __rmod__(self, other: IntegralMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...
    @overload
    def __rmod__(self, other: RealMatrixLike[M, N]) -> RealMatrixLike[M, N]: ...

    def __rmod__(self, other):
        return self._target.__rmod__(other)

    @overload
    def __rdivmod__(self, other: IntegralMatrixLike[M, N]) -> tuple[RealMatrixLike[M, N], RealMatrixLike[M, N]]: ...
    @overload
    def __rdivmod__(self, other: RealMatrixLike[M, N]) -> tuple[RealMatrixLike[M, N], RealMatrixLike[M, N]]: ...

    def __rdivmod__(self, other):
        return self._target.__rdivmod__(other)

    def __neg__(self) -> RealMatrixLike[M, N]:
        return self._target.__neg__()

    def __abs__(self) -> RealMatrixLike[M, N]:
        return self._target.__abs__()

    def transpose(self) -> RealMatrixLike[N, M]:
        return self._target.transpose()

    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[M, N]:
        return self._target.flip(by=by)

    def reverse(self) -> RealMatrixLike[M, N]:
        return self._target.reverse()

    @overload
    def lesser(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]: ...
    @overload
    def lesser(self, other: RealMatrixLike[M, N]) -> IntegralMatrixLike[M, N]: ...

    def lesser(self, other):
        return self._target.lesser(other)

    @overload
    def lesser_equal(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]: ...
    @overload
    def lesser_equal(self, other: RealMatrixLike[M, N]) -> IntegralMatrixLike[M, N]: ...

    def lesser_equal(self, other):
        return self._target.lesser_equal(other)

    @overload
    def greater(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]: ...
    @overload
    def greater(self, other: RealMatrixLike[M, N]) -> IntegralMatrixLike[M, N]: ...

    def greater(self, other):
        return self._target.greater(other)

    @overload
    def greater_equal(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]: ...
    @overload
    def greater_equal(self, other: RealMatrixLike[M, N]) -> IntegralMatrixLike[M, N]: ...

    def greater_equal(self, other):
        return self._target.greater_equal(other)


class IntegralMatrixView(IntegralMatrixLike[M, N], MatrixView[int, M, N]):

    __slots__ = ()

    def __init__(self, target: IntegralMatrixLike[M, N]) -> None:
        self._target: IntegralMatrixLike[M, N] = target

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
        return MatrixView.__getitem__(self, key)

    def __add__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__add__(other)

    def __sub__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__sub__(other)

    def __mul__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__mul__(other)

    def __matmul__(self, other: IntegralMatrixLike[N, P]) -> IntegralMatrixLike[M, P]:
        return self._target.__matmul__(other)

    def __truediv__(self, other: IntegralMatrixLike[M, N]) -> RealMatrixLike[M, N]:
        return self._target.__truediv__(other)

    def __floordiv__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__floordiv__(other)

    def __mod__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__mod__(other)

    def __divmod__(self, other: IntegralMatrixLike[M, N]) -> tuple[IntegralMatrixLike[M, N], IntegralMatrixLike[M, N]]:
        return self._target.__divmod__(other)

    def __lshift__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__lshift__(other)

    def __rshift__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__rshift__(other)

    def __and__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__and__(other)

    def __xor__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__xor__(other)

    def __or__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__or__(other)

    def __radd__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__radd__(other)

    def __rsub__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__rsub__(other)

    def __rmul__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__rmul__(other)

    def __rmatmul__(self, other: IntegralMatrixLike[P, M]) -> IntegralMatrixLike[P, N]:
        return self._target.__rmatmul__(other)

    def __rtruediv__(self, other: IntegralMatrixLike[M, N]) -> RealMatrixLike[M, N]:
        return self._target.__rtruediv__(other)

    def __rfloordiv__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__rfloordiv__(other)

    def __rmod__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__rmod__(other)

    def __rdivmod__(self, other: IntegralMatrixLike[M, N]) -> tuple[IntegralMatrixLike[M, N], IntegralMatrixLike[M, N]]:
        return self._target.__rdivmod__(other)

    def __rlshift__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__rlshift__(other)

    def __rrshift__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__rrshift__(other)

    def __rand__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__rand__(other)

    def __rxor__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__rxor__(other)

    def __ror__(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.__ror__(other)

    def __neg__(self) -> IntegralMatrixLike[M, N]:
        return self._target.__neg__()

    def __abs__(self) -> IntegralMatrixLike[M, N]:
        return self._target.__abs__()

    def __invert__(self) -> IntegralMatrixLike[M, N]:
        return self._target.__invert__()

    def transpose(self) -> IntegralMatrixLike[N, M]:
        return self._target.transpose()

    def flip(self, *, by: Rule = Rule.ROW) -> IntegralMatrixLike[M, N]:
        return self._target.flip(by=by)

    def reverse(self) -> IntegralMatrixLike[M, N]:
        return self._target.reverse()

    def lesser(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.lesser(other)

    def lesser_equal(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.lesser_equal(other)

    def greater(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.greater(other)

    def greater_equal(self, other: IntegralMatrixLike[M, N]) -> IntegralMatrixLike[M, N]:
        return self._target.greater_equal(other)


class MatrixTransformFlags(MatrixLikeFlags):

    __slots__ = ()

    @property
    def lazy_array(self) -> bool:
        return True

    @property
    def lazy_shape(self) -> bool:
        return self._matrix.flags.lazy_shape

    @property
    def writable(self) -> bool:
        return self._matrix.flags.writable

    @property
    def growable(self) -> bool:
        return self._matrix.flags.growable

    @property
    def shrinkable(self) -> bool:
        return self._matrix.flags.shrinkable


class MatrixTransposeFlags(MatrixTransformFlags):

    __slots__ = ()

    @property
    def lazy_shape(self) -> bool:
        return True


class MatrixTransform(MatrixOperatorsMixin[T, M, N], MatrixViewLike[T, M, N]):

    __slots__ = ("_target",)

    def __init__(self, target: MatrixLike[T, M, N]) -> None:
        self._target: MatrixLike[T, M, N] = target

    def __repr__(self) -> str:
        """Return a canonical representation of the view"""
        return f"{self.__class__.__name__}(target={self._target!r})"

    @overload
    def __getitem__(self, key: int) -> T: ...
    @overload
    def __getitem__(self, key: slice) -> MatrixLike[T, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> T: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> MatrixLike[T, Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> MatrixLike[T, Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> MatrixLike[T, Any, Any]: ...

    def __getitem__(self, key):
        target = self._target

        if target.flags.lazy_array:
            array = target
        else:
            array = target.array

        permute_matrix_index = self._permute_matrix_index
        permute_vector_index = self._permute_vector_index

        if isinstance(key, tuple):
            row_key, col_key = key

            if isinstance(row_key, slice):
                row_indices = self._resolve_matrix_slice(row_key, by=ROW)

                if isinstance(col_key, slice):
                    col_indices = self._resolve_matrix_slice(col_key, by=COL)
                    return self._decompose(
                        (
                            array[permute_matrix_index(row_index, col_index)]
                            for row_index in row_indices
                            for col_index in col_indices
                        ),
                        shape=(len(row_indices), len(col_indices)),
                    )

                col_index = self._resolve_matrix_index(col_key, by=COL)
                return self._decompose(
                    (
                        array[permute_matrix_index(row_index, col_index)]
                        for row_index in row_indices
                    ),
                    shape=(len(row_indices), 1),
                )

            row_index = self._resolve_matrix_index(row_key, by=ROW)

            if isinstance(col_key, slice):
                col_indices = self._resolve_matrix_slice(col_key, by=COL)
                return self._decompose(
                    (
                        array[permute_matrix_index(row_index, col_index)]
                        for col_index in col_indices
                    ),
                    shape=(1, len(col_indices)),
                )

            col_index = self._resolve_matrix_index(col_key, by=COL)
            return array[permute_matrix_index(row_index, col_index)]

        if isinstance(key, slice):
            val_indices = self._resolve_vector_slice(key)
            return self._decompose(
                (
                    array[permute_vector_index(val_index)]
                    for val_index in val_indices
                ),
                shape=(1, len(val_indices)),
            )

        val_index = self._resolve_vector_index(key)
        return array[permute_vector_index(val_index)]

    def __deepcopy__(self: MatrixTransformT, memo: Optional[dict[int, Any]] = None) -> MatrixTransformT:
        """Return the view"""
        return self

    __copy__ = __deepcopy__

    @property
    def array(self) -> Sequence[T]:
        return tuple(self.values(by=ROW, reverse=False))

    @property
    def shape(self) -> tuple[M, N]:
        return self._target.shape

    @property
    def flags(self) -> MatrixTransformFlags:
        return MatrixTransformFlags(self._target)

    def transpose(self) -> MatrixLike[T, N, M]:
        return MatrixTranspose(self)

    def flip(self, *, by=Rule.ROW) -> MatrixLike[T, M, N]:
        MatrixTransform = (MatrixRowFlip, MatrixColFlip)[by.value]
        return MatrixTransform(self)

    def reverse(self) -> MatrixLike[T, M, N]:
        return MatrixReverse(self)

    def _decompose(self, array: Iterable[T], shape: tuple[M, N]) -> MatrixLike[T, M, N]:
        return Matrix(array, shape)

    def _permute_vector_index(self, val_index: int) -> int:
        return val_index

    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        return row_index * self.ncols + col_index


class ComplexMatrixTransform(
    ComplexMatrixOperatorsMixin[M, N],
    ComplexMatrixLike[M, N],
    MatrixTransform[complex, M, N],
):

    __slots__ = ()

    def __init__(self, target: ComplexMatrixLike[M, N]) -> None:
        super().__init__(target)
        self._target: ComplexMatrixLike[M, N]

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
        return MatrixTransform.__getitem__(self, key)

    def transpose(self) -> ComplexMatrixLike[N, M]:
        return ComplexMatrixTranspose(self)

    def flip(self, *, by=Rule.ROW) -> ComplexMatrixLike[M, N]:
        ComplexMatrixTransform = (ComplexMatrixRowFlip, ComplexMatrixColFlip)[by.value]
        return ComplexMatrixTransform(self)

    def reverse(self) -> ComplexMatrixLike[M, N]:
        return ComplexMatrixReverse(self)

    def _decompose(self, array: Iterable[complex], shape: tuple[M, N]) -> ComplexMatrixLike[M, N]:
        return ComplexMatrix(array, shape)


class RealMatrixTransform(
    RealMatrixOperatorsMixin[M, N],
    RealMatrixLike[M, N],
    MatrixTransform[float, M, N],
):

    __slots__ = ()

    def __init__(self, target: RealMatrixLike[M, N]) -> None:
        super().__init__(target)
        self._target: RealMatrixLike[M, N]

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
        return MatrixTransform.__getitem__(self, key)

    def transpose(self) -> RealMatrixLike[N, M]:
        return RealMatrixTranspose(self)

    def flip(self, *, by=Rule.ROW) -> RealMatrixLike[M, N]:
        RealMatrixTransform = (RealMatrixRowFlip, RealMatrixColFlip)[by.value]
        return RealMatrixTransform(self)

    def reverse(self) -> RealMatrixLike[M, N]:
        return RealMatrixReverse(self)

    def _decompose(self, array: Iterable[float], shape: tuple[M, N]) -> RealMatrixLike[M, N]:
        return RealMatrix(array, shape)


class IntegralMatrixTransform(
    IntegralMatrixOperatorsMixin[M, N],
    IntegralMatrixLike[M, N],
    MatrixTransform[int, M, N],
):

    __slots__ = ()

    def __init__(self, target: IntegralMatrixLike[M, N]) -> None:
        super().__init__(target)
        self._target: IntegralMatrixLike[M, N]

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
        return MatrixTransform.__getitem__(self, key)

    def transpose(self) -> IntegralMatrixLike[N, M]:
        return IntegralMatrixTranspose(self)

    def flip(self, *, by=Rule.ROW) -> IntegralMatrixLike[M, N]:
        IntegralMatrixTransform = (IntegralMatrixRowFlip, IntegralMatrixColFlip)[by.value]
        return IntegralMatrixTransform(self)

    def reverse(self) -> IntegralMatrixLike[M, N]:
        return IntegralMatrixReverse(self)

    def _decompose(self, array: Iterable[int], shape: tuple[M, N]) -> IntegralMatrix[M, N]:
        return IntegralMatrix(array, shape)


class MatrixTranspose(MatrixTransform[T, M, N]):

    __slots__ = ()

    def __init__(self, target: MatrixLike[T, N, M]) -> None:
        super().__init__(target)  # type: ignore[arg-type]
        self._target: MatrixLike[T, N, M]  # type: ignore[assignment]

    @property
    def shape(self) -> tuple[M, N]:
        shape = self._target.shape
        return (shape[1], shape[0])

    @property
    def nrows(self) -> M:
        return self._target.ncols

    @property
    def ncols(self) -> N:
        return self._target.nrows

    @property
    def flags(self) -> MatrixTransposeFlags:
        return MatrixTransposeFlags(self._target)

    def transpose(self) -> MatrixLike[T, N, M]:
        return MatrixView(self._target)

    def _permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        return self._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )

    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        return col_index * self.nrows + row_index


class ComplexMatrixTranspose(ComplexMatrixTransform[M, N], MatrixTranspose[complex, M, N]):

    __slots__ = ()

    def __init__(self, target: ComplexMatrixLike[N, M]) -> None:
        super().__init__(target)  # type: ignore[arg-type]
        self._target: ComplexMatrixLike[N, M]  # type: ignore[assignment]

    def transpose(self) -> ComplexMatrixLike[N, M]:
        return ComplexMatrixView(self._target)


class RealMatrixTranspose(RealMatrixTransform[M, N], MatrixTranspose[float, M, N]):

    __slots__ = ()

    def __init__(self, target: RealMatrixLike[N, M]) -> None:
        super().__init__(target)  # type: ignore[arg-type]
        self._target: RealMatrixLike[N, M]  # type: ignore[assignment]

    def transpose(self) -> RealMatrixLike[N, M]:
        return RealMatrixView(self._target)


class IntegralMatrixTranspose(IntegralMatrixTransform[M, N], MatrixTranspose[int, M, N]):

    __slots__ = ()

    def __init__(self, target: IntegralMatrixLike[N, M]) -> None:
        super().__init__(target)  # type: ignore[arg-type]
        self._target: IntegralMatrixLike[N, M]  # type: ignore[assignment]

    def transpose(self) -> IntegralMatrixLike[N, M]:
        return IntegralMatrixView(self._target)


class MatrixRowFlip(MatrixTransform[T, M, N]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T, M, N]:
        if by is Rule.ROW:
            return MatrixView(self._target)
        return super().flip(by=by)

    def _permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        return self._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )

    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        row_index = self.nrows - row_index - 1
        return super()._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )


class ComplexMatrixRowFlip(ComplexMatrixTransform[M, N], MatrixRowFlip[complex, M, N]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[M, N]:
        if by is Rule.ROW:
            return ComplexMatrixView(self._target)
        return super().flip(by=by)


class RealMatrixRowFlip(RealMatrixTransform[M, N], MatrixRowFlip[float, M, N]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[M, N]:
        if by is Rule.ROW:
            return RealMatrixView(self._target)
        return super().flip(by=by)


class IntegralMatrixRowFlip(IntegralMatrixTransform[M, N], MatrixRowFlip[int, M, N]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> IntegralMatrixLike[M, N]:
        if by is Rule.ROW:
            return IntegralMatrixView(self._target)
        return super().flip(by=by)


class MatrixColFlip(MatrixTransform[T, M, N]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T, M, N]:
        if by is Rule.COL:
            return MatrixView(self._target)
        return super().flip(by=by)

    def _permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        return self._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )

    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        col_index = self.ncols - col_index - 1
        return super()._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )


class ComplexMatrixColFlip(ComplexMatrixTransform[M, N], MatrixColFlip[complex, M, N]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[M, N]:
        if by is Rule.COL:
            return ComplexMatrixView(self._target)
        return super().flip(by=by)


class RealMatrixColFlip(RealMatrixTransform[M, N], MatrixColFlip[float, M, N]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[M, N]:
        if by is Rule.COL:
            return RealMatrixView(self._target)
        return super().flip(by=by)


class IntegralMatrixColFlip(IntegralMatrixTransform[M, N], MatrixColFlip[int, M, N]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> IntegralMatrixLike[M, N]:
        if by is Rule.COL:
            return IntegralMatrixView(self._target)
        return super().flip(by=by)


class MatrixReverse(MatrixTransform[T, M, N]):

    __slots__ = ()

    def reverse(self) -> MatrixLike[T, M, N]:
        return MatrixView(self._target)

    def _permute_vector_index(self, val_index: int) -> int:
        return len(self) - val_index - 1

    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        return self._permute_vector_index(
            val_index=super()._permute_matrix_index(
                row_index=row_index,
                col_index=col_index,
            ),
        )


class ComplexMatrixReverse(ComplexMatrixTransform[M, N], MatrixReverse[complex, M, N]):

    __slots__ = ()

    def reverse(self) -> ComplexMatrixLike[M, N]:
        return ComplexMatrixView(self._target)


class RealMatrixReverse(RealMatrixTransform[M, N], MatrixReverse[float, M, N]):

    __slots__ = ()

    def reverse(self) -> RealMatrixLike[M, N]:
        return RealMatrixView(self._target)


class IntegralMatrixReverse(IntegralMatrixTransform[M, N], MatrixReverse[int, M, N]):

    __slots__ = ()

    def reverse(self) -> IntegralMatrixLike[M, N]:
        return IntegralMatrixView(self._target)
