from reprlib import recursive_repr
from typing import TypeVar

from ..abc import ComplexMatrixLike, IntegralMatrixLike, RealMatrixLike
from ..rule import Rule
from .abc import MatrixViewLike

T = TypeVar("T")

M = TypeVar("M", covariant=True, bound=int)
N = TypeVar("N", covariant=True, bound=int)

ComplexT = TypeVar("ComplexT", bound=complex)
RealT = TypeVar("RealT", bound=float)
IntegralT = TypeVar("IntegralT", bound=int)


class MatrixView(MatrixViewLike[T, M, N]):

    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    @recursive_repr("...")
    def __repr__(self):
        """Return a canonical representation of the view"""
        return f"{self.__class__.__name__}(target={self._target!r})"

    def __getitem__(self, key):
        return self._target.__getitem__(key)

    def __deepcopy__(self, memo=None):
        """Return the view"""
        return self

    __copy__ = __deepcopy__

    @property
    def array(self):
        return self._target.array

    @property
    def shape(self):
        return self._target.shape

    def equal(self, other):
        return self._target.equal(other)

    def not_equal(self, other):
        return self._target.not_equal(other)

    def transpose(self):
        return self._target.transpose()

    def flip(self, *, by=Rule.ROW):
        return self._target.flip(by=by)

    def reverse(self):
        return self._target.reverse()


class ComplexMatrixView(ComplexMatrixLike[ComplexT, M, N], MatrixView[ComplexT, M, N]):

    __slots__ = ()

    def __add__(self, other):
        return self._target.__add__(other)

    def __sub__(self, other):
        return self._target.__sub__(other)

    def __mul__(self, other):
        return self._target.__mul__(other)

    def __matmul__(self, other):
        return self._target.__matmul__(other)

    def __truediv__(self, other):
        return self._target.__truediv__(other)

    def __radd__(self, other):
        return self._target.__radd__(other)

    def __rsub__(self, other):
        return self._target.__rsub__(other)

    def __rmul__(self, other):
        return self._target.__rmul__(other)

    def __rmatmul__(self, other):
        return self._target.__rmatmul__(other)

    def __rtruediv__(self, other):
        return self._target.__rtruediv__(other)

    def __neg__(self):
        return self._target.__neg__()

    def __abs__(self):
        return self._target.__abs__()

    def conjugate(self):
        return self._target.conjugate()


class RealMatrixView(RealMatrixLike[RealT, M, N], MatrixView[RealT, M, N]):

    __slots__ = ()

    def __lt__(self, other):
        return self._target.__lt__(other)

    def __le__(self, other):
        return self._target.__le__(other)

    def __gt__(self, other):
        return self._target.__gt__(other)

    def __ge__(self, other):
        return self._target.__ge__(other)

    def __add__(self, other):
        return self._target.__add__(other)

    def __sub__(self, other):
        return self._target.__sub__(other)

    def __mul__(self, other):
        return self._target.__mul__(other)

    def __matmul__(self, other):
        return self._target.__matmul__(other)

    def __truediv__(self, other):
        return self._target.__truediv__(other)

    def __floordiv__(self, other):
        return self._target.__floordiv__(other)

    def __mod__(self, other):
        return self._target.__mod__(other)

    def __divmod__(self, other):
        return self._target.__divmod__(other)

    def __radd__(self, other):
        return self._target.__radd__(other)

    def __rsub__(self, other):
        return self._target.__rsub__(other)

    def __rmul__(self, other):
        return self._target.__rmul__(other)

    def __rmatmul__(self, other):
        return self._target.__rmatmul__(other)

    def __rtruediv__(self, other):
        return self._target.__rtruediv__(other)

    def __rfloordiv__(self, other):
        return self._target.__rfloordiv__(other)

    def __rmod__(self, other):
        return self._target.__rmod__(other)

    def __rdivmod__(self, other):
        return self._target.__rdivmod__(other)

    def __neg__(self):
        return self._target.__neg__()

    def __abs__(self):
        return self._target.__abs__()

    def lesser(self, other):
        return self._target.lesser(other)

    def lesser_equal(self, other):
        return self._target.lesser_equal(other)

    def greater(self, other):
        return self._target.greater(other)

    def greater_equal(self, other):
        return self._target.greater_equal(other)


class IntegralMatrixView(IntegralMatrixLike[IntegralT, M, N], MatrixView[IntegralT, M, N]):

    __slots__ = ()

    def __lt__(self, other):
        return self._target.__lt__(other)

    def __le__(self, other):
        return self._target.__le__(other)

    def __gt__(self, other):
        return self._target.__gt__(other)

    def __ge__(self, other):
        return self._target.__ge__(other)

    def __add__(self, other):
        return self._target.__add__(other)

    def __sub__(self, other):
        return self._target.__sub__(other)

    def __mul__(self, other):
        return self._target.__mul__(other)

    def __matmul__(self, other):
        return self._target.__matmul__(other)

    def __truediv__(self, other):
        return self._target.__truediv__(other)

    def __floordiv__(self, other):
        return self._target.__floordiv__(other)

    def __mod__(self, other):
        return self._target.__mod__(other)

    def __divmod__(self, other):
        return self._target.__divmod__(other)

    def __lshift__(self, other):
        return self._target.__lshift__(other)

    def __rshift__(self, other):
        return self._target.__rshift__(other)

    def __and__(self, other):
        return self._target.__and__(other)

    def __xor__(self, other):
        return self._target.__xor__(other)

    def __or__(self, other):
        return self._target.__or__(other)

    def __radd__(self, other):
        return self._target.__radd__(other)

    def __rsub__(self, other):
        return self._target.__rsub__(other)

    def __rmul__(self, other):
        return self._target.__rmul__(other)

    def __rmatmul__(self, other):
        return self._target.__rmatmul__(other)

    def __rtruediv__(self, other):
        return self._target.__rtruediv__(other)

    def __rfloordiv__(self, other):
        return self._target.__rfloordiv__(other)

    def __rmod__(self, other):
        return self._target.__rmod__(other)

    def __rdivmod__(self, other):
        return self._target.__rdivmod__(other)

    def __rlshift__(self, other):
        return self._target.__rlshift__(other)

    def __rrshift__(self, other):
        return self._target.__rrshift__(other)

    def __rand__(self, other):
        return self._target.__rand__(other)

    def __rxor__(self, other):
        return self._target.__rxor__(other)

    def __ror__(self, other):
        return self._target.__ror__(other)

    def __neg__(self):
        return self._target.__neg__()

    def __abs__(self):
        return self._target.__abs__()

    def __invert__(self):
        return self._target.__invert__()

    def lesser(self, other):
        return self._target.lesser(other)

    def lesser_equal(self, other):
        return self._target.lesser_equal(other)

    def greater(self, other):
        return self._target.greater(other)

    def greater_equal(self, other):
        return self._target.greater_equal(other)
