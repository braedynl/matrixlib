import operator
from collections.abc import Callable
from typing import Any, TypeVar

from ..abc import ShapedIndexable, ShapedIterable
from .matrix_map import MatrixMap
from .matrix_product import MatrixProduct
from .vectorize import vectorize

__all__ = [
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
    "__gt__",
    "__ge__",
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__divmod__",
    "__lshift__",
    "__rshift__",
    "__and__",
    "__xor__",
    "__or__",
    "__neg__",
    "__pos__",
    "__abs__",
    "__invert__",
    "__matmul__",
    "conjugate",
]

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
P = TypeVar("P", bound=int)


__lt__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]] = vectorize()(operator.__lt__)
__le__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]] = vectorize()(operator.__le__)
__eq__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]] = vectorize()(operator.__eq__)
__ne__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]] = vectorize()(operator.__ne__)
__gt__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]] = vectorize()(operator.__gt__)
__ge__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]] = vectorize()(operator.__ge__)

__add__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]]      = vectorize()(operator.__add__)
__sub__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]]      = vectorize()(operator.__sub__)
__mul__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]]      = vectorize()(operator.__mul__)
__truediv__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]]  = vectorize()(operator.__truediv__)
__floordiv__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]] = vectorize()(operator.__floordiv__)
__mod__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]]      = vectorize()(operator.__mod__)
__divmod__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]]   = vectorize()(divmod)
__lshift__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]]   = vectorize()(operator.__lshift__)
__rshift__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]]   = vectorize()(operator.__rshift__)
__and__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]]      = vectorize()(operator.__and__)
__xor__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]]      = vectorize()(operator.__xor__)
__or__: Callable[[ShapedIterable[Any, M, N], ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]]       = vectorize()(operator.__or__)
__neg__: Callable[[ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]]                                 = vectorize()(operator.__neg__)
__pos__: Callable[[ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]]                                 = vectorize()(operator.__pos__)
__abs__: Callable[[ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]]                                 = vectorize()(abs)
__invert__: Callable[[ShapedIterable[Any, M, N]], MatrixMap[Any, M, N]]                              = vectorize()(operator.__invert__)

def __matmul__(a: ShapedIndexable[Any, M, N], b: ShapedIndexable[Any, N, P], /) -> MatrixProduct[Any, M, P]:
    return MatrixProduct(a, b)

@vectorize()
def conjugate(a: Any, /) -> Any:
    return a.conjugate()
