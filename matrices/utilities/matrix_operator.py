import operator

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
    "__neg__",
    "__pos__",
    "__abs__",
    "__and__",
    "__xor__",
    "__or__",
    "__invert__",
    "__matmul__",
    "conjugate",
]


__lt__ = vectorize()(operator.__lt__)
__le__ = vectorize()(operator.__le__)
__eq__ = vectorize()(operator.__eq__)
__ne__ = vectorize()(operator.__ne__)
__gt__ = vectorize()(operator.__gt__)
__ge__ = vectorize()(operator.__ge__)

__add__      = vectorize()(operator.__add__)
__sub__      = vectorize()(operator.__sub__)
__mul__      = vectorize()(operator.__mul__)
__truediv__  = vectorize()(operator.__truediv__)
__floordiv__ = vectorize()(operator.__floordiv__)
__mod__      = vectorize()(operator.__mod__)
__divmod__   = vectorize()(divmod)
__lshift__   = vectorize()(operator.__lshift__)
__rshift__   = vectorize()(operator.__rshift__)
__neg__      = vectorize()(operator.__neg__)
__pos__      = vectorize()(operator.__pos__)
__abs__      = vectorize()(abs)

__and__    = vectorize()(operator.__and__)
__xor__    = vectorize()(operator.__xor__)
__or__     = vectorize()(operator.__or__)
__invert__ = vectorize()(operator.__invert__)

def __matmul__(a, b, /):
    return MatrixProduct(a, b)

@vectorize()
def conjugate(a, /):
    return a.conjugate()
