import operator

from .matrix_product import MatrixProduct
from .vectorize import vectorize

__all__ = [
    "cmp",
    "lt",
    "le",
    "eq",
    "ne",
    "gt",
    "ge",
    "add",
    "sub",
    "mul",
    "div",
    "quo",
    "rem",
    "quo_rem",
    "shl",
    "shr",
    "neg",
    "pos",
    "abs",
    "bit_and",
    "bit_xor",
    "bit_or",
    "bit_not",
    "log_and",
    "log_xor",
    "log_or",
    "mat_mul",
    "conj",
]


def cmp(a, b, /):
    if a is b:
        return 0
    for x, y in zip(a, b):
        if x == y:
            continue
        if x < y:
            return -1
        if x > y:
            return 1
        raise RuntimeError
    u = a.shape
    v = b.shape
    if u is v:
        return 0
    for m, n in zip(u, v):
        if m == n:
            continue
        if m < n:
            return -1
        if m > n:
            return 1
    return 0

lt = vectorize()(operator.__lt__)
le = vectorize()(operator.__le__)
eq = vectorize()(operator.__eq__)
ne = vectorize()(operator.__ne__)
gt = vectorize()(operator.__gt__)
ge = vectorize()(operator.__ge__)

add     = vectorize()(operator.__add__)
sub     = vectorize()(operator.__sub__)
mul     = vectorize()(operator.__mul__)
div     = vectorize()(operator.__truediv__)
quo     = vectorize()(operator.__floordiv__)
rem     = vectorize()(operator.__mod__)
quo_rem = vectorize()(divmod)
shl     = vectorize()(operator.__lshift__)
shr     = vectorize()(operator.__rshift__)
neg     = vectorize()(operator.__neg__)
pos     = vectorize()(operator.__pos__)
abs     = vectorize()(abs)

bit_and = vectorize()(operator.__and__)
bit_xor = vectorize()(operator.__xor__)
bit_or  = vectorize()(operator.__or__)
bit_not = vectorize()(operator.__invert__)

def mat_mul(a, b, /):
    return MatrixProduct(a, b)

@vectorize()
def conj(a, /):
    return a.conjugate()
