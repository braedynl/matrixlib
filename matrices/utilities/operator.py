from operator import add
from operator import and_ as bitwise_and
from operator import (eq, floordiv, ge, gt, invert, le, lshift, lt, mod, mul,
                      ne, neg)
from operator import or_ as bitwise_or
from operator import pos, rshift, sub, truediv
from operator import xor as bitwise_xor


def logical_and(a, b, /) -> bool:
    return (not not a) and (not not b)


def logical_or(a, b, /) -> bool:
    return (not not a) or (not not b)


def logical_not(a, /) -> bool:
    return not a
