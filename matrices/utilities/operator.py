from operator import add as scalar_add
from operator import and_ as scalar_bitwise_and
from operator import eq as scalar_equal
from operator import floordiv as scalar_floordiv
from operator import ge as scalar_greater_equal
from operator import gt as scalar_greater
from operator import invert as scalar_invert
from operator import le as scalar_lesser_equal
from operator import lshift as scalar_lshift
from operator import lt as scalar_lesser
from operator import mod as scalar_mod
from operator import mul as scalar_mul
from operator import ne as scalar_not_equal
from operator import neg as scalar_neg
from operator import or_ as scalar_bitwise_or
from operator import pos as scalar_pos
from operator import rshift as scalar_rshift
from operator import sub as scalar_sub
from operator import truediv as scalar_truediv
from operator import xor as scalar_bitwise_xor

from .vectorize import vectorize

__all__ = [
    "add",
    "bitwise_and",
    "equal",
    "floordiv",
    "greater_equal",
    "greater",
    "invert",
    "lesser_equal",
    "lshift",
    "lesser",
    "mod",
    "mul",
    "not_equal",
    "neg",
    "bitwise_or",
    "pos",
    "rshift",
    "sub",
    "truediv",
    "bitwise_xor",
    "logical_and",
    "logical_or",
    "logical_not",
]


add = vectorize()(scalar_add)
bitwise_and = vectorize()(scalar_bitwise_and)
equal = vectorize()(scalar_equal)
floordiv = vectorize()(scalar_floordiv)
greater_equal = vectorize()(scalar_greater_equal)
greater = vectorize()(scalar_greater)
invert = vectorize()(scalar_invert)
lesser_equal = vectorize()(scalar_lesser_equal)
lshift = vectorize()(scalar_lshift)
lesser = vectorize()(scalar_lesser)
mod = vectorize()(scalar_mod)
mul = vectorize()(scalar_mul)
not_equal = vectorize()(scalar_not_equal)
neg = vectorize()(scalar_neg)
bitwise_or = vectorize()(scalar_bitwise_or)
pos = vectorize()(scalar_pos)
rshift = vectorize()(scalar_rshift)
sub = vectorize()(scalar_sub)
truediv = vectorize()(scalar_truediv)
bitwise_xor = vectorize()(scalar_bitwise_xor)


@vectorize()
def logical_and(a, b, /) -> bool:
    return not not (a and b)

@vectorize()
def logical_or(a, b, /) -> bool:
    return not not (a or b)

@vectorize()
def logical_not(a, /) -> bool:
    return not a
