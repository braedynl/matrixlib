from builtins import abs as _abs
from builtins import divmod as _divmod
from operator import add as _add
from operator import and_ as _bitwise_and
from operator import eq as _equal
from operator import floordiv as _floordiv
from operator import ge as _greater_equal
from operator import gt as _greater
from operator import invert as _invert
from operator import le as _lesser_equal
from operator import lshift as _lshift
from operator import lt as _lesser
from operator import mod as _mod
from operator import mul as _mul
from operator import ne as _not_equal
from operator import neg as _neg
from operator import or_ as _bitwise_or
from operator import pos as _pos
from operator import rshift as _rshift
from operator import sub as _sub
from operator import truediv as _truediv
from operator import xor as _bitwise_xor

from .vectorize import vectorize

__all__ = [
    "abs",
    "divmod",
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
    "conjugate",
]

abs = vectorize()(_abs)
divmod = vectorize()(_divmod)
add = vectorize()(_add)
bitwise_and = vectorize()(_bitwise_and)
equal = vectorize()(_equal)
floordiv = vectorize()(_floordiv)
greater_equal = vectorize()(_greater_equal)
greater = vectorize()(_greater)
invert = vectorize()(_invert)
lesser_equal = vectorize()(_lesser_equal)
lshift = vectorize()(_lshift)
lesser = vectorize()(_lesser)
mod = vectorize()(_mod)
mul = vectorize()(_mul)
not_equal = vectorize()(_not_equal)
neg = vectorize()(_neg)
bitwise_or = vectorize()(_bitwise_or)
pos = vectorize()(_pos)
rshift = vectorize()(_rshift)
sub = vectorize()(_sub)
truediv = vectorize()(_truediv)
bitwise_xor = vectorize()(_bitwise_xor)


@vectorize()
def logical_and(a, b, /):
    return not not (a and b)

@vectorize()
def logical_or(a, b, /):
    return not not (a or b)

@vectorize()
def logical_not(a, /):
    return not a

@vectorize()
def conjugate(a, /):
    return a.conjugate()
