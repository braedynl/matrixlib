import functools
import operator

__all__ = ["checked_map", "compare", "multiply"]


def checked_map(func, a, *bx):
    u = a.shape
    for b in bx:
        if u != (v := b.shape):
            raise ValueError(f"matrix of shape {u} is incompatible with operand shape {v}")
    yield from map(func, a, *bx)


def compare(a, b):
    u, v = (a.shape, b.shape)
    if u != v:
        raise ValueError(f"matrix of shape {u} is incompatible with operand shape {v}")
    for x, y in zip(a, b):
        if x < y:
            return -1
        if x > y:
            return 1
    return 0


def multiply(a, b):
    u, v = (a.shape, b.shape)
    m, n = u
    p, q = v
    if n != p:
        raise ValueError(f"matrix of shape {u} is incompatible with operand shape {v}")
    if not n:
        raise ValueError  # TODO: error message
    ix = range(m)
    jx = range(q)
    kx = range(n)
    yield from (
        functools.reduce(
            operator.add,
            map(lambda k: a[i * n + k] * b[k * q + j], kx),
        )
        for i in ix
        for j in jx
    )
