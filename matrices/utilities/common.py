__all__ = ["checked_map"]


def checked_map(func, a, *bx):
    u = a.shape
    for b in bx:
        if u != (v := b.shape):
            raise ValueError(f"matrix of shape {u} is incompatible with operand shape {v}")
    yield from map(func, a, *bx)
