from .rule import COL, ROW, Rule

__all__ = [
    "Rule", "ROW", "COL",
    "checked_map",
]


def checked_map(func, a, *bx):
    """Return an iterator that computes a function from the arguments of
    equally-shaped matrices

    Raises `ValueError` if there exists a matrix whose shape differs from the
    others.
    """
    if not bx:
        yield from map(func, a)
        return
    u = a.shape
    for b in bx:
        if u != (v := b.shape):
            raise ValueError(f"incompatible shapes {u}, {v}")
    yield from map(func, a, *bx)
