
__all__ = ["checked_map"]


def checked_map(func, matrix, /, *matrices, reverse_args=False):
    """Return a ``map`` across one or more matrices, raising ``ValueError`` if
    not all of the input matrices have equal shapes

    If ``reverse_args=True``, the input matrices are passed to the ``map``
    constructor in reverse order (equivalent to reversing the arguments of
    ``func``).
    """
    if not matrices:
        return map(func, matrix)
    u = matrix.shape
    for other_matrix in matrices:
        v = other_matrix.shape
        if u != v:
            raise ValueError(f"incompatible shapes {u}, {v}")
    if reverse_args:
        return map(func, *reversed(matrices), matrix)
    return map(func, matrix, *matrices)
