
__all__ = ["checked_map"]


def checked_map(func, matrix, /, *matrices):
    """Return a ``map`` across one or more matrices, raising ``ValueError`` if
    not all of the input matrices have equal shapes
    """
    if not matrices:
        return map(func, matrix)
    u = matrix.shape
    for other_matrix in matrices:
        v = other_matrix.shape
        if u != v:
            raise ValueError(f"incompatible shapes {u}, {v}")
    return map(func, matrix, *matrices)
