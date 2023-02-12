
__all__ = ["checked_map", "checked_rmap"]


def checked_map(func, matrix, /, *matrices):
    """Return a ``map`` across one or more matrices, raising ``ValueError`` if
    not all of the input matrices have equal shapes
    """
    if not matrices:
        return map(func, matrix)
    shape1 = matrix.shape
    for other_matrix in matrices:
        shape2 = other_matrix.shape
        if shape1 != shape2:
            raise ValueError(f"incompatible shapes {shape1}, {shape2}")
    return map(func, matrix, *matrices)


def checked_rmap(func, matrix, /, *matrices):
    """Return a ``map`` across one or more matrices in reverse order, raising
    ``ValueError`` if not all of the input matrices have equal shapes
    """
    return checked_map(func, *reversed(matrices), matrix)
