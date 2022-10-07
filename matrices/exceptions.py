__all__ = ["ShapeError", "IncompatibleShapesError"]

class ShapeError(ValueError):
    """Exception type that acts as the root of all errors related to matrix
    shapes

    Derives from `ValueError`.
    """
    pass


class IncompatibleShapesError(ShapeError):
    """Raised if two matrix shapes are incompatible

    Derives from `ShapeError`.

    Shapes are considered "compatible" if at least one of the following
    criteria is met:
    - The shapes are equivalent
    - The shapes' products are equivalent, and both contain at least one
      dimension equal to 1 (i.e., both could be represented one-dimensionally)

    In a nutshell: vectors are compatible so long as their sizes match -
    matrices are compatible so long as their shapes match.
    """
    pass
