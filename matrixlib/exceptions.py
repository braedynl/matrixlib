from __future__ import annotations

__all__ = [
    "ShapeError",
    "NegativeDimensionError",
    "MismatchedDimensionError",
    "ReshapeError",
]


class ShapeError(ValueError):
    """Raised when an operation is provided an unexpected shape.

    **Note**: ``ShapeError`` exceptions should **not** be caught in a
    try-except. Most, if not all ``Matrix`` operations raise ``ShapeError``
    solely in debug mode - checks that would raise ``ShapeError`` under normal
    circumstances are removed entirely when debug mode is active.
    """

    __slots__ = ()


class NegativeDimensionError(ShapeError):
    """Raised when a shape contains a negative dimension."""

    __slots__ = ()


class MismatchedDimensionError(ShapeError):
    """Raised when an operation is given shapes that are unequal in some
    manner, where the operation expected otherwise.
    """

    __slots__ = ()


class ReshapeError(ShapeError):
    """Raised when attempting to cast a sized object (matrices included) into
    a shape whose dimensions cannot losslessly fit the object's size.
    """

    __slots__ = ()
