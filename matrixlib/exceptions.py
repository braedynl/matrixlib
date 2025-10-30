from __future__ import annotations

__all__ = [
    "ShapeError",
    "NegativeDimensionError",
    "MismatchedDimensionError",
    "ReshapeError",
    "DemotionError",
    "check_positive_shape",
    "check_equal_shapes",
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


class DemotionError(ShapeError):
    """Raised when attempting to demote a shaped object that does not contain
    exactly one value.
    """

    __slots__ = ()


def check_positive_shape(shape: tuple[int, int], /) -> None:
    """Raise ``NegativeDimensionError`` if the given shape contains a negative
    dimension, otherwise do nothing.
    """
    if shape[0] < 0 or shape[1] < 0:
        raise NegativeDimensionError("shape dimensions must be positive")


def check_equal_shapes(shape1: tuple[int, int], shape2: tuple[int, int], /) -> None:
    """Raise ``MismatchedDimensionError`` if the two given shapes are not
    equal, otherwise do nothing.
    """
    if shape1 != shape2:
        raise MismatchedDimensionError(f"unequal shapes, {shape1} and {shape2}")
