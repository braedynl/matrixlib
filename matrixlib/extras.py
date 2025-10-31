from __future__ import annotations

__all__ = ["vec2", "vec3"]

from typing import Literal

from .builtins import Real, RealMatrix


def vec2[RealT: Real = Real](x: RealT, y: RealT) -> RealMatrix[Literal[2], Literal[1], RealT]:
    """Convenience function for constructing a ``RealMatrix`` of shape
    ``(2, 1)``, commonly used in 2D space models.
    """
    return RealMatrix[Literal[2], Literal[1], RealT].col((x, y))


def vec3[RealT: Real = Real](x: RealT, y: RealT, z: RealT) -> RealMatrix[Literal[3], Literal[1], RealT]:
    """Convenience function for constructing a ``RealMatrix`` of shape
    ``(3, 1)``, commonly used in 3D space models.
    """
    return RealMatrix[Literal[3], Literal[1], RealT].col((x, y, z))
