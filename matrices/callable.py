from .base import BaseMatrix

__all__ = ["CallableMatrix"]


class CallableMatrix(BaseMatrix):
    """Subclass of `BaseMatrix` that adds element-wise calling"""

    __slots__ = ()

    def __call__(self, *args, **kwargs):
        """Element-wise call"""
        data = [func(*args, **kwargs) for func in self.data]
        shape = self.shape
        return BaseMatrix.wrap(data, shape=shape.copy())
