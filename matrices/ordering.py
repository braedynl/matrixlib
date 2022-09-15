import operator

from .base import BaseMatrix

__all__ = ["OrderingMatrix"]


class OrderingMatrix(BaseMatrix):
    """Subclass of `BaseMatrix` that adds element-wise `<`, `>`, `<=` and `>=`
    operators
    """

    __slots__ = ()

    def __lt__(self, other):
        """Element-wise less than"""
        data = self.map(operator.lt, other)
        if data is NotImplemented:
            return data
        shape = self.shape
        return BaseMatrix.wrap(data, shape=shape.copy())

    def __gt__(self, other):
        """Element-wise greater than"""
        data = self.map(operator.gt, other)
        if data is NotImplemented:
            return data
        shape = self.shape
        return BaseMatrix.wrap(data, shape=shape.copy())

    def __le__(self, other):
        """Element-wise less than/equal to"""
        data = self.map(operator.le, other)
        if data is NotImplemented:
            return data
        shape = self.shape
        return BaseMatrix.wrap(data, shape=shape.copy())

    def __ge__(self, other):
        """Element-wise greater than/equal to"""
        data = self.map(operator.ge, other)
        if data is NotImplemented:
            return data
        shape = self.shape
        return BaseMatrix.wrap(data, shape=shape.copy())
