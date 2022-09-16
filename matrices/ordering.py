import operator

from .generic import GenericMatrix

__all__ = ["OrderingMatrix"]


class OrderingMatrix(GenericMatrix):
    """Subclass of `GenericMatrix` that adds element-wise `<`, `>`, `<=` and
    `>=` operators
    """

    __slots__ = ()

    def __lt__(self, other):
        """Element-wise `__lt__()`"""
        return self.binary_operator(operator.lt, other)

    def __gt__(self, other):
        """Element-wise `__gt__()`"""
        return self.binary_operator(operator.gt, other)

    def __le__(self, other):
        """Element-wise `__le__()`"""
        return self.binary_operator(operator.le, other)

    def __ge__(self, other):
        """Element-wise `__ge__()`"""
        return self.binary_operator(operator.ge, other)
