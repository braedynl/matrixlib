import functools

from .generic import GenericMatrix

__all__ = ["CallableMatrix"]

def call(func, args, kwargs): return func(*args, **kwargs)


class CallableMatrix(GenericMatrix):
    """Subclass of `GenericMatrix` that adds element-wise calling"""

    __slots__ = ()

    def __call__(self, *args, **kwargs):
        """Element-wise call"""
        return self.unary_operator(
            functools.partial(call, args, kwargs),
            out=GenericMatrix,
        )
