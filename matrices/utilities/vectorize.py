from .matrix_map import MatrixMap

__all__ = ["vectorize"]


def vectorize():
    """Convert a scalar-based function into a matrix-based one

    For a function, ``f()``, whose signature is ``f(x: T, ...) -> S``,
    ``vectorize()`` returns a wrapper of ``f()`` whose signature is
    ``f(x: MatrixLike[T, M, N], ...) -> MatrixMap[S, M, N]``. Equivalent to
    constructing ``MatrixMap(f, x, ...)``.

    Vectorization will only be applied to positional function arguments.
    Keyword arguments are stripped, and will no longer be passable.
    """

    def vectorize_decorator(func, /):

        def vectorize_wrapper(matrix1, /, *matrices):
            return MatrixMap(func, matrix1, *matrices)

        vectorize_wrapper.__module__   = func.__module__
        vectorize_wrapper.__name__     = func.__name__
        vectorize_wrapper.__qualname__ = func.__qualname__
        vectorize_wrapper.__doc__      = func.__doc__

        return vectorize_wrapper

    return vectorize_decorator
