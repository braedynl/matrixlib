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

        return vectorize_wrapper

    return vectorize_decorator
