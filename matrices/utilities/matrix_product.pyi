from collections.abc import Iterator
from typing import Literal, TypeVar

from ..abc import MatrixLike, ShapedCollection

__all__ = ["MatrixProduct"]

ComplexT_co = TypeVar("ComplexT_co", covariant=True, bound=complex)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)
P_co = TypeVar("P_co", covariant=True, bound=int)


class MatrixProduct(ShapedCollection[ComplexT_co, M_co, P_co]):

    __slots__: tuple[Literal["_shape"], Literal["_matrix1"], Literal["_matrix2"]]

    def __init__(self, matrix1: MatrixLike[ComplexT_co, M_co, N_co], matrix2: MatrixLike[ComplexT_co, N_co, P_co]) -> None: ...
    def __iter__(self) -> Iterator[ComplexT_co]: ...

    @property
    def shape(self) -> tuple[M_co, P_co]: ...
