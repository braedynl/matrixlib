from abc import ABCMeta
from typing import Any, Literal, Optional, TypeVar

from ..abc import MatrixLike

__all__ = ["MatrixLikeView"]

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", bound=int, covariant=True)
N_co = TypeVar("N_co", bound=int, covariant=True)

MatrixLikeViewT = TypeVar("MatrixLikeViewT", bound=MatrixLikeView)


class MatrixLikeView(MatrixLike[T_co, M_co, N_co], metaclass=ABCMeta):

    __slots__: tuple[Literal["_target"]]

    def __init__(self, target: MatrixLike[T_co, M_co, N_co]) -> None: ...
    def __repr__(self) -> str: ...
    def __deepcopy__(self: MatrixLikeViewT, memo: Optional[dict[int, Any]] = None) -> MatrixLikeViewT: ...
    def __copy__(self: MatrixLikeViewT) -> MatrixLikeViewT: ...
