from typing import Any, Generic, Literal, TypeVar, final

from typing_extensions import override

from .abc import Mesh, OuterMesh

__all__ = ["RowSheer", "ColSheer"]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


@final
class RowSheer(OuterMesh[Literal[1], N_co, T_co], Generic[N_co, T_co]):

    __slots__ = ("target", "row_index")

    target: Mesh[Any, N_co, T_co]
    row_index: int

    def __init__(self, target: Mesh[Any, N_co, T_co], row_index: int) -> None:
        self.target = target
        self.row_index = row_index

    @override
    def __repr__(self) -> str:
        return f"RowSheer(target={self.target!r}, row_index={self.row_index!r})"

    @property
    @override
    def nrows(self) -> Literal[1]:
        return 1

    @property
    @override
    def ncols(self) -> N_co:
        return self.target.ncols

    @override
    def outer_get(self, row_index: int, col_index: int) -> T_co:
        row_index = self.row_index + row_index
        return self.target.outer_get(row_index, col_index)


@final
class ColSheer(OuterMesh[M_co, Literal[1], T_co], Generic[M_co, T_co]):

    __slots__ = ("target", "col_index")

    target: Mesh[M_co, Any, T_co]
    col_index: int

    def __init__(self, target: Mesh[M_co, Any, T_co], col_index: int) -> None:
        self.target = target
        self.col_index = col_index

    @override
    def __repr__(self) -> str:
        return f"ColSheer(target={self.target!r}, col_index={self.col_index!r})"

    @property
    @override
    def nrows(self) -> M_co:
        return self.target.nrows

    @property
    @override
    def ncols(self) -> Literal[1]:
        return 1

    @override
    def outer_get(self, row_index: int, col_index: int) -> T_co:
        col_index = self.col_index + col_index
        return self.target.outer_get(row_index, col_index)
