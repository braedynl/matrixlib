from typing import Generic, TypeVar, final

from typing_extensions import override

from .abc import InnerMesh, Mesh, OuterMesh

__all__ = [
    "Transposition",
    "RowFlip",
    "ColFlip",
    "Rotation090",
    "Rotation180",
    "Rotation270",
]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


@final
class Transposition(OuterMesh[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: Mesh[N_co, M_co, T_co]

    def __init__(self, target: Mesh[N_co, M_co, T_co]) -> None:
        self.target = target

    @override
    def __repr__(self) -> str:
        return f"Transposition(target={self.target!r})"

    @property
    @override
    def nrows(self) -> M_co:
        return self.target.ncols

    @property
    @override
    def ncols(self) -> N_co:
        return self.target.nrows

    @override
    def outer_get(self, row_index: int, col_index: int) -> T_co:
        return self.target.outer_get(col_index, row_index)


@final
class RowFlip(OuterMesh[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: Mesh[M_co, N_co, T_co]

    def __init__(self, target: Mesh[M_co, N_co, T_co]) -> None:
        self.target = target

    @override
    def __repr__(self) -> str:
        return f"RowFlip(target={self.target!r})"

    @property
    @override
    def nrows(self) -> M_co:
        return self.target.nrows

    @property
    @override
    def ncols(self) -> N_co:
        return self.target.ncols

    @override
    def outer_get(self, row_index: int, col_index: int) -> T_co:
        row_index = self.nrows - row_index - 1
        return self.target.outer_get(row_index, col_index)


@final
class ColFlip(OuterMesh[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: Mesh[M_co, N_co, T_co]

    def __init__(self, target: Mesh[M_co, N_co, T_co]) -> None:
        self.target = target

    @override
    def __repr__(self) -> str:
        return f"ColFlip(target={self.target!r})"

    @property
    @override
    def nrows(self) -> M_co:
        return self.target.nrows

    @property
    @override
    def ncols(self) -> N_co:
        return self.target.ncols

    @override
    def outer_get(self, row_index: int, col_index: int) -> T_co:
        col_index = self.ncols - col_index - 1
        return self.target.outer_get(row_index, col_index)


@final
class Rotation090(OuterMesh[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: Mesh[N_co, M_co, T_co]

    def __init__(self, target: Mesh[N_co, M_co, T_co]) -> None:
        self.target = target

    @override
    def __repr__(self) -> str:
        return f"Rotation090(target={self.target!r})"

    @property
    @override
    def nrows(self) -> M_co:
        return self.target.ncols

    @property
    @override
    def ncols(self) -> N_co:
        return self.target.nrows

    @override
    def outer_get(self, row_index: int, col_index: int) -> T_co:
        row_index = self.nrows - row_index - 1
        return self.target.outer_get(col_index, row_index)


@final
class Rotation180(InnerMesh[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: Mesh[M_co, N_co, T_co]

    def __init__(self, target: Mesh[M_co, N_co, T_co]) -> None:
        self.target = target

    @override
    def __repr__(self) -> str:
        return f"Rotation180(target={self.target!r})"

    @property
    @override
    def nrows(self) -> M_co:
        return self.target.nrows

    @property
    @override
    def ncols(self) -> N_co:
        return self.target.ncols

    @override
    def inner_get(self, index: int) -> T_co:
        index = len(self) - index - 1
        return self.target.inner_get(index)


@final
class Rotation270(OuterMesh[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: Mesh[N_co, M_co, T_co]

    def __init__(self, target: Mesh[N_co, M_co, T_co]) -> None:
        self.target = target

    @override
    def __repr__(self) -> str:
        return f"Rotation270(target={self.target!r})"

    @property
    @override
    def nrows(self) -> M_co:
        return self.target.ncols

    @property
    @override
    def ncols(self) -> N_co:
        return self.target.nrows

    @override
    def outer_get(self, row_index: int, col_index: int) -> T_co:
        col_index = self.ncols - col_index - 1
        return self.target.outer_get(col_index, row_index)
