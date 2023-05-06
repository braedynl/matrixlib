from typing import Generic, TypeVar

from .abc import BaseAccessor, BaseMatrixAccessor, BaseVectorAccessor

__all__ = [
    "TransposeAccessor",
    "RowFlipAccessor",
    "ColFlipAccessor",
    "Rotate090Accessor",
    "Rotate180Accessor",
    "Rotate270Accessor",
]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class TransposeAccessor(BaseMatrixAccessor[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: BaseAccessor[N_co, M_co, T_co]

    def __init__(self, target: BaseAccessor[N_co, M_co, T_co]) -> None:
        self.target = target

    def __repr__(self) -> str:
        return f"TransposeAccessor(target={self.target!r})"

    def __hash__(self) -> int:
        return hash(self.target)

    @property
    def nrows(self) -> M_co:
        return self.target.ncols

    @property
    def ncols(self) -> N_co:
        return self.target.nrows

    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        return self.target.matrix_access(col_index, row_index)


class RowFlipAccessor(BaseMatrixAccessor[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: BaseAccessor[M_co, N_co, T_co]

    def __init__(self, target: BaseAccessor[M_co, N_co, T_co]) -> None:
        self.target = target

    def __repr__(self) -> str:
        return f"RowFlipAccessor(target={self.target!r})"

    def __hash__(self) -> int:
        return hash(self.target)

    @property
    def nrows(self) -> M_co:
        return self.target.nrows

    @property
    def ncols(self) -> N_co:
        return self.target.ncols

    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        row_index = self.nrows - row_index - 1
        return self.target.matrix_access(row_index, col_index)


class ColFlipAccessor(BaseMatrixAccessor[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: BaseAccessor[M_co, N_co, T_co]

    def __init__(self, target: BaseAccessor[M_co, N_co, T_co]) -> None:
        self.target = target

    def __repr__(self) -> str:
        return f"ColFlipAccessor(target={self.target!r})"

    def __hash__(self) -> int:
        return hash(self.target)

    @property
    def nrows(self) -> M_co:
        return self.target.nrows

    @property
    def ncols(self) -> N_co:
        return self.target.ncols

    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        col_index = self.ncols - col_index - 1
        return self.target.matrix_access(row_index, col_index)


class Rotate090Accessor(BaseMatrixAccessor[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: BaseAccessor[N_co, M_co, T_co]

    def __init__(self, target: BaseAccessor[N_co, M_co, T_co]) -> None:
        self.target = target

    def __repr__(self) -> str:
        return f"Rotate090Accessor(target={self.target!r})"

    def __hash__(self) -> int:
        return hash(self.target)

    @property
    def nrows(self) -> M_co:
        return self.target.ncols

    @property
    def ncols(self) -> N_co:
        return self.target.nrows

    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        row_index = self.nrows - row_index - 1
        return self.target.matrix_access(col_index, row_index)


class Rotate180Accessor(BaseVectorAccessor[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: BaseAccessor[M_co, N_co, T_co]

    def __init__(self, target: BaseAccessor[M_co, N_co, T_co]) -> None:
        self.target = target

    def __repr__(self) -> str:
        return f"Rotate180Accessor(target={self.target!r})"

    def __hash__(self) -> int:
        return hash(self.target)

    @property
    def nrows(self) -> M_co:
        return self.target.nrows

    @property
    def ncols(self) -> N_co:
        return self.target.ncols

    def vector_access(self, index: int) -> T_co:
        index = len(self) - index - 1
        return self.target.vector_access(index)


class Rotate270Accessor(BaseMatrixAccessor[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("target")

    target: BaseAccessor[N_co, M_co, T_co]

    def __init__(self, target: BaseAccessor[N_co, M_co, T_co]) -> None:
        self.target = target

    def __repr__(self) -> str:
        return f"Rotate270Accessor(target={self.target!r})"

    def __hash__(self) -> int:
        return hash(self.target)

    @property
    def nrows(self) -> M_co:
        return self.target.ncols

    @property
    def ncols(self) -> N_co:
        return self.target.nrows

    def matrix_access(self, row_index: int, col_index: int) -> T_co:
        col_index = self.ncols - col_index - 1
        return self.target.matrix_access(col_index, row_index)
