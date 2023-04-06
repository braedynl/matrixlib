from typing import Any, Generic, Literal, TypeVar, final

from typing_extensions import override

from .abc import InnerMesh, Mesh, OuterMesh

__all__ = [
    "Slice",
    "RowSlice",
    "ColSlice",
    "RowColSlice",
]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


@final
class Slice(InnerMesh[Literal[1], int, T_co], Generic[T_co]):

    __slots__ = ("target", "window")

    target: Mesh[Any, Any, T_co]
    window: range

    def __init__(self, target: Mesh[Any, Any, T_co], window: range) -> None:
        self.target = target
        self.window = window

    @override
    def __repr__(self) -> str:
        return f"Slice(target={self.target!r}, window={self.window!r})"

    @property
    @override
    def nrows(self) -> Literal[1]:
        return 1

    @property
    @override
    def ncols(self) -> int:
        return len(self.window)

    @override
    def inner_get(self, index: int) -> T_co:
        index = self.window[index]
        return self.target.inner_get(index)


@final
class RowSlice(OuterMesh[Literal[1], int, T_co], Generic[T_co]):

    __slots__ = ("target", "row_index", "col_window")

    target: Mesh[Any, Any, T_co]
    row_index: int
    col_window: range

    def __init__(self, target: Mesh[Any, Any, T_co], row_index: int, col_window: range) -> None:
        self.target = target
        self.row_index = row_index
        self.col_window = col_window

    @override
    def __repr__(self) -> str:
        return f"RowSlice(target={self.target!r}, row_index={self.row_index!r}, col_window={self.col_window!r})"

    @property
    @override
    def nrows(self) -> Literal[1]:
        return 1

    @property
    @override
    def ncols(self) -> int:
        return len(self.col_window)

    @override
    def outer_get(self, row_index: int, col_index: int) -> T_co:
        row_index = self.row_index + row_index
        col_index = self.col_window[col_index]
        return self.target.outer_get(row_index, col_index)


@final
class ColSlice(OuterMesh[int, Literal[1], T_co], Generic[T_co]):

    __slots__ = ("target", "row_window", "col_index")

    target: Mesh[Any, Any, T_co]
    row_window: range
    col_index: int

    def __init__(self, target: Mesh[Any, Any, T_co], row_window: range, col_index: int) -> None:
        self.target = target
        self.row_window = row_window
        self.col_index = col_index

    @override
    def __repr__(self) -> str:
        return f"ColSlice(target={self.target!r}, row_window={self.row_window!r}, col_index={self.col_index!r})"

    @property
    @override
    def nrows(self) -> int:
        return len(self.row_window)

    @property
    @override
    def ncols(self) -> Literal[1]:
        return 1

    @override
    def outer_get(self, row_index: int, col_index: int) -> T_co:
        row_index = self.row_window[row_index]
        col_index = self.col_index + col_index
        return self.target.outer_get(row_index, col_index)


@final
class RowColSlice(OuterMesh[int, int, T_co], Generic[T_co]):

    __slots__ = ("target", "row_window", "col_window")

    target: Mesh[Any, Any, T_co]
    row_window: range
    col_window: range

    def __init__(self, target: Mesh[Any, Any, T_co], row_window: range, col_window: range) -> None:
        self.target = target
        self.row_window = row_window
        self.col_window = col_window

    @override
    def __repr__(self) -> str:
        return f"RowColSlice(target={self.target!r}, row_window={self.row_window!r}, col_window={self.col_window!r})"

    @property
    @override
    def nrows(self) -> int:
        return len(self.row_window)

    @property
    @override
    def ncols(self) -> int:
        return len(self.col_window)

    @override
    def outer_get(self, row_index: int, col_index: int) -> T_co:
        row_index = self.row_window[row_index]
        col_index = self.col_window[col_index]
        return self.target.outer_get(row_index, col_index)
