from collections.abc import Iterator
from typing import (Any, Literal, Optional, SupportsIndex, TypeVar, final,
                    overload)

from .core import Rule, ShapeLike

__all__ = ["Shape", "ShapeView"]

NRowsT = TypeVar("NRowsT", bound=int)
NColsT = TypeVar("NColsT", bound=int)
NRowsT_co = TypeVar("NRowsT_co", covariant=True, bound=int)
NColsT_co = TypeVar("NColsT_co", covariant=True, bound=int)


@final
class Shape(ShapeLike[NRowsT, NColsT]):

    __slots__: tuple[Literal["_data"]]

    def __init__(self, nrows: NRowsT, ncols: NColsT) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @overload
    def __getitem__(self, key: Literal[0]) -> NRowsT: ...
    @overload
    def __getitem__(self, key: Literal[1]) -> NColsT: ...
    @overload
    def __getitem__(self, key: SupportsIndex) -> NRowsT | NColsT: ...
    @overload
    def __setitem__(self, key: Literal[0], value: NRowsT) -> None: ...
    @overload
    def __setitem__(self, key: Literal[1], value: NColsT) -> None: ...
    @overload
    def __setitem__(self, key: SupportsIndex, value: NRowsT | NColsT) -> None: ...
    def __iter__(self) -> Iterator[NRowsT | NColsT]: ...
    def __reversed__(self) -> Iterator[NRowsT | NColsT]: ...
    def __contains__(self, value: Any) -> bool: ...
    def __deepcopy__(self, memo: Optional[dict[int, Any]] = None) -> Shape[NRowsT, NColsT]: ...
    def __copy__(self) -> Shape[NRowsT, NColsT]: ...

    @property
    def nrows(self) -> NRowsT: ...
    @nrows.setter
    def nrows(self, value: NRowsT) -> None: ...
    @property
    def ncols(self) -> NColsT: ...
    @ncols.setter
    def ncols(self, value: NColsT) -> None: ...

    def reverse(self) -> Shape[NColsT, NRowsT]: ...
    def copy(self) -> Shape[NRowsT, NColsT]: ...
    @overload
    def subshape(self, *, by: Literal[Rule.ROW]) -> Shape[Literal[1], NColsT]: ...  # type: ignore[misc]
    @overload
    def subshape(self, *, by: Literal[Rule.COL]) -> Shape[NRowsT, Literal[1]]: ...  # type: ignore[misc]
    @overload
    def subshape(self, *, by: Rule) -> Shape[int, int]: ...
    @overload
    def subshape(self) -> Shape[Literal[1], NColsT]: ...
    def resolve_index(self, key: SupportsIndex, *, by: Rule = Rule.ROW) -> int: ...
    def resolve_slice(self, key: slice, *, by: Rule = Rule.ROW) -> range: ...
    def sequence(self, index: int, *, by: Rule = Rule.ROW) -> tuple[int, int, int]: ...
    def range(self, index: int, *, by: Rule = Rule.ROW) -> range: ...
    def slice(self, index: int, *, by: Rule = Rule.ROW) -> slice: ...


@final
class ShapeView(ShapeLike[NRowsT_co, NColsT_co]):

    __slots__: tuple[Literal["_target"]]

    def __init__(self, target: ShapeLike[NRowsT_co, NColsT_co]) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @overload
    def __getitem__(self, key: Literal[0]) -> NRowsT_co: ...
    @overload
    def __getitem__(self, key: Literal[1]) -> NColsT_co: ...
    @overload
    def __getitem__(self, key: SupportsIndex) -> NRowsT_co | NColsT_co: ...
    def __iter__(self) -> Iterator[NRowsT_co | NColsT_co]: ...
    def __reversed__(self) -> Iterator[NRowsT_co | NColsT_co]: ...
    def __contains__(self, value: Any) -> bool: ...
    def __deepcopy__(self, memo: Optional[dict[int, Any]] = None) -> ShapeView[NRowsT_co, NColsT_co]: ...
    def __copy__(self) -> ShapeView[NRowsT_co, NColsT_co]: ...