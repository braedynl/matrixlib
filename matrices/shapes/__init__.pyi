from collections.abc import Iterator, MutableSequence
from typing import (Any, Literal, Optional, SupportsIndex, TypeVar, Union,
                    overload)

from ..utilities import Rule
from .abc import *

__all__ = ["ShapeLike", "Shape"]

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)

ShapeT = TypeVar("ShapeT", bound=Shape)


class Shape(ShapeLike[M, N]):

    __slots__: tuple[Literal["_array"]]

    def __init__(self, nrows: M, ncols: N) -> None: ...
    def __repr__(self) -> str: ...
    @overload
    def __getitem__(self, key: Literal[0]) -> M: ...
    @overload
    def __getitem__(self, key: Literal[1]) -> N: ...
    @overload
    def __getitem__(self, key: SupportsIndex) -> Union[M, N]: ...
    @overload
    def __setitem__(self, key: Literal[0], value: M) -> None: ...
    @overload
    def __setitem__(self, key: Literal[1], value: N) -> None: ...
    @overload
    def __setitem__(self, key: SupportsIndex, value: Union[M, N]) -> None: ...
    def __iter__(self) -> Iterator[Union[M, N]]: ...
    def __reversed__(self) -> Iterator[Union[M, N]]: ...
    def __contains__(self, value: Any) -> bool: ...
    def __deepcopy__(self: ShapeT, memo: Optional[dict[int, Any]] = None) -> ShapeT: ...
    def __copy__(self: ShapeT) -> ShapeT: ...

    @classmethod
    def wrap(cls: type[ShapeT], array: MutableSequence[int]) -> ShapeT: ...

    @property
    def nrows(self) -> M: ...
    @nrows.setter
    def nrows(self, value: M) -> None: ...
    @property
    def ncols(self) -> N: ...
    @ncols.setter
    def ncols(self, value: N) -> None: ...

    def copy(self: ShapeT) -> ShapeT: ...
    @overload
    def subshape(self, *, by: Literal[Rule.ROW]) -> Shape[Literal[1], N]: ...  # type: ignore[misc]
    @overload
    def subshape(self, *, by: Literal[Rule.COL]) -> Shape[M, Literal[1]]: ...  # type: ignore[misc]
    @overload
    def subshape(self, *, by: Rule) -> Shape[int, int]: ...
    @overload
    def subshape(self) -> Shape[Literal[1], N]: ...
    def sequence(self, index: int, *, by: Rule = Rule.ROW) -> tuple[int, int, int]: ...
    def range(self, index: int, *, by: Rule = Rule.ROW) -> range: ...
    def slice(self, index: int, *, by: Rule = Rule.ROW) -> slice: ...
