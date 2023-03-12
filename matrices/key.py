from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, Literal, TypeVar, Union, overload

from .rule import Rule

__all__ = ["Key"]

RowKeyT = TypeVar("RowKeyT", bound=Union[slice, int])
ColKeyT = TypeVar("ColKeyT", bound=Union[slice, int])


class Key(Generic[RowKeyT, ColKeyT]):

    __slots__ = __match_args__ = ("row_key", "col_key")

    def __init__(self) -> None:
        self.row_key: RowKeyT
        self.col_key: ColKeyT

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, Key):
            return (
                self.row_key == other.row_key
                and
                self.col_key == other.col_key
            )
        return NotImplemented

    def __len__(self) -> Literal[2]:
        return 2

    @overload
    def __getitem__(self, key: Literal[Rule.ROW]) -> RowKeyT: ...
    @overload
    def __getitem__(self, key: Literal[Rule.COL]) -> ColKeyT: ...
    @overload
    def __getitem__(self, key: Rule) -> Union[RowKeyT, ColKeyT]: ...

    def __getitem__(self, key):
        if key is Rule.ROW:
            return self.row_key
        else:
            return self.col_key

    @overload
    def __setitem__(self, key: Literal[Rule.ROW], value: RowKeyT) -> None: ...
    @overload
    def __setitem__(self, key: Literal[Rule.COL], value: ColKeyT) -> None: ...
    @overload
    def __setitem__(self, key: Rule, value: Union[RowKeyT, ColKeyT]) -> None: ...

    def __setitem__(self, key, value):
        if key is Rule.ROW:
            self.row_key = value
        else:
            self.col_key = value

    def __iter__(self) -> Iterator[Union[RowKeyT, ColKeyT]]:
        yield self.row_key
        yield self.col_key
