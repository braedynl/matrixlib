from enum import Enum
from typing import Literal, final, overload

__all__ = ["Rule", "ROW", "COL"]


@final
class Rule(Enum):

    ROW: Literal[0]
    COL: Literal[1]

    @overload
    def label(self: Literal[Rule.ROW]) -> Literal["row"]: ...  # type: ignore[misc]
    @overload
    def label(self: Literal[Rule.COL]) -> Literal["column"]: ...  # type: ignore[misc]
    @overload
    def label(self) -> str: ...
    @overload
    def invert(self: Literal[Rule.ROW]) -> Literal[Rule.COL]: ...  # type: ignore[misc]
    @overload
    def invert(self: Literal[Rule.COL]) -> Literal[Rule.ROW]: ...  # type: ignore[misc]
    @overload
    def invert(self) -> Rule: ...


ROW: Literal[Rule.ROW]
COL: Literal[Rule.COL]
