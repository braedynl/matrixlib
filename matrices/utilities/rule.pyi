from enum import Enum
from typing import Literal, final, overload

__all__ = ["Rule", "ROW", "COL"]


@final
class Rule(Enum):

    ROW: Literal[0]
    COL: Literal[1]

    @property
    def handle(self) -> Literal["row", "column"]: ...

    @overload
    def invert(self: Literal[Rule.ROW]) -> Literal[Rule.COL]: ...  # type: ignore[misc]
    @overload
    def invert(self: Literal[Rule.COL]) -> Literal[Rule.ROW]: ...  # type: ignore[misc]
    @overload
    def invert(self) -> Literal[Rule.ROW, Rule.COL]: ...


ROW: Literal[Rule.ROW]
COL: Literal[Rule.COL]
