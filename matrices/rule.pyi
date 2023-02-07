from enum import Enum
from typing import Literal, final, overload

__all__ = ["Rule", "ROW", "COL"]


@final
class Rule(Enum):

    ROW: Literal[0]
    COL: Literal[1]

    @overload
    def __invert__(self: Literal[Rule.ROW]) -> Literal[Rule.COL]: ...  # type: ignore[misc]
    @overload
    def __invert__(self: Literal[Rule.COL]) -> Literal[Rule.ROW]: ...  # type: ignore[misc]
    @overload
    def __invert__(self) -> Rule: ...

    @property
    def handle(self) -> Literal["row", "column"]: ...


ROW: Literal[Rule.ROW]
COL: Literal[Rule.COL]
