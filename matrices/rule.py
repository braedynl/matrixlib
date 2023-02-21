from __future__ import annotations

from enum import Enum
from typing import Literal, final, overload

__all__ = ["Rule", "ROW", "COL"]


@final
class Rule(Enum):
    """The direction by which to operate within a matrix

    Rule values are usable as an index that retrieves the rule's corresponding
    dimension from a matrix's shape (or any two-element sequence type).
    """

    ROW: Literal[0] = 0  #: Retrieves the item at the zeroth index
    COL: Literal[1] = 1  #: Retrieves the item at the first index

    @overload
    def __invert__(self: Literal[Rule.ROW]) -> Literal[Rule.COL]: ...  # type: ignore[misc]
    @overload
    def __invert__(self: Literal[Rule.COL]) -> Literal[Rule.ROW]: ...  # type: ignore[misc]
    @overload
    def __invert__(self) -> Rule: ...

    def __invert__(self):
        """Return the rule corresponding to the opposite dimension"""
        return Rule(not self.value)

    @property
    def handle(self) -> Literal["row", "column"]:
        """The rule's non-Pythonized name"""
        return ("row", "column")[self.value]


ROW: Literal[Rule.ROW] = Rule.ROW
COL: Literal[Rule.COL] = Rule.COL
