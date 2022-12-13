from __future__ import annotations

from enum import IntEnum
from typing import Literal, final

__all__ = ["Rule", "ROW", "COL"]


@final
class Rule(IntEnum):
    """The direction by which to operate within a matrix

    Rule members are usable as an index that retrieves the rule's corresponding
    dimension from a matrix's shape (or any two-element sequence type).
    """

    ROW: Literal[0] = 0
    COL: Literal[1] = 1

    @property
    def inverse(self) -> Rule:
        """The rule corresponding to the opposite dimension"""
        return Rule(not self)

    @property
    def handle(self) -> str:
        """The rule's non-Pythonized name"""
        return ("row", "column")[self]


ROW: Literal[Rule.ROW] = Rule.ROW
COL: Literal[Rule.COL] = Rule.COL
