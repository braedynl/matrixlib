from enum import IntEnum
from typing import Literal, TypeVar

__all__ = ["Rule", "ROW", "COL"]


RuleT = TypeVar("RuleT", bound="Rule")

class Rule(IntEnum):
    """The direction by which to operate within a matrix

    Rule members are usable as an index that retrieves the rule's corresponding
    dimension from a matrix's shape (or any two-element sequence type).
    """

    ROW: Literal[0] = 0
    COL: Literal[1] = 1

    @property
    def inverse(self: RuleT) -> RuleT:
        """The rule corresponding to the opposite dimension"""
        return self.__class__(not self)

    @property
    def handle(self) -> str:
        """The rule's non-Pythonized name"""
        return ("row", "column")[self]


ROW: Literal[Rule.ROW] = Rule.ROW
COL: Literal[Rule.COL] = Rule.COL
