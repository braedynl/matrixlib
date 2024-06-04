from __future__ import annotations

__all__ = ["Rule", "ROW", "COL"]

from enum import Enum
from typing import Final, Literal, final, overload


@final
class Rule(Enum):
    """Enum used to dictate row or column-wise interpretation

    Each member maps to an index that will retrieve its corresponding dimension
    from a matrix shape (or any two-element sequence).

    ``Rule`` should not be sub-classed, as many functions make the assumption
    that the class has just two members with values ``0`` and ``1``.
    """

    ROW = 0  #: Maps to literal ``0``
    COL = 1  #: Maps to literal ``1``

    @overload
    def __invert__(self: Literal[Rule.ROW]) -> Literal[Rule.COL]: ...  # type: ignore
    @overload
    def __invert__(self: Literal[Rule.COL]) -> Literal[Rule.ROW]: ...  # type: ignore
    @overload
    def __invert__(self) -> Rule: ...

    def __invert__(self) -> Rule:
        """Return the rule corresponding to the opposite dimension"""
        return Rule(not self.value)

    def __index__(self) -> Literal[0, 1]:
        """Return the rule's value"""
        return self.value  # type: ignore

    @property
    def handle(self) -> Literal["row", "column"]:
        """The rule's un-Pythonized name"""
        if self is Rule.ROW:
            return "row"
        else:
            return "column"


ROW: Final[Literal[Rule.ROW]] = Rule.ROW  #: Equivalent to ``Rule.ROW``
COL: Final[Literal[Rule.COL]] = Rule.COL  #: Equivalent to ``Rule.COL``
