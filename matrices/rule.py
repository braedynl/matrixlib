from enum import Enum

__all__ = ["Rule", "ROW", "COL"]


class Rule(Enum):
    """The direction by which to operate within a matrix

    Rule values are usable as an index that retrieves the rule's corresponding
    dimension from a matrix's shape (or any two-element sequence type).
    """

    ROW = 0  #: Retrieves the item at the zeroth index
    COL = 1  #: Retrieves the item at the first index

    def __invert__(self):
        """Return the rule corresponding to the opposite dimension"""
        return Rule(not self.value)

    @property
    def handle(self):
        """The rule's non-Pythonized name"""
        return ("row", "column")[self.value]


ROW = Rule.ROW
COL = Rule.COL
