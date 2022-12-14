from enum import Enum

__all__ = ["Rule", "ROW", "COL"]


class Rule(Enum):
    """The direction by which to operate within a matrix

    Rule values are usable as an index that retrieves the rule's corresponding
    dimension from a matrix's shape (or any two-element sequence type).
    """

    ROW = 0
    COL = 1

    def handle(self):
        """Return the rule's non-Pythonized name"""
        return ("row", "column")[self.value]

    def invert(self):
        """Return the rule corresponding to the opposite dimension"""
        return Rule(not self.value)


ROW = Rule.ROW
COL = Rule.COL
