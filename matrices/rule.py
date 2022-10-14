from enum import IntEnum

__all__ = [
    "Rule",
    "ROW",
    "COL",
]


class Rule(IntEnum):
    """The direction by which to operate within a matrix

    Rule members are usable as an index that retrieves the rule's corresponding
    dimension from a matrix's shape (or any two-element sequence type).
    """

    ROW = 0
    COL = 1

    @property
    def inverse(self):
        """The rule corresponding to the opposite dimension"""
        return Rule(not self)

    @property
    def true_name(self):
        """The rule's "true", unformatted name"""
        return ["row", "column"][self]


ROW = Rule.ROW
COL = Rule.COL
