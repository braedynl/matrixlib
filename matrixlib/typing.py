from typing import Literal

from typing_extensions import TypeAlias

__all__ = ["EvenNumber", "OddNumber"]

EvenNumber: TypeAlias = Literal[-16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16]
OddNumber: TypeAlias = Literal[-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
