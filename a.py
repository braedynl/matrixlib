from matrixlib import Matrix, Rule
from typing import Literal

a = Matrix[Literal[4], Literal[3], int](
    array=(
         1,  2,  3,
         4,  5,  6,
         7,  8,  9,
        10, 11, 12,
    ),
    shape=(4, 3),
)

for thing in a.slices(by=Rule.COL):
    print(thing.array)
