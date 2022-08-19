import operator
from fractions import Fraction
from numbers import Real
from typing import NamedTuple

from matrixlib import Matrix, Rule, Shape


class LUFactorizationResult(NamedTuple):
    lu: Matrix[Real]
    p: list[int]


def lu_factor(a: Matrix[Real]) -> LUFactorizationResult:
    m, n = a.shape
    if m != n:
        raise ValueError("cannot factorize non-square matrix")

    lu = a.copy()
    p = list(range(n))

    for i in range(n):
        j, x = max(((j, abs(lu[j, i])) for j in range(i, n)), key=operator.itemgetter(1))
        if not x:
            raise ValueError("cannot factorize singular matrix")

        if i != j:
            p[i], p[j] = p[j], p[i]
            lu.swap(i, j, by=Rule.ROW)

        for j in range(i + 1, n):
            lu[j, i] /= lu[i, i]

            for k in range(i + 1, n):
                lu[j, k] -= lu[j, i] * lu[i, k]

    return LUFactorizationResult(lu, p)


def lu_solve(res: LUFactorizationResult, b: Matrix[Real]) -> Matrix[Real]:
    lu, p = res
    n = lu.ncols

    if b.shape != (n, 1):
        raise ValueError(f"matrix b must have shape {n} × 1")

    data: list[Real] = []

    for i in range(n):
        data.append(b[p[i]])
        for k in range(i):
            data[i] -= lu[i, k] * data[k]

    for i in reversed(range(n)):
        for k in range(i + 1, n):
            data[i] -= lu[i, k] * data[k]
        data[i] /= lu[i, i]

    x = Matrix.new(data, Shape(n, 1))

    return x


a = Matrix([
    2, 5, 8, 7,
    5, 2, 2, 8,
    7, 5, 6, 6,
    5, 4, 4, 8,
], nrows=4, ncols=4).apply(Fraction)

bx = [
    Matrix([   1,  1,   1,  1], nrows=4, ncols=1),
    Matrix([   1,  2,   2,  0], nrows=4, ncols=1),
    Matrix([  -1, -2,  -3, -4], nrows=4, ncols=1),
    Matrix([-100,  1, 100,  5], nrows=4, ncols=1),
    Matrix([   0,  0,   0,  0], nrows=4, ncols=1),
]

res = lu_factor(a)
for b in bx:
    x = lu_solve(res, b)
    assert all(a @ x == b)
