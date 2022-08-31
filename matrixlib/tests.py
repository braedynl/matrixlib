import itertools
from unittest import TestCase

from . import Matrix, Rule

M0x0 = Matrix([], nrows=0, ncols=0)

M0x1 = Matrix([], nrows=0, ncols=1)
M1x0 = Matrix([], nrows=1, ncols=0)

M0x2 = Matrix([], nrows=0, ncols=2)
M2x0 = Matrix([], nrows=2, ncols=0)

M1x1 = Matrix([
    0
], nrows=1, ncols=1)

M1x2 = Matrix([
    0, 1,
], nrows=1, ncols=2)

M2x1 = Matrix([
    0,
    1,
], nrows=2, ncols=1)

M2x2 = Matrix([
    0, 1,
    2, 3,
], nrows=2, ncols=2)

M2x3 = Matrix([
    0, 1, 2,
    3, 4, 5,
], nrows=2, ncols=3)

M3x2 = Matrix([
    0, 1,
    2, 3,
    4, 5,
], nrows=3, ncols=2)

M3x3 = Matrix([
    0, 1, 2,
    3, 4, 5,
    6, 7, 8,
], nrows=3, ncols=3)


def eq(a: Matrix, b: Matrix) -> bool:
    """Return true if the matrices have matching shapes and elements, otherwise
    false
    """
    return a.shape == b.shape and all(a == b)


class TestMatrix(TestCase):
    """Simple tests for methods of the `Matrix` class

    These tests are not exhaustive, both in terms of coverage across methods
    and for the tested methods themselves.

    Most methods are wrappers around built-ins and/or call one another. These
    tests are for methods that are a bit more "involved" or are heavily relied
    on for building other methods (e.g., most operator overloads call
    `flat_map()`, and so `flat_map()` is scrutinized).
    """

    def testGetItem(self) -> None:
        """Tests for `Matrix.__getitem__()`"""

        for (i, j), x in zip(
            itertools.product(range(2), range(3)),
            range(6),
        ):
            y = M2x3[i, j]
            self.assertEqual(x, y)

        with self.assertRaises(IndexError):
            M2x3[2, 0]

        with self.assertRaises(IndexError):
            M2x3[0, 3]

        with self.assertRaises(IndexError):
            M2x3[2, 3]

        res = M2x3[0, :]
        exp = Matrix([
            0, 1, 2,
        ], nrows=1, ncols=3)

        self.assertTrue(eq(res, exp))

        res = M2x3[1, :]
        exp = Matrix([
            3, 4, 5,
        ], nrows=1, ncols=3)

        self.assertTrue(eq(res, exp))

        with self.assertRaises(IndexError):
            M2x3[2, :]

        res = M2x3[:, 0]
        exp = Matrix([
            0,
            3,
        ], nrows=2, ncols=1)

        self.assertTrue(eq(res, exp))

        res = M2x3[:, 1]
        exp = Matrix([
            1,
            4,
        ], nrows=2, ncols=1)

        self.assertTrue(eq(res, exp))

        res = M2x3[:, 2]
        exp = Matrix([
            2,
            5,
        ], nrows=2, ncols=1)

        self.assertTrue(eq(res, exp))

        with self.assertRaises(IndexError):
            M2x3[:, 3]

        res = M2x3[:, :]
        exp = M2x3

        self.assertTrue(eq(res, exp))

        res = M3x3[1:, :]
        exp = Matrix([
            3, 4, 5,
            6, 7, 8,
        ], nrows=2, ncols=3)

        self.assertTrue(eq(res, exp))

        res = M3x3[:, 1:]
        exp = Matrix([
            1, 2,
            4, 5,
            7, 8,
        ], nrows=3, ncols=2)

        self.assertTrue(eq(res, exp))

        res = M3x3[1:, 1:]
        exp = Matrix([
            4, 5,
            7, 8,
        ], nrows=2, ncols=2)

        self.assertTrue(eq(res, exp))

    def testSetItem(self) -> None:
        """Tests for `Matrix.__setitem__()`"""

        res = M2x3.copy()

        for (i, j), x in zip(
            itertools.product(range(2), range(3)),
            range(-1, -6 - 1, -1),
        ):
            res[i, j] = x
            y = res[i, j]
            self.assertEqual(x, y)

        with self.assertRaises(IndexError):
            res[2, 0] = -1

        with self.assertRaises(IndexError):
            res[0, 3] = -1

        with self.assertRaises(IndexError):
            res[2, 3] = -1

        res = M2x3.copy()

        res[0, :] = (-1, -2, -3)
        exp = Matrix([
            -1, -2, -3,
             3,  4,  5,
        ], nrows=2, ncols=3)

        self.assertTrue(eq(res, exp))

        res[1, :] = (-4, -5, -6)
        exp = Matrix([
            -1, -2, -3,
            -4, -5, -6,
        ], nrows=2, ncols=3)

        self.assertTrue(eq(res, exp))

        with self.assertRaises(IndexError):
            res[2, :] = (-7, -8, -9)

        with self.assertRaises(ValueError):
            res[0, :] = (-1, -2, -3, -4)

        res = M2x3.copy()

        res[:, 0] = (
            -1,
            -2,
        )
        exp = Matrix([
            -1, 1, 2,
            -2, 4, 5,
        ], nrows=2, ncols=3)

        self.assertTrue(eq(res, exp))

        res[:, 1] = (
            -3,
            -4,
        )
        exp = Matrix([
            -1, -3, 2,
            -2, -4, 5,
        ], nrows=2, ncols=3)

        self.assertTrue(eq(res, exp))

        res[:, 2] = (
            -5,
            -6,
        )
        exp = Matrix([
            -1, -3, -5,
            -2, -4, -6,
        ], nrows=2, ncols=3)

        self.assertTrue(eq(res, exp))

        with self.assertRaises(IndexError):
            res[:, 3] = (
                -7,
                -8,
            )

        with self.assertRaises(ValueError):
            res[:, 0] = (
                -1,
                -2,
                -3,
            )

        res = M3x3.copy()

        res[:, :] = (
            -1, -2, -3,
            -4, -5, -6,
            -7, -8, -9,
        )
        exp = Matrix([
            -1, -2, -3,
            -4, -5, -6,
            -7, -8, -9,
        ], nrows=3, ncols=3)

        self.assertTrue(eq(res, exp))

        res = M3x3.copy()

        res[1:, :] = (
            -4, -5, -6,
            -7, -8, -9,
        )
        exp = Matrix([
             0,  1,  2,
            -4, -5, -6,
            -7, -8, -9,
        ], nrows=3, ncols=3)

        self.assertTrue(eq(res, exp))

        res = M3x3.copy()

        res[:, 1:] = (
            -1, -2,
            -3, -4,
            -5, -6,
        )
        exp = Matrix([
            0, -1, -2,
            3, -3, -4,
            6, -5, -6,
        ], nrows=3, ncols=3)

        self.assertTrue(eq(res, exp))

        res = M3x3.copy()

        res[1:, 1:] = (
            -1, -2,
            -3, -4,
        )
        exp = Matrix([
            0,  1,  2,
            3, -1, -2,
            6, -3, -4,
        ], nrows=3, ncols=3)

        self.assertTrue(eq(res, exp))

    def testReshape(self) -> None:
        """Tests for `Matrix.reshape()`"""

        res = M0x0.copy().reshape(0, 3)
        exp = Matrix([], nrows=0, ncols=3)

        self.assertTrue(eq(res, exp))

        res = M0x0.copy().reshape(3, 0)
        exp = Matrix([], nrows=3, ncols=0)

        self.assertTrue(eq(res, exp))

        res = M2x3.copy().reshape(3, 2)
        exp = Matrix([
            0, 1,
            2, 3,
            4, 5,
        ], nrows=3, ncols=2)

        self.assertTrue(eq(res, exp))

        with self.assertRaises(ValueError):
            M2x3.copy().reshape(3, 4)

    def testFlatMap(self) -> None:
        """Tests for `Matrix.flat_map()`"""

        res = M0x0.copy().flat_map(lambda x: x + 1)
        exp = Matrix([], nrows=0, ncols=0)

        self.assertTrue(eq(res, exp))

        res = M2x2.copy().flat_map(lambda x: x + 1)
        exp = Matrix([
            1, 2,
            3, 4,
        ], nrows=2, ncols=2)

        self.assertTrue(eq(res, exp))

        res = M2x2.copy().flat_map(lambda x, y: x + y, M2x2)
        exp = Matrix([
            0 * 2, 1 * 2,
            2 * 2, 3 * 2,
        ], nrows=2, ncols=2)

        self.assertTrue(eq(res, exp))

        res = M2x2.copy().flat_map(lambda x, y, z: x + y + z, M2x2, M2x2)
        exp = Matrix([
            0 * 3, 1 * 3,
            2 * 3, 3 * 3,
        ], nrows=2, ncols=2)

        self.assertTrue(eq(res, exp))

        res = M2x2.copy().flat_map(lambda x, y: x + y, 1)
        exp = Matrix([
            0 + 1, 1 + 1,
            2 + 1, 3 + 1,
        ], nrows=2, ncols=2)

        self.assertTrue(eq(res, exp))

        res = M2x2.copy().flat_map(lambda x, y: x + y, range(M2x2.size))
        exp = Matrix([
            0 * 2, 1 * 2,
            2 * 2, 3 * 2,
        ], nrows=2, ncols=2)

        self.assertTrue(eq(res, exp))

        with self.assertRaises(ValueError):
            M2x2.copy().flat_map(lambda x, y: x + y, M0x0)

        with self.assertRaises(ValueError):
            M2x2.copy().flat_map(lambda x, y, z: x + y + z, 1, M0x0)

    def testMap(self) -> None:
        """Tests for `Matrix.map()`"""

        res1 = M0x0.copy().map(lambda row: row.reverse(), by=Rule.ROW)
        res2 = M0x0.copy().map(lambda col: col.reverse(), by=Rule.COL)
        exp = Matrix([], nrows=0, ncols=0)

        self.assertTrue(eq(res1, exp))
        self.assertTrue(eq(res2, exp))

        res = M2x2.copy().map(lambda row: row.reverse(), by=Rule.ROW)
        exp = Matrix([
            1, 0,
            3, 2,
        ], nrows=2, ncols=2)

        self.assertTrue(eq(res, exp))

        res = M2x2.copy().map(lambda col: col.reverse(), by=Rule.COL)
        exp = Matrix([
            2, 3,
            0, 1,
        ], nrows=2, ncols=2)

        self.assertTrue(eq(res, exp))

        res1 = M2x2.copy().map(lambda row1, row2: row1 + row2, M2x2, by=Rule.ROW)
        res2 = M2x2.copy().map(lambda col1, col2: col1 + col2, M2x2, by=Rule.COL)
        exp = Matrix([
            0 * 2, 1 * 2,
            2 * 2, 3 * 2,
        ], nrows=2, ncols=2)

        self.assertTrue(eq(res1, exp))
        self.assertTrue(eq(res2, exp))

        res1 = M2x2.copy().map(lambda row, x: row + x, 1, by=Rule.ROW)
        res2 = M2x2.copy().map(lambda col, x: col + x, 1, by=Rule.COL)
        exp = Matrix([
            0 + 1, 1 + 1,
            2 + 1, 3 + 1,
        ], nrows=2, ncols=2)

        self.assertTrue(eq(res1, exp))
        self.assertTrue(eq(res2, exp))

        with self.assertRaises(ValueError):
            M2x3.copy().map(lambda row: [0, 1], by=Rule.ROW)

        with self.assertRaises(ValueError):
            M2x3.copy().map(lambda col: [0, 1, 2], by=Rule.COL)

    def testSlices(self) -> None:
        """Tests for `Matrix.slices()`"""

        it = M0x0.slices(by=Rule.ROW)

        with self.assertRaises(StopIteration):
            next(it)

        it = M0x0.slices(by=Rule.ROW)

        with self.assertRaises(StopIteration):
            next(it)

        it = M0x1.slices(by=Rule.ROW)

        with self.assertRaises(StopIteration):
            next(it)

        it = M0x1.slices(by=Rule.COL)

        res = next(it)
        exp = Matrix([], nrows=0, ncols=1)

        self.assertTrue(eq(res, exp))

        with self.assertRaises(StopIteration):
            next(it)

        it = M2x3.slices(by=Rule.ROW)

        res = next(it)
        exp = Matrix([
            0, 1, 2,
        ], nrows=1, ncols=3)

        self.assertTrue(eq(res, exp))

        res = next(it)
        exp = Matrix([
            3, 4, 5,
        ], nrows=1, ncols=3)

        self.assertTrue(eq(res, exp))

        with self.assertRaises(StopIteration):
            next(it)

        it = M2x3.slices(by=Rule.COL)

        res = next(it)
        exp = Matrix([
            0,
            3,
        ], nrows=2, ncols=1)

        self.assertTrue(eq(res, exp))

        res = next(it)
        exp = Matrix([
            1,
            4,
        ], nrows=2, ncols=1)

        self.assertTrue(eq(res, exp))

        res = next(it)
        exp = Matrix([
            2,
            5,
        ], nrows=2, ncols=1)

        self.assertTrue(eq(res, exp))

        with self.assertRaises(StopIteration):
            next(it)

    def testTranspose(self) -> None:
        """Tests for `Matrix.transpose()`"""

        res = M0x0.copy().transpose()
        exp = Matrix([], nrows=0, ncols=0)

        self.assertTrue(eq(res, exp))

        res = M0x1.copy().transpose()
        exp = Matrix([], nrows=1, ncols=0)

        self.assertTrue(eq(res, exp))

        res = M1x0.copy().transpose()
        exp = Matrix([], nrows=0, ncols=1)

        self.assertTrue(eq(res, exp))

        res = M2x2.copy().transpose()
        exp = Matrix([
            0, 2,
            1, 3,
        ], nrows=2, ncols=2)

        self.assertTrue(eq(res, exp))

        res = M2x3.copy().transpose()
        exp = Matrix([
            0, 3,
            1, 4,
            2, 5,
        ], nrows=3, ncols=2)

        self.assertTrue(eq(res, exp))

        res = M3x2.copy().transpose()
        exp = Matrix([
            0, 2, 4,
            1, 3, 5,
        ], nrows=2, ncols=3)

        self.assertTrue(eq(res, exp))

    def testCollapse(self) -> None:
        """Tests for `Matrix.collapse()`"""

        res = M0x2.copy().collapse(sum, by=Rule.ROW)
        exp = Matrix([], nrows=0, ncols=1)

        self.assertTrue(eq(res, exp))

        res = M0x2.copy().collapse(sum, by=Rule.COL)
        exp = Matrix([
            0, 0,
        ], nrows=1, ncols=2)

        self.assertTrue(eq(res, exp))

        res = M2x3.copy().collapse(max, by=Rule.ROW)
        exp = Matrix([
            2,
            5,
        ], nrows=2, ncols=1)

        self.assertTrue(eq(res, exp))

        res = M2x3.copy().collapse(max, by=Rule.COL)
        exp = Matrix([
            3, 4, 5,
        ], nrows=1, ncols=3)

        self.assertTrue(eq(res, exp))

    def testSwap(self) -> None:
        """Tests for `Matrix.swap()`"""

        res = M2x0.copy().swap(0, 1, by=Rule.ROW)
        exp = Matrix([], nrows=2, ncols=0)

        self.assertTrue(eq(res, exp))

        with self.assertRaises(IndexError):
            M2x0.copy().swap(0, 1, by=Rule.COL)

        res = M2x3.copy().swap(0, 1, by=Rule.ROW)
        exp = Matrix([
            3, 4, 5,
            0, 1, 2,
        ], nrows=2, ncols=3)

        self.assertTrue(eq(res, exp))

        res = M2x3.copy().swap(0, -1, by=Rule.COL)
        exp = Matrix([
            2, 1, 0,
            5, 4, 3,
        ], nrows=2, ncols=3)

        self.assertTrue(eq(res, exp))

        with self.assertRaises(IndexError):
            M2x3.copy().swap(0, 2, by=Rule.ROW)

        with self.assertRaises(IndexError):
            M2x3.copy().swap(0, 3, by=Rule.COL)

    def testFlip(self) -> None:
        """Tests for `Matrix.flip()`"""

        res = M0x2.copy().flip(by=Rule.ROW)
        exp = Matrix([], nrows=0, ncols=2)

        self.assertTrue(eq(res, exp))

        res = M0x2.copy().flip(by=Rule.COL)
        exp = Matrix([], nrows=0, ncols=2)

        self.assertTrue(eq(res, exp))

        res = M2x3.copy().flip(by=Rule.ROW)
        exp = Matrix([
            3, 4, 5,
            0, 1, 2,
        ], nrows=2, ncols=3)

        self.assertTrue(eq(res, exp))

        res = M3x2.copy().flip(by=Rule.COL)
        exp = Matrix([
            1, 0,
            3, 2,
            5, 4,
        ], nrows=3, ncols=2)

        self.assertTrue(eq(res, exp))

        res = M3x3.copy().flip(by=Rule.ROW)
        exp = Matrix([
            6, 7, 8,
            3, 4, 5,
            0, 1, 2,
        ], nrows=3, ncols=3)

        self.assertTrue(eq(res, exp))

        res = M3x3.copy().flip(by=Rule.COL)
        exp = Matrix([
            2, 1, 0,
            5, 4, 3,
            8, 7, 6,
        ], nrows=3, ncols=3)

        self.assertTrue(eq(res, exp))

    def testStack(self) -> None:
        """Tests for `Matrix.stack()`"""

        res = M2x0.copy().stack([], by=Rule.ROW)
        exp = Matrix([], nrows=3, ncols=0)

        self.assertTrue(eq(res, exp))

        with self.assertRaises(ValueError):
            M2x0.copy().stack([], by=Rule.COL)

        res = M0x2.copy().stack([], by=Rule.COL)
        exp = Matrix([], nrows=0, ncols=3)

        self.assertTrue(eq(res, exp))

        with self.assertRaises(ValueError):
            M0x2.copy().stack([], by=Rule.ROW)

        res = M2x3.copy().stack([6, 7, 8], by=Rule.ROW)
        exp = Matrix([
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
        ], nrows=3, ncols=3)

        self.assertTrue(eq(res, exp))

        res = M2x3.copy().stack(
            Matrix([
                6,  7,  8,
                9, 10, 11,
            ], nrows=2, ncols=3),
            by=Rule.ROW,
        )
        exp = Matrix([
            0,  1,  2,
            3,  4,  5,
            6,  7,  8,
            9, 10, 11,
        ], nrows=4, ncols=3)

        self.assertTrue(eq(res, exp))

        res = M3x2.copy().stack([-1, -2, -3], by=Rule.COL)
        exp = Matrix([
            0, 1, -1,
            2, 3, -2,
            4, 5, -3,
        ], nrows=3, ncols=3)

        self.assertTrue(eq(res, exp))

        res = M3x2.copy().stack(
            Matrix([
                -1, -2,
                -3, -4,
                -5, -6,
            ], nrows=3, ncols=2),
            by=Rule.COL,
        )
        exp = Matrix([
            0, 1, -1, -2,
            2, 3, -3, -4,
            4, 5, -5, -6,
        ], nrows=3, ncols=4)

        self.assertTrue(eq(res, exp))

        with self.assertRaises(ValueError):
            M3x3.copy().stack(M3x2, by=Rule.ROW)

        with self.assertRaises(ValueError):
            M3x3.copy().stack(M2x3, by=Rule.COL)

    def testPull(self) -> None:
        """Tests for `Matrix.pull()`"""

        res1 = M2x0.copy()
        res2 = res1.pull(by=Rule.ROW)

        exp1 = Matrix([], nrows=1, ncols=0)
        exp2 = Matrix([], nrows=1, ncols=0)

        self.assertTrue(eq(res1, exp1))
        self.assertTrue(eq(res2, exp2))

        res2 = res1.pull(by=Rule.ROW)

        exp1 = Matrix([], nrows=0, ncols=0)

        self.assertTrue(eq(res1, exp1))
        self.assertTrue(eq(res2, exp2))

        with self.assertRaises(IndexError):
            res1.pull(by=Rule.ROW)

        res1 = M3x3.copy()
        res2 = res1.pull(0, by=Rule.ROW)

        exp1 = Matrix([
            3, 4, 5,
            6, 7, 8,
        ], nrows=2, ncols=3)
        exp2 = Matrix([
            0, 1, 2,
        ], nrows=1, ncols=3)

        self.assertTrue(eq(exp1, res1))
        self.assertTrue(eq(exp2, res2))

        res1 = M3x3.copy()
        res2 = res1.pull(0, by=Rule.COL)

        exp1 = Matrix([
            1, 2,
            4, 5,
            7, 8,
        ], nrows=3, ncols=2)
        exp2 = Matrix([
            0,
            3,
            6,
        ], nrows=3, ncols=1)

        self.assertTrue(eq(exp1, res1))
        self.assertTrue(eq(exp2, res2))

        with self.assertRaises(IndexError):
            M3x3.copy().pull(3, by=Rule.ROW)

        with self.assertRaises(IndexError):
            M3x3.copy().pull(3, by=Rule.COL)
