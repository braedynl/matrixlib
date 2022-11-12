from unittest import TestCase

from ..matrices import Matrix
from ..rule import Rule

__all__ = ["TestMatrix"]


class TestMatrix(TestCase):

    def testGetItem(self):
        a = Matrix([
            0, 1, 2,
            3, 4, 5,
        ], nrows=2, ncols=3)

        self.assertEqual(a[0, 0], 0)
        self.assertEqual(a[0, 1], 1)
        self.assertEqual(a[0, 2], 2)
        self.assertEqual(a[1, 0], 3)
        self.assertEqual(a[1, 1], 4)
        self.assertEqual(a[1, 2], 5)

        with self.assertRaises(IndexError):
            a[2, 0]

        with self.assertRaises(IndexError):
            a[0, 3]

        with self.assertRaises(IndexError):
            a[2, 3]

        res = a[0, :]
        exp = Matrix([
            0, 1, 2,
        ], nrows=1, ncols=3)

        self.assertEqual(res, exp)

        res = a[1, :]
        exp = Matrix([
            3, 4, 5,
        ], nrows=1, ncols=3)

        self.assertEqual(res, exp)

        with self.assertRaises(IndexError):
            a[2, :]

        res = a[:, 0]
        exp = Matrix([
            0,
            3,
        ], nrows=2, ncols=1)

        self.assertEqual(res, exp)

        res = a[:, 1]
        exp = Matrix([
            1,
            4,
        ], nrows=2, ncols=1)

        self.assertEqual(res, exp)

        res = a[:, 2]
        exp = Matrix([
            2,
            5,
        ], nrows=2, ncols=1)

        self.assertEqual(res, exp)

        with self.assertRaises(IndexError):
            a[:, 3]

        res = a[:, :]
        exp = a

        self.assertEqual(res, exp)

        a = Matrix([
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
        ], nrows=3, ncols=3)

        res = a[1:, :]
        exp = Matrix([
            3, 4, 5,
            6, 7, 8,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

        res = a[:, 1:]
        exp = Matrix([
            1, 2,
            4, 5,
            7, 8,
        ], nrows=3, ncols=2)

        self.assertEqual(res, exp)

        res = a[1:, 1:]
        exp = Matrix([
            4, 5,
            7, 8,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)

    def testSetItem(self):
        res = Matrix([
            0, 1, 2,
            3, 4, 5,
        ], nrows=2, ncols=3)

        res[0, 0] = -1
        res[0, 1] = -2
        res[0, 2] = -3
        res[1, 0] = -4
        res[1, 1] = -5
        res[1, 2] = -6

        exp = Matrix([
            -1, -2, -3,
            -4, -5, -6,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

        with self.assertRaises(IndexError):
            res[2, 0] = -1

        with self.assertRaises(IndexError):
            res[0, 3] = -1

        with self.assertRaises(IndexError):
            res[2, 3] = -1

        res = Matrix([
            0, 1, 2,
            3, 4, 5,
        ], nrows=2, ncols=3)

        res[0, :] = Matrix([
            -1, -2, -3,
        ], nrows=1, ncols=3)
        exp = Matrix([
            -1, -2, -3,
             3,  4,  5,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

        res[1, :] = Matrix([
            -4, -5, -6,
        ], nrows=1, ncols=3)
        exp = Matrix([
            -1, -2, -3,
            -4, -5, -6,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

        with self.assertRaises(IndexError):
            res[2, :] = Matrix([
                -7, -8, -9,
            ], nrows=1, ncols=3)

        with self.assertRaises(ValueError):
            res[0, :] = Matrix([
                -1, -2, -3, -4,
            ], nrows=1, ncols=4)

        res = Matrix([
            0, 1, 2,
            3, 4, 5,
        ], nrows=2, ncols=3)

        res[:, 0] = Matrix([
            -1,
            -2,
        ], nrows=2, ncols=1)
        exp = Matrix([
            -1, 1, 2,
            -2, 4, 5,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

        res[:, 1] = Matrix([
            -3,
            -4,
        ], nrows=2, ncols=1)
        exp = Matrix([
            -1, -3, 2,
            -2, -4, 5,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

        res[:, 2] = Matrix([
            -5,
            -6,
        ], nrows=2, ncols=1)
        exp = Matrix([
            -1, -3, -5,
            -2, -4, -6,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

        with self.assertRaises(IndexError):
            res[:, 3] = Matrix([
                -7,
                -8,
            ], nrows=2, ncols=1)

        with self.assertRaises(ValueError):
            res[:, 0] = Matrix([
                -1,
                -2,
                -3,
            ], nrows=3, ncols=1)

        res = Matrix([
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
        ], nrows=3, ncols=3)

        res[:, :] = Matrix([
            -1, -2, -3,
            -4, -5, -6,
            -7, -8, -9,
        ], nrows=3, ncols=3)
        exp = Matrix([
            -1, -2, -3,
            -4, -5, -6,
            -7, -8, -9,
        ], nrows=3, ncols=3)

        self.assertEqual(res, exp)

        res = Matrix([
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
        ], nrows=3, ncols=3)

        res[1:, :] = Matrix([
            -4, -5, -6,
            -7, -8, -9,
        ], nrows=2, ncols=3)
        exp = Matrix([
             0,  1,  2,
            -4, -5, -6,
            -7, -8, -9,
        ], nrows=3, ncols=3)

        self.assertEqual(res, exp)

        res = Matrix([
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
        ], nrows=3, ncols=3)

        res[:, 1:] = Matrix([
            -1, -2,
            -3, -4,
            -5, -6,
        ], nrows=3, ncols=2)
        exp = Matrix([
            0, -1, -2,
            3, -3, -4,
            6, -5, -6,
        ], nrows=3, ncols=3)

        self.assertEqual(res, exp)

        res = Matrix([
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
        ], nrows=3, ncols=3)

        res[1:, 1:] = Matrix([
            -1, -2,
            -3, -4,
        ], nrows=2, ncols=2)
        exp = Matrix([
            0,  1,  2,
            3, -1, -2,
            6, -3, -4,
        ], nrows=3, ncols=3)

        self.assertEqual(res, exp)

    def testReshape(self):
        a = Matrix([], nrows=0, ncols=0)

        res = a.copy().reshape(0, 3)
        exp = Matrix([], nrows=0, ncols=3)

        self.assertEqual(res, exp)

        res = a.copy().reshape(3, 0)
        exp = Matrix([], nrows=3, ncols=0)

        self.assertEqual(res, exp)

        res = Matrix([
            0, 1, 2,
            3, 4, 5,
        ], nrows=2, ncols=3).reshape(3, 2)
        exp = Matrix([
            0, 1,
            2, 3,
            4, 5,
        ], nrows=3, ncols=2)

        self.assertEqual(res, exp)

        with self.assertRaises(ValueError):
            Matrix([
                0, 1, 2,
                3, 4, 5,
            ], nrows=2, ncols=3).reshape(3, 4)

        res = Matrix([], nrows=1, ncols=0).reshape(nrows=0)
        exp = Matrix([], nrows=0, ncols=0)

        self.assertEqual(res, exp)

        res = Matrix([], nrows=0, ncols=1).reshape(ncols=0)
        exp = Matrix([], nrows=0, ncols=0)

        self.assertEqual(res, exp)

        a = Matrix(range(6), nrows=1, ncols=6)

        res = a.copy().reshape(nrows=1)
        exp = a

        self.assertEqual(res, exp)

        res = a.copy().reshape(ncols=1)
        exp = Matrix(range(6), nrows=6, ncols=1)

        self.assertEqual(res, exp)

        res = a.copy().reshape(nrows=2)
        exp = Matrix(range(6), nrows=2, ncols=3)

        self.assertEqual(res, exp)

        res = a.copy().reshape(nrows=3)
        exp = Matrix(range(6), nrows=3, ncols=2)

        self.assertEqual(res, exp)

        res = a.copy().reshape(ncols=2)
        exp = Matrix(range(6), nrows=3, ncols=2)

        self.assertEqual(res, exp)

        res = a.copy().reshape(ncols=3)
        exp = Matrix(range(6), nrows=2, ncols=3)

        self.assertEqual(res, exp)

        res = a.copy().reshape(nrows=6)
        exp = Matrix(range(6), nrows=6, ncols=1)

        self.assertEqual(res, exp)

        res = a.copy().reshape(ncols=6)
        exp = Matrix(range(6), nrows=1, ncols=6)

        self.assertEqual(res, exp)

        res = a.copy().reshape()
        exp = a

        self.assertEqual(res, exp)

    def testSlices(self):
        a = Matrix([], nrows=0, ncols=0)

        it = a.slices(by=Rule.ROW)

        with self.assertRaises(StopIteration):
            next(it)

        it = a.slices(by=Rule.COL)

        with self.assertRaises(StopIteration):
            next(it)

        a = Matrix([], nrows=0, ncols=1)

        it = a.slices(by=Rule.ROW)

        with self.assertRaises(StopIteration):
            next(it)

        it = a.slices(by=Rule.COL)

        res = next(it)
        exp = Matrix([], nrows=0, ncols=1)

        self.assertEqual(res, exp)

        with self.assertRaises(StopIteration):
            next(it)

        a = Matrix([
            0, 1, 2,
            3, 4, 5,
        ], nrows=2, ncols=3)

        it = a.slices(by=Rule.ROW)

        res = next(it)
        exp = Matrix([
            0, 1, 2,
        ], nrows=1, ncols=3)

        self.assertEqual(res, exp)

        res = next(it)
        exp = Matrix([
            3, 4, 5,
        ], nrows=1, ncols=3)

        self.assertEqual(res, exp)

        with self.assertRaises(StopIteration):
            next(it)

        it = a.slices(by=Rule.COL)

        res = next(it)
        exp = Matrix([
            0,
            3,
        ], nrows=2, ncols=1)

        self.assertEqual(res, exp)

        res = next(it)
        exp = Matrix([
            1,
            4,
        ], nrows=2, ncols=1)

        self.assertEqual(res, exp)

        res = next(it)
        exp = Matrix([
            2,
            5,
        ], nrows=2, ncols=1)

        self.assertEqual(res, exp)

        with self.assertRaises(StopIteration):
            next(it)

    def testTranspose(self):
        res = Matrix([], nrows=0, ncols=0).transpose()
        exp = Matrix([], nrows=0, ncols=0)

        self.assertEqual(res, exp)

        res = Matrix([], nrows=0, ncols=1).transpose()
        exp = Matrix([], nrows=1, ncols=0)

        self.assertEqual(res, exp)

        res = Matrix([], nrows=1, ncols=0).transpose()
        exp = Matrix([], nrows=0, ncols=1)

        self.assertEqual(res, exp)

        res = Matrix([
            0, 1,
            2, 3,
        ], nrows=2, ncols=2).transpose()
        exp = Matrix([
            0, 2,
            1, 3,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)

        res = Matrix([
            0, 1, 2,
            3, 4, 5,
        ], nrows=2, ncols=3).transpose()
        exp = Matrix([
            0, 3,
            1, 4,
            2, 5,
        ], nrows=3, ncols=2)

        self.assertEqual(res, exp)

        res = Matrix([
            0, 1,
            2, 3,
            4, 5,
        ], nrows=3, ncols=2).transpose()
        exp = Matrix([
            0, 2, 4,
            1, 3, 5,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

    def testSwap(self):
        a = Matrix([], nrows=2, ncols=0)

        res = a.copy().swap(0, 1, by=Rule.ROW)
        exp = Matrix([], nrows=2, ncols=0)

        self.assertEqual(res, exp)

        with self.assertRaises(IndexError):
            a.copy().swap(0, 1, by=Rule.COL)

        a = Matrix([
            0, 1, 2,
            3, 4, 5,
        ], nrows=2, ncols=3)

        res = a.copy().swap(0, 1, by=Rule.ROW)
        exp = Matrix([
            3, 4, 5,
            0, 1, 2,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

        res = a.copy().swap(0, -1, by=Rule.COL)
        exp = Matrix([
            2, 1, 0,
            5, 4, 3,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

        with self.assertRaises(IndexError):
            a.copy().swap(0, 2, by=Rule.ROW)

        with self.assertRaises(IndexError):
            a.copy().swap(0, 3, by=Rule.COL)

    def testFlip(self):
        a = Matrix([], nrows=0, ncols=2)

        res = a.copy().flip(by=Rule.ROW)
        exp = Matrix([], nrows=0, ncols=2)

        self.assertEqual(res, exp)

        res = a.copy().flip(by=Rule.COL)
        exp = Matrix([], nrows=0, ncols=2)

        self.assertEqual(res, exp)

        res = Matrix([
            0, 1, 2,
            3, 4, 5,
        ], nrows=2, ncols=3).flip(by=Rule.ROW)
        exp = Matrix([
            3, 4, 5,
            0, 1, 2,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

        res = Matrix([
            0, 1,
            2, 3,
            4, 5,
        ], nrows=3, ncols=2).flip(by=Rule.COL)
        exp = Matrix([
            1, 0,
            3, 2,
            5, 4,
        ], nrows=3, ncols=2)

        self.assertEqual(res, exp)

        a = Matrix([
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
        ], nrows=3, ncols=3)

        res = a.copy().flip(by=Rule.ROW)
        exp = Matrix([
            6, 7, 8,
            3, 4, 5,
            0, 1, 2,
        ], nrows=3, ncols=3)

        self.assertEqual(res, exp)

        res = a.copy().flip(by=Rule.COL)
        exp = Matrix([
            2, 1, 0,
            5, 4, 3,
            8, 7, 6,
        ], nrows=3, ncols=3)

        self.assertEqual(res, exp)

    def testStack(self):
        a = Matrix([], nrows=2, ncols=0)

        res = a.copy().stack(
            Matrix([], nrows=1, ncols=0),
            by=Rule.ROW,
        )
        exp = Matrix([], nrows=3, ncols=0)

        self.assertEqual(res, exp)

        with self.assertRaises(ValueError):
            a.copy().stack(
                Matrix([], nrows=1, ncols=0),
                by=Rule.COL,
            )

        a = Matrix([], nrows=0, ncols=2)

        res = a.copy().stack(
            Matrix([], nrows=0, ncols=1),
            by=Rule.COL,
        )
        exp = Matrix([], nrows=0, ncols=3)

        self.assertEqual(res, exp)

        with self.assertRaises(ValueError):
            a.copy().stack(
                Matrix([], nrows=0, ncols=1),
                by=Rule.ROW,
            )

        a = Matrix([
            0, 1, 2,
            3, 4, 5,
        ], nrows=2, ncols=3)

        res = a.copy().stack(
            Matrix([
                6, 7, 8,
            ], nrows=1, ncols=3),
            by=Rule.ROW,
        )
        exp = Matrix([
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
        ], nrows=3, ncols=3)

        self.assertEqual(res, exp)

        res = a.copy().stack(
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

        self.assertEqual(res, exp)

        a = Matrix([
            0, 1,
            2, 3,
            4, 5,
        ], nrows=3, ncols=2)

        res = a.copy().stack(
            Matrix([
                -1,
                -2,
                -3,
            ], nrows=3, ncols=1),
            by=Rule.COL,
        )
        exp = Matrix([
            0, 1, -1,
            2, 3, -2,
            4, 5, -3,
        ], nrows=3, ncols=3)

        self.assertEqual(res, exp)

        res = a.copy().stack(
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

        self.assertEqual(res, exp)

        a = Matrix([
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
        ], nrows=3, ncols=3)

        with self.assertRaises(ValueError):
            a.copy().stack(
                Matrix([
                    0, 1,
                    2, 3,
                    4, 5,
                ], nrows=3, ncols=2),
                by=Rule.ROW,
            )

        with self.assertRaises(ValueError):
            a.copy().stack(
                Matrix([
                    0, 1, 2,
                    3, 4, 5,
                ], nrows=2, ncols=3),
                by=Rule.COL,
            )

    def testPull(self):
        res1 = Matrix([], nrows=2, ncols=0)
        res2 = res1.pull(by=Rule.ROW)

        exp1 = Matrix([], nrows=1, ncols=0)
        exp2 = Matrix([], nrows=1, ncols=0)

        self.assertEqual(res1, exp1)
        self.assertEqual(res2, exp2)

        res2 = res1.pull(by=Rule.ROW)

        exp1 = Matrix([], nrows=0, ncols=0)

        self.assertEqual(res1, exp1)
        self.assertEqual(res2, exp2)

        with self.assertRaises(IndexError):
            res1.pull(by=Rule.ROW)

        res1 = Matrix([
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
        ], nrows=3, ncols=3)
        res2 = res1.pull(0, by=Rule.ROW)

        exp1 = Matrix([
            3, 4, 5,
            6, 7, 8,
        ], nrows=2, ncols=3)
        exp2 = Matrix([
            0, 1, 2,
        ], nrows=1, ncols=3)

        self.assertEqual(res1, exp1)
        self.assertEqual(res2, exp2)

        res1 = Matrix([
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
        ], nrows=3, ncols=3)
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

        self.assertEqual(res1, exp1)
        self.assertEqual(res2, exp2)

        with self.assertRaises(IndexError):
            Matrix([
                0, 1, 2,
                3, 4, 5,
                6, 7, 8,
            ], nrows=3, ncols=3).pull(3, by=Rule.ROW)

        with self.assertRaises(IndexError):
            Matrix([
                0, 1, 2,
                3, 4, 5,
                6, 7, 8,
            ], nrows=3, ncols=3).pull(3, by=Rule.COL)
