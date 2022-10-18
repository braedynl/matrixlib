import itertools
from unittest import TestCase

from . import ComplexMatrix, IntegralMatrix, Matrix, RealMatrix, Rule

M0x0 = Matrix([], nrows=0, ncols=0)

M0x1 = Matrix([], nrows=0, ncols=1)
M1x0 = Matrix([], nrows=1, ncols=0)

M0x2 = Matrix([], nrows=0, ncols=2)
M2x0 = Matrix([], nrows=2, ncols=0)

M1x1 = Matrix([
    0,
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


class TestMatrix(TestCase):
    """Simple tests for methods of the `Matrix` class

    These tests are not exhaustive, both in terms of coverage across methods
    and for the tested methods themselves.

    Most methods are wrappers around built-ins and/or call one another. These
    tests are for methods that are a bit more "involved" or are heavily relied
    on for building other methods.
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

        self.assertTrue(res.equal(exp))

        res = M2x3[1, :]
        exp = Matrix([
            3, 4, 5,
        ], nrows=1, ncols=3)

        self.assertTrue(res.equal(exp))

        with self.assertRaises(IndexError):
            M2x3[2, :]

        res = M2x3[:, 0]
        exp = Matrix([
            0,
            3,
        ], nrows=2, ncols=1)

        self.assertTrue(res.equal(exp))

        res = M2x3[:, 1]
        exp = Matrix([
            1,
            4,
        ], nrows=2, ncols=1)

        self.assertTrue(res.equal(exp))

        res = M2x3[:, 2]
        exp = Matrix([
            2,
            5,
        ], nrows=2, ncols=1)

        self.assertTrue(res.equal(exp))

        with self.assertRaises(IndexError):
            M2x3[:, 3]

        res = M2x3[:, :]
        exp = M2x3

        self.assertTrue(res.equal(exp))

        res = M3x3[1:, :]
        exp = Matrix([
            3, 4, 5,
            6, 7, 8,
        ], nrows=2, ncols=3)

        self.assertTrue(res.equal(exp))

        res = M3x3[:, 1:]
        exp = Matrix([
            1, 2,
            4, 5,
            7, 8,
        ], nrows=3, ncols=2)

        self.assertTrue(res.equal(exp))

        res = M3x3[1:, 1:]
        exp = Matrix([
            4, 5,
            7, 8,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))

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

        res[0, :] = Matrix([
            -1, -2, -3,
        ], nrows=1, ncols=3)
        exp = Matrix([
            -1, -2, -3,
             3,  4,  5,
        ], nrows=2, ncols=3)

        self.assertTrue(res.equal(exp))

        res[1, :] = Matrix([
            -4, -5, -6,
        ], nrows=1, ncols=3)
        exp = Matrix([
            -1, -2, -3,
            -4, -5, -6,
        ], nrows=2, ncols=3)

        self.assertTrue(res.equal(exp))

        with self.assertRaises(IndexError):
            res[2, :] = Matrix([
                -7, -8, -9,
            ], nrows=1, ncols=3)

        with self.assertRaises(ValueError):
            res[0, :] = Matrix([
                -1, -2, -3, -4,
            ], nrows=1, ncols=4)

        res = M2x3.copy()

        res[:, 0] = Matrix([
            -1,
            -2,
        ], nrows=2, ncols=1)
        exp = Matrix([
            -1, 1, 2,
            -2, 4, 5,
        ], nrows=2, ncols=3)

        self.assertTrue(res.equal(exp))

        res[:, 1] = Matrix([
            -3,
            -4,
        ], nrows=2, ncols=1)
        exp = Matrix([
            -1, -3, 2,
            -2, -4, 5,
        ], nrows=2, ncols=3)

        self.assertTrue(res.equal(exp))

        res[:, 2] = Matrix([
            -5,
            -6,
        ], nrows=2, ncols=1)
        exp = Matrix([
            -1, -3, -5,
            -2, -4, -6,
        ], nrows=2, ncols=3)

        self.assertTrue(res.equal(exp))

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

        res = M3x3.copy()

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

        self.assertTrue(res.equal(exp))

        res = M3x3.copy()

        res[1:, :] = Matrix([
            -4, -5, -6,
            -7, -8, -9,
        ], nrows=2, ncols=3)
        exp = Matrix([
             0,  1,  2,
            -4, -5, -6,
            -7, -8, -9,
        ], nrows=3, ncols=3)

        self.assertTrue(res.equal(exp))

        res = M3x3.copy()

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

        self.assertTrue(res.equal(exp))

        res = M3x3.copy()

        res[1:, 1:] = Matrix([
            -1, -2,
            -3, -4,
        ], nrows=2, ncols=2)
        exp = Matrix([
            0,  1,  2,
            3, -1, -2,
            6, -3, -4,
        ], nrows=3, ncols=3)

        self.assertTrue(res.equal(exp))

    def testReshape(self) -> None:
        """Tests for `Matrix.reshape()`"""

        res = M0x0.copy().reshape(0, 3)
        exp = Matrix([], nrows=0, ncols=3)

        self.assertTrue(res.equal(exp))

        res = M0x0.copy().reshape(3, 0)
        exp = Matrix([], nrows=3, ncols=0)

        self.assertTrue(res.equal(exp))

        res = M2x3.copy().reshape(3, 2)
        exp = Matrix([
            0, 1,
            2, 3,
            4, 5,
        ], nrows=3, ncols=2)

        self.assertTrue(res.equal(exp))

        with self.assertRaises(ValueError):
            M2x3.copy().reshape(3, 4)

    def testSlices(self) -> None:
        """Tests for `Matrix.slices()`"""

        it = M0x0.slices(by=Rule.ROW)

        with self.assertRaises(StopIteration):
            next(it)

        it = M0x0.slices(by=Rule.COL)

        with self.assertRaises(StopIteration):
            next(it)

        it = M0x1.slices(by=Rule.ROW)

        with self.assertRaises(StopIteration):
            next(it)

        it = M0x1.slices(by=Rule.COL)

        res = next(it)
        exp = Matrix([], nrows=0, ncols=1)

        self.assertTrue(res.equal(exp))

        with self.assertRaises(StopIteration):
            next(it)

        it = M2x3.slices(by=Rule.ROW)

        res = next(it)
        exp = Matrix([
            0, 1, 2,
        ], nrows=1, ncols=3)

        self.assertTrue(res.equal(exp))

        res = next(it)
        exp = Matrix([
            3, 4, 5,
        ], nrows=1, ncols=3)

        self.assertTrue(res.equal(exp))

        with self.assertRaises(StopIteration):
            next(it)

        it = M2x3.slices(by=Rule.COL)

        res = next(it)
        exp = Matrix([
            0,
            3,
        ], nrows=2, ncols=1)

        self.assertTrue(res.equal(exp))

        res = next(it)
        exp = Matrix([
            1,
            4,
        ], nrows=2, ncols=1)

        self.assertTrue(res.equal(exp))

        res = next(it)
        exp = Matrix([
            2,
            5,
        ], nrows=2, ncols=1)

        self.assertTrue(res.equal(exp))

        with self.assertRaises(StopIteration):
            next(it)

    def testTranspose(self) -> None:
        """Tests for `Matrix.transpose()`"""

        res = M0x0.copy().transpose()
        exp = Matrix([], nrows=0, ncols=0)

        self.assertTrue(res.equal(exp))

        res = M0x1.copy().transpose()
        exp = Matrix([], nrows=1, ncols=0)

        self.assertTrue(res.equal(exp))

        res = M1x0.copy().transpose()
        exp = Matrix([], nrows=0, ncols=1)

        self.assertTrue(res.equal(exp))

        res = M2x2.copy().transpose()
        exp = Matrix([
            0, 2,
            1, 3,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))

        res = M2x3.copy().transpose()
        exp = Matrix([
            0, 3,
            1, 4,
            2, 5,
        ], nrows=3, ncols=2)

        self.assertTrue(res.equal(exp))

        res = M3x2.copy().transpose()
        exp = Matrix([
            0, 2, 4,
            1, 3, 5,
        ], nrows=2, ncols=3)

        self.assertTrue(res.equal(exp))

    def testSwap(self) -> None:
        """Tests for `Matrix.swap()`"""

        res = M2x0.copy().swap(0, 1, by=Rule.ROW)
        exp = Matrix([], nrows=2, ncols=0)

        self.assertTrue(res.equal(exp))

        with self.assertRaises(IndexError):
            M2x0.copy().swap(0, 1, by=Rule.COL)

        res = M2x3.copy().swap(0, 1, by=Rule.ROW)
        exp = Matrix([
            3, 4, 5,
            0, 1, 2,
        ], nrows=2, ncols=3)

        self.assertTrue(res.equal(exp))

        res = M2x3.copy().swap(0, -1, by=Rule.COL)
        exp = Matrix([
            2, 1, 0,
            5, 4, 3,
        ], nrows=2, ncols=3)

        self.assertTrue(res.equal(exp))

        with self.assertRaises(IndexError):
            M2x3.copy().swap(0, 2, by=Rule.ROW)

        with self.assertRaises(IndexError):
            M2x3.copy().swap(0, 3, by=Rule.COL)

    def testFlip(self) -> None:
        """Tests for `Matrix.flip()`"""

        res = M0x2.copy().flip(by=Rule.ROW)
        exp = Matrix([], nrows=0, ncols=2)

        self.assertTrue(res.equal(exp))

        res = M0x2.copy().flip(by=Rule.COL)
        exp = Matrix([], nrows=0, ncols=2)

        self.assertTrue(res.equal(exp))

        res = M2x3.copy().flip(by=Rule.ROW)
        exp = Matrix([
            3, 4, 5,
            0, 1, 2,
        ], nrows=2, ncols=3)

        self.assertTrue(res.equal(exp))

        res = M3x2.copy().flip(by=Rule.COL)
        exp = Matrix([
            1, 0,
            3, 2,
            5, 4,
        ], nrows=3, ncols=2)

        self.assertTrue(res.equal(exp))

        res = M3x3.copy().flip(by=Rule.ROW)
        exp = Matrix([
            6, 7, 8,
            3, 4, 5,
            0, 1, 2,
        ], nrows=3, ncols=3)

        self.assertTrue(res.equal(exp))

        res = M3x3.copy().flip(by=Rule.COL)
        exp = Matrix([
            2, 1, 0,
            5, 4, 3,
            8, 7, 6,
        ], nrows=3, ncols=3)

        self.assertTrue(res.equal(exp))

    def testStack(self) -> None:
        """Tests for `Matrix.stack()`"""

        res = M2x0.copy().stack(
            Matrix([], nrows=1, ncols=0),
            by=Rule.ROW,
        )
        exp = Matrix([], nrows=3, ncols=0)

        self.assertTrue(res.equal(exp))

        with self.assertRaises(ValueError):
            M2x0.copy().stack(
                Matrix([], nrows=1, ncols=0),
                by=Rule.COL,
            )

        res = M0x2.copy().stack(
            Matrix([], nrows=0, ncols=1),
            by=Rule.COL,
        )
        exp = Matrix([], nrows=0, ncols=3)

        self.assertTrue(res.equal(exp))

        with self.assertRaises(ValueError):
            M0x2.copy().stack(
                Matrix([], nrows=0, ncols=1),
                by=Rule.ROW,
            )

        res = M2x3.copy().stack(
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

        self.assertTrue(res.equal(exp))

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

        self.assertTrue(res.equal(exp))

        res = M3x2.copy().stack(
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

        self.assertTrue(res.equal(exp))

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

        self.assertTrue(res.equal(exp))

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

        self.assertTrue(res1.equal(exp1))
        self.assertTrue(res2.equal(exp2))

        res2 = res1.pull(by=Rule.ROW)

        exp1 = Matrix([], nrows=0, ncols=0)

        self.assertTrue(res1.equal(exp1))
        self.assertTrue(res2.equal(exp2))

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

        self.assertTrue(res1.equal(exp1))
        self.assertTrue(res2.equal(exp2))

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

        self.assertTrue(res1.equal(exp1))
        self.assertTrue(res2.equal(exp2))

        with self.assertRaises(IndexError):
            M3x3.copy().pull(3, by=Rule.ROW)

        with self.assertRaises(IndexError):
            M3x3.copy().pull(3, by=Rule.COL)


class TestComplexMatrix(TestCase):

    def testOperatorClassDeduction(self):
        """Tests for binary operator class deduction"""

        a = ComplexMatrix([
            1j, 2j,
            3j, 4j,
        ], nrows=2, ncols=2)
        b = a

        res = a + b  # ComplexMatrix + ComplexMatrix -> ComplexMatrix
        exp = ComplexMatrix([
            2j, 4j,
            6j, 8j,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = 1j

        res = a + b  # ComplexMatrix + Complex -> ComplexMatrix
        exp = ComplexMatrix([
            2j, 3j,
            4j, 5j,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = RealMatrix([
            1.0, 2.0,
            3.0, 4.0,
        ], nrows=2, ncols=2)

        res = a + b  # ComplexMatrix + RealMatrix -> ComplexMatrix
        exp = ComplexMatrix([
            1+1j, 2+2j,
            3+3j, 4+4j,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = 1.0

        res = a + b  # ComplexMatrix + Real -> ComplexMatrix
        exp = ComplexMatrix([
            1+1j, 1+2j,
            1+3j, 1+4j,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = IntegralMatrix([
            1, 2,
            3, 4,
        ], nrows=2, ncols=2)

        res = a + b  # ComplexMatrix + IntegralMatrix -> ComplexMatrix
        exp = ComplexMatrix([
            1+1j, 2+2j,
            3+3j, 4+4j,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = 1

        res = a + b  # ComplexMatrix + Integral -> ComplexMatrix
        exp = ComplexMatrix([
            1+1j, 1+2j,
            1+3j, 1+4j,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = Matrix([
            1, 2,
            3, 4,
        ], nrows=2, ncols=2)

        res = a + b  # ComplexMatrix + (Matrix or Any) -> Matrix
        exp = Matrix([
            1+1j, 2+2j,
            3+3j, 4+4j,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, Matrix))


class TestRealMatrix(TestCase):

    def testOperatorClassDeduction(self):
        """Tests for binary operator class deduction"""

        a = RealMatrix([
            1.0, 2.0,
            3.0, 4.0,
        ], nrows=2, ncols=2)
        b = a

        res = a + b  # RealMatrix + RealMatrix -> RealMatrix
        exp = RealMatrix([
            2.0, 4.0,
            6.0, 8.0,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, RealMatrix))

        b = 1.0

        res = a + b  # RealMatrix + Real -> RealMatrix
        exp = RealMatrix([
            2.0, 3.0,
            4.0, 5.0,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, RealMatrix))

        b = ComplexMatrix([
            1j, 2j,
            3j, 4j,
        ], nrows=2, ncols=2)

        res = a + b  # RealMatrix + ComplexMatrix -> ComplexMatrix
        exp = ComplexMatrix([
            1+1j, 2+2j,
            3+3j, 4+4j,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = 1j

        res = a + b  # RealMatrix + Complex -> ComplexMatrix
        exp = ComplexMatrix([
            1+1j, 2+1j,
            3+1j, 4+1j,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = IntegralMatrix([
            1, 2,
            3, 4,
        ], nrows=2, ncols=2)

        res = a + b  # RealMatrix + IntegralMatrix -> RealMatrix
        exp = RealMatrix([
            2.0, 4.0,
            6.0, 8.0,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, RealMatrix))

        b = 1

        res = a + b  # RealMatrix + Integral -> RealMatrix
        exp = RealMatrix([
            2.0, 3.0,
            4.0, 5.0,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, RealMatrix))

        b = Matrix([
            1, 2,
            3, 4,
        ], nrows=2, ncols=2)

        res = a + b  # RealMatrix + (Matrix or Any) -> Matrix
        exp = Matrix([
            2.0, 4.0,
            6.0, 8.0,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, Matrix))


class TestIntegralMatrix(TestCase):

    def testOperatorClassDeduction(self):
        """Tests for binary operator class deduction"""

        a = IntegralMatrix([
            1, 2,
            3, 4,
        ], nrows=2, ncols=2)
        b = a

        res = a + b  # IntegralMatrix + IntegralMatrix -> IntegralMatrix
        exp = IntegralMatrix([
            2, 4,
            6, 8,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, IntegralMatrix))

        b = 1

        res = a + b  # IntegralMatrix + Integral -> IntegralMatrix
        exp = IntegralMatrix([
            2, 3,
            4, 5,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, IntegralMatrix))

        b = ComplexMatrix([
            1j, 2j,
            3j, 4j,
        ], nrows=2, ncols=2)

        res = a + b  # IntegralMatrix + ComplexMatrix -> ComplexMatrix
        exp = ComplexMatrix([
            1+1j, 2+2j,
            3+3j, 4+4j,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = 1j

        res = a + b  # IntegralMatrix + Complex -> ComplexMatrix
        exp = ComplexMatrix([
            1+1j, 2+1j,
            3+1j, 4+1j,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = RealMatrix([
            1.0, 2.0,
            3.0, 4.0,
        ], nrows=2, ncols=2)

        res = a + b  # IntegralMatrix + RealMatrix -> RealMatrix
        exp = RealMatrix([
            2.0, 4.0,
            6.0, 8.0,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, RealMatrix))

        b = 1.0

        res = a + b  # IntegralMatrix + Real -> RealMatrix
        exp = RealMatrix([
            2.0, 3.0,
            4.0, 5.0,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, RealMatrix))

        b = Matrix([
            1, 2,
            3, 4,
        ], nrows=2, ncols=2)

        res = a + b  # IntegralMatrix + (Matrix or Any) -> Matrix
        exp = Matrix([
            2, 4,
            6, 8,
        ], nrows=2, ncols=2)

        self.assertTrue(res.equal(exp))
        self.assertTrue(isinstance(res, Matrix))
