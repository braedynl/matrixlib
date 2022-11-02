import itertools
from decimal import Decimal
from fractions import Fraction
from unittest import TestCase

from .matrices import ComplexMatrix, IntegralMatrix, Matrix, RealMatrix
from .protocols import (ComplexLike, ComplexMatrixLike, IntegralLike,
                        IntegralMatrixLike, MatrixLike, RealLike,
                        RealMatrixLike, ShapeLike)
from .rule import Rule
from .shape import Shape

__all__ = [
    "TestMatrix",
    "TestComplexMatrix",
    "TestRealMatrix",
    "TestIntegralMatrix",
    "TestProtocols",
]

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

    def testGetItem(self):
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

        self.assertEqual(res, exp)

        res = M2x3[1, :]
        exp = Matrix([
            3, 4, 5,
        ], nrows=1, ncols=3)

        self.assertEqual(res, exp)

        with self.assertRaises(IndexError):
            M2x3[2, :]

        res = M2x3[:, 0]
        exp = Matrix([
            0,
            3,
        ], nrows=2, ncols=1)

        self.assertEqual(res, exp)

        res = M2x3[:, 1]
        exp = Matrix([
            1,
            4,
        ], nrows=2, ncols=1)

        self.assertEqual(res, exp)

        res = M2x3[:, 2]
        exp = Matrix([
            2,
            5,
        ], nrows=2, ncols=1)

        self.assertEqual(res, exp)

        with self.assertRaises(IndexError):
            M2x3[:, 3]

        res = M2x3[:, :]
        exp = M2x3

        self.assertEqual(res, exp)

        res = M3x3[1:, :]
        exp = Matrix([
            3, 4, 5,
            6, 7, 8,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

        res = M3x3[:, 1:]
        exp = Matrix([
            1, 2,
            4, 5,
            7, 8,
        ], nrows=3, ncols=2)

        self.assertEqual(res, exp)

        res = M3x3[1:, 1:]
        exp = Matrix([
            4, 5,
            7, 8,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)

    def testSetItem(self):
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

        res = M2x3.copy()

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

        self.assertEqual(res, exp)

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

        self.assertEqual(res, exp)

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

        self.assertEqual(res, exp)

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

        self.assertEqual(res, exp)

    def testReshape(self):
        """Tests for `Matrix.reshape()`"""

        res = M0x0.copy().reshape(0, 3)
        exp = Matrix([], nrows=0, ncols=3)

        self.assertEqual(res, exp)

        res = M0x0.copy().reshape(3, 0)
        exp = Matrix([], nrows=3, ncols=0)

        self.assertEqual(res, exp)

        res = M2x3.copy().reshape(3, 2)
        exp = Matrix([
            0, 1,
            2, 3,
            4, 5,
        ], nrows=3, ncols=2)

        self.assertEqual(res, exp)

        with self.assertRaises(ValueError):
            M2x3.copy().reshape(3, 4)

        res = M1x0.copy().reshape(nrows=0)
        exp = M0x0

        self.assertEqual(res, exp)

        res = M0x1.copy().reshape(ncols=0)
        exp = M0x0

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

        self.assertEqual(res, exp)

        with self.assertRaises(StopIteration):
            next(it)

        it = M2x3.slices(by=Rule.ROW)

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

        it = M2x3.slices(by=Rule.COL)

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
        """Tests for `Matrix.transpose()`"""

        res = M0x0.copy().transpose()
        exp = Matrix([], nrows=0, ncols=0)

        self.assertEqual(res, exp)

        res = M0x1.copy().transpose()
        exp = Matrix([], nrows=1, ncols=0)

        self.assertEqual(res, exp)

        res = M1x0.copy().transpose()
        exp = Matrix([], nrows=0, ncols=1)

        self.assertEqual(res, exp)

        res = M2x2.copy().transpose()
        exp = Matrix([
            0, 2,
            1, 3,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)

        res = M2x3.copy().transpose()
        exp = Matrix([
            0, 3,
            1, 4,
            2, 5,
        ], nrows=3, ncols=2)

        self.assertEqual(res, exp)

        res = M3x2.copy().transpose()
        exp = Matrix([
            0, 2, 4,
            1, 3, 5,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

    def testSwap(self):
        """Tests for `Matrix.swap()`"""

        res = M2x0.copy().swap(0, 1, by=Rule.ROW)
        exp = Matrix([], nrows=2, ncols=0)

        self.assertEqual(res, exp)

        with self.assertRaises(IndexError):
            M2x0.copy().swap(0, 1, by=Rule.COL)

        res = M2x3.copy().swap(0, 1, by=Rule.ROW)
        exp = Matrix([
            3, 4, 5,
            0, 1, 2,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

        res = M2x3.copy().swap(0, -1, by=Rule.COL)
        exp = Matrix([
            2, 1, 0,
            5, 4, 3,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

        with self.assertRaises(IndexError):
            M2x3.copy().swap(0, 2, by=Rule.ROW)

        with self.assertRaises(IndexError):
            M2x3.copy().swap(0, 3, by=Rule.COL)

    def testFlip(self):
        """Tests for `Matrix.flip()`"""

        res = M0x2.copy().flip(by=Rule.ROW)
        exp = Matrix([], nrows=0, ncols=2)

        self.assertEqual(res, exp)

        res = M0x2.copy().flip(by=Rule.COL)
        exp = Matrix([], nrows=0, ncols=2)

        self.assertEqual(res, exp)

        res = M2x3.copy().flip(by=Rule.ROW)
        exp = Matrix([
            3, 4, 5,
            0, 1, 2,
        ], nrows=2, ncols=3)

        self.assertEqual(res, exp)

        res = M3x2.copy().flip(by=Rule.COL)
        exp = Matrix([
            1, 0,
            3, 2,
            5, 4,
        ], nrows=3, ncols=2)

        self.assertEqual(res, exp)

        res = M3x3.copy().flip(by=Rule.ROW)
        exp = Matrix([
            6, 7, 8,
            3, 4, 5,
            0, 1, 2,
        ], nrows=3, ncols=3)

        self.assertEqual(res, exp)

        res = M3x3.copy().flip(by=Rule.COL)
        exp = Matrix([
            2, 1, 0,
            5, 4, 3,
            8, 7, 6,
        ], nrows=3, ncols=3)

        self.assertEqual(res, exp)

    def testStack(self):
        """Tests for `Matrix.stack()`"""

        res = M2x0.copy().stack(
            Matrix([], nrows=1, ncols=0),
            by=Rule.ROW,
        )
        exp = Matrix([], nrows=3, ncols=0)

        self.assertEqual(res, exp)

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

        self.assertEqual(res, exp)

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

        self.assertEqual(res, exp)

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

        self.assertEqual(res, exp)

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

        self.assertEqual(res, exp)

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

        self.assertEqual(res, exp)

        with self.assertRaises(ValueError):
            M3x3.copy().stack(M3x2, by=Rule.ROW)

        with self.assertRaises(ValueError):
            M3x3.copy().stack(M2x3, by=Rule.COL)

    def testPull(self):
        """Tests for `Matrix.pull()`"""

        res1 = M2x0.copy()
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

        res1 = M3x3.copy()
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

        self.assertEqual(res1, exp1)
        self.assertEqual(res2, exp2)

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

        # ComplexMatrix + ComplexMatrix -> ComplexMatrix
        res = a + b
        exp = ComplexMatrix([
            2j, 4j,
            6j, 8j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = 1j

        # ComplexMatrix + Complex -> ComplexMatrix
        res = a + b
        exp = ComplexMatrix([
            2j, 3j,
            4j, 5j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = RealMatrix([
            1.0, 2.0,
            3.0, 4.0,
        ], nrows=2, ncols=2)

        # ComplexMatrix + RealMatrix -> ComplexMatrix
        res = a + b
        exp = ComplexMatrix([
            1+1j, 2+2j,
            3+3j, 4+4j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = 1.0

        # ComplexMatrix + Real -> ComplexMatrix
        res = a + b
        exp = ComplexMatrix([
            1+1j, 1+2j,
            1+3j, 1+4j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = IntegralMatrix([
            1, 2,
            3, 4,
        ], nrows=2, ncols=2)

        # ComplexMatrix + IntegralMatrix -> ComplexMatrix
        res = a + b
        exp = ComplexMatrix([
            1+1j, 2+2j,
            3+3j, 4+4j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = 1

        # ComplexMatrix + Integral -> ComplexMatrix
        res = a + b
        exp = ComplexMatrix([
            1+1j, 1+2j,
            1+3j, 1+4j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = Matrix([
            1, 2,
            3, 4,
        ], nrows=2, ncols=2)

        # ComplexMatrix + (Matrix or Any) -> Matrix
        res = a + b
        exp = Matrix([
            1+1j, 2+2j,
            3+3j, 4+4j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, Matrix))


class TestRealMatrix(TestCase):

    def testOperatorClassDeduction(self):
        """Tests for binary operator class deduction"""

        a = RealMatrix([
            1.0, 2.0,
            3.0, 4.0,
        ], nrows=2, ncols=2)
        b = a

        # RealMatrix + RealMatrix -> RealMatrix
        res = a + b
        exp = RealMatrix([
            2.0, 4.0,
            6.0, 8.0,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, RealMatrix))

        b = 1.0

        # RealMatrix + Real -> RealMatrix
        res = a + b
        exp = RealMatrix([
            2.0, 3.0,
            4.0, 5.0,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, RealMatrix))

        b = ComplexMatrix([
            1j, 2j,
            3j, 4j,
        ], nrows=2, ncols=2)

        # RealMatrix + ComplexMatrix -> ComplexMatrix
        res = a + b
        exp = ComplexMatrix([
            1+1j, 2+2j,
            3+3j, 4+4j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = 1j

        # RealMatrix + Complex -> ComplexMatrix
        res = a + b
        exp = ComplexMatrix([
            1+1j, 2+1j,
            3+1j, 4+1j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = IntegralMatrix([
            1, 2,
            3, 4,
        ], nrows=2, ncols=2)

        # RealMatrix + IntegralMatrix -> RealMatrix
        res = a + b
        exp = RealMatrix([
            2.0, 4.0,
            6.0, 8.0,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, RealMatrix))

        b = 1

        # RealMatrix + Integral -> RealMatrix
        res = a + b
        exp = RealMatrix([
            2.0, 3.0,
            4.0, 5.0,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, RealMatrix))

        b = Matrix([
            1, 2,
            3, 4,
        ], nrows=2, ncols=2)

        # RealMatrix + (Matrix or Any) -> Matrix
        res = a + b
        exp = Matrix([
            2.0, 4.0,
            6.0, 8.0,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, Matrix))


class TestIntegralMatrix(TestCase):

    def testOperatorClassDeduction(self):
        """Tests for binary operator class deduction"""

        a = IntegralMatrix([
            1, 2,
            3, 4,
        ], nrows=2, ncols=2)
        b = a

        # IntegralMatrix + IntegralMatrix -> IntegralMatrix
        res = a + b
        exp = IntegralMatrix([
            2, 4,
            6, 8,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, IntegralMatrix))

        b = 1

        # IntegralMatrix + Integral -> IntegralMatrix
        res = a + b
        exp = IntegralMatrix([
            2, 3,
            4, 5,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, IntegralMatrix))

        b = ComplexMatrix([
            1j, 2j,
            3j, 4j,
        ], nrows=2, ncols=2)

        # IntegralMatrix + ComplexMatrix -> ComplexMatrix
        res = a + b
        exp = ComplexMatrix([
            1+1j, 2+2j,
            3+3j, 4+4j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = 1j

        # IntegralMatrix + Complex -> ComplexMatrix
        res = a + b
        exp = ComplexMatrix([
            1+1j, 2+1j,
            3+1j, 4+1j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = RealMatrix([
            1.0, 2.0,
            3.0, 4.0,
        ], nrows=2, ncols=2)

        # IntegralMatrix + RealMatrix -> RealMatrix
        res = a + b
        exp = RealMatrix([
            2.0, 4.0,
            6.0, 8.0,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, RealMatrix))

        b = 1.0

        # IntegralMatrix + Real -> RealMatrix
        res = a + b
        exp = RealMatrix([
            2.0, 3.0,
            4.0, 5.0,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, RealMatrix))

        b = Matrix([
            1, 2,
            3, 4,
        ], nrows=2, ncols=2)

        # IntegralMatrix + (Matrix or Any) -> Matrix
        res = a + b
        exp = Matrix([
            2, 4,
            6, 8,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, Matrix))


class TestProtocols(TestCase):

    def testNumeric(self):
        """Tests for numeric-like protocols"""

        a = 0j

        self.assertTrue(isinstance(a, ComplexLike))
        self.assertFalse(isinstance(a, RealLike))
        self.assertFalse(isinstance(a, IntegralLike))

        a = 0.0

        self.assertTrue(isinstance(a, ComplexLike))
        self.assertTrue(isinstance(a, RealLike))
        self.assertFalse(isinstance(a, IntegralLike))

        a = 0

        self.assertTrue(isinstance(a, ComplexLike))
        self.assertTrue(isinstance(a, RealLike))
        self.assertTrue(isinstance(a, IntegralLike))

        a = ComplexMatrix([], 0, 0)

        self.assertTrue(isinstance(a, ComplexLike))
        self.assertFalse(isinstance(a, RealLike))
        self.assertFalse(isinstance(a, IntegralLike))

        a = RealMatrix([], 0, 0)

        self.assertTrue(isinstance(a, ComplexLike))
        self.assertTrue(isinstance(a, RealLike))
        self.assertFalse(isinstance(a, IntegralLike))

        a = IntegralMatrix([], 0, 0)

        self.assertTrue(isinstance(a, ComplexLike))
        self.assertTrue(isinstance(a, RealLike))
        self.assertTrue(isinstance(a, IntegralLike))

        a = Fraction()

        self.assertTrue(isinstance(a, ComplexLike))
        self.assertTrue(isinstance(a, RealLike))
        self.assertFalse(isinstance(a, IntegralLike))

        a = Decimal()  # Should pass, but is heterogeneously unsafe

        self.assertTrue(isinstance(a, ComplexLike))
        self.assertTrue(isinstance(a, RealLike))
        self.assertFalse(isinstance(a, IntegralLike))

    def testShape(self):
        """Tests for the `ShapeLike` protocol"""

        a = Shape()

        self.assertTrue(isinstance(a, ShapeLike))

    def testMatrix(self):
        """Tests for the `MatrixLike` protocol"""

        a = Matrix([], 0, 0)

        self.assertTrue(isinstance(a, MatrixLike))

    def testNumericMatrix(self):
        """Tests for numeric matrix-like protocols"""

        a = ComplexMatrix([], 0, 0)

        self.assertTrue(isinstance(a, ComplexMatrixLike))
        self.assertFalse(isinstance(a, RealMatrixLike))
        self.assertFalse(isinstance(a, IntegralMatrixLike))

        a = RealMatrix([], 0, 0)

        self.assertTrue(isinstance(a, ComplexMatrixLike))
        self.assertTrue(isinstance(a, RealMatrixLike))
        self.assertFalse(isinstance(a, IntegralMatrixLike))

        a = IntegralMatrix([], 0, 0)

        self.assertTrue(isinstance(a, ComplexMatrixLike))
        self.assertTrue(isinstance(a, RealMatrixLike))
        self.assertTrue(isinstance(a, IntegralMatrixLike))
