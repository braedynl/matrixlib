from decimal import Decimal
from fractions import Fraction
from unittest import TestCase

from ..matrices import ComplexMatrix, IntegralMatrix, Matrix, RealMatrix
from ..protocols import (ComplexLike, ComplexMatrixLike, IntegralLike,
                         IntegralMatrixLike, MatrixLike, RealLike,
                         RealMatrixLike, ShapeLike)
from ..shape import Shape

__all__ = [
    "TestProtocols",
    "TestComplexMatrix",
    "TestRealMatrix",
    "TestIntegralMatrix",
]


class TestProtocols(TestCase):

    def testNumeric(self):
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

        a = ComplexMatrix()

        self.assertTrue(isinstance(a, ComplexLike))
        self.assertFalse(isinstance(a, RealLike))
        self.assertFalse(isinstance(a, IntegralLike))

        a = RealMatrix()

        self.assertTrue(isinstance(a, ComplexLike))
        self.assertTrue(isinstance(a, RealLike))
        self.assertFalse(isinstance(a, IntegralLike))

        a = IntegralMatrix()

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
        a = Shape()

        self.assertTrue(isinstance(a, ShapeLike))

    def testMatrix(self):
        a = Matrix()

        self.assertTrue(isinstance(a, MatrixLike))

    def testNumericMatrix(self):
        a = ComplexMatrix()

        self.assertTrue(isinstance(a, ComplexMatrixLike))
        self.assertFalse(isinstance(a, RealMatrixLike))
        self.assertFalse(isinstance(a, IntegralMatrixLike))

        a = RealMatrix()

        self.assertTrue(isinstance(a, ComplexMatrixLike))
        self.assertTrue(isinstance(a, RealMatrixLike))
        self.assertFalse(isinstance(a, IntegralMatrixLike))

        a = IntegralMatrix()

        self.assertTrue(isinstance(a, ComplexMatrixLike))
        self.assertTrue(isinstance(a, RealMatrixLike))
        self.assertTrue(isinstance(a, IntegralMatrixLike))


class TestComplexMatrix(TestCase):

    def testOperatorClassDeduction(self):
        a = ComplexMatrix([
            1j, 2j,
            3j, 4j,
        ], nrows=2, ncols=2)
        b = a

        res = a + b
        exp = ComplexMatrix([
            2j, 4j,
            6j, 8j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = 1j

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

        res = a + b
        exp = ComplexMatrix([
            1+1j, 2+2j,
            3+3j, 4+4j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = 1.0

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

        res = a + b
        exp = ComplexMatrix([
            1+1j, 2+2j,
            3+3j, 4+4j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = 1

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

        res = a + b
        exp = Matrix([
            1+1j, 2+2j,
            3+3j, 4+4j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, Matrix))


class TestRealMatrix(TestCase):

    def testOperatorClassDeduction(self):
        a = RealMatrix([
            1.0, 2.0,
            3.0, 4.0,
        ], nrows=2, ncols=2)
        b = a

        res = a + b
        exp = RealMatrix([
            2.0, 4.0,
            6.0, 8.0,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, RealMatrix))

        b = 1.0

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

        res = a + b
        exp = ComplexMatrix([
            1+1j, 2+2j,
            3+3j, 4+4j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = 1j

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

        res = a + b
        exp = RealMatrix([
            2.0, 4.0,
            6.0, 8.0,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, RealMatrix))

        b = 1

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

        res = a + b
        exp = Matrix([
            2.0, 4.0,
            6.0, 8.0,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, Matrix))


class TestIntegralMatrix(TestCase):

    def testOperatorClassDeduction(self):
        a = IntegralMatrix([
            1, 2,
            3, 4,
        ], nrows=2, ncols=2)
        b = a

        res = a + b
        exp = IntegralMatrix([
            2, 4,
            6, 8,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, IntegralMatrix))

        b = 1

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

        res = a + b
        exp = ComplexMatrix([
            1+1j, 2+2j,
            3+3j, 4+4j,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, ComplexMatrix))

        b = 1j

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

        res = a + b
        exp = RealMatrix([
            2.0, 4.0,
            6.0, 8.0,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, RealMatrix))

        b = 1.0

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

        res = a + b
        exp = Matrix([
            2, 4,
            6, 8,
        ], nrows=2, ncols=2)

        self.assertEqual(res, exp)
        self.assertTrue(isinstance(res, Matrix))
