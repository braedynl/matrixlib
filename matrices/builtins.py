from __future__ import annotations

import itertools
import operator
from abc import abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal, Optional, Protocol, TypeVar, Union, overload

from typing_extensions import Self

from .abc import Shaped
from .rule import Rule
from .utilities import EvenNumber, Grid, Mesh, OddNumber

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)
P_co = TypeVar("P_co", covariant=True, bound=int)

C_co = TypeVar("C_co", covariant=True, bound=complex)
R_co = TypeVar("R_co", covariant=True, bound=float)
I_co = TypeVar("I_co", covariant=True, bound=int)


class SupportsMatrixProperties(Shaped[M_co, N_co], Protocol[M_co, N_co, T_co]):

    @property
    @abstractmethod
    def data(self) -> Mesh[M_co, N_co, T_co]:
        """The matrix's grid object

        Every matrix holds an instance of a ``Mesh`` class that provides the
        "hybrid" one/two-dimensional interface for the matrix. Certain
        operations that permute the matrix's values (such as transposition,
        rotation, etc.) are implemented as ``Mesh`` sub-classes that "move"
        indices to their permuted positions before in-memory access occurs.

        This composition structure allows for "permutation types" to exist
        beneath the ``Matrix`` abstraction layer - allowing for easier
        ``Matrix`` sub-classing, since, the implementor would not have to
        create a set of permutation sub-classes that each provide an interface
        alike the material sub-class.
        """
        raise NotImplementedError

    @property
    def array(self) -> Sequence[T_co]:
        """The underlying sequence of matrix values, aligned in row-major order

        This is usually a built-in ``tuple``, but may vary depending on how the
        matrix was created.
        """
        return self.data.array

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self.data.shape


class Matrix(SupportsMatrixProperties[M_co, N_co, T_co], Sequence[T_co]):

    __slots__ = ("data",)
    __match_args__ = ("array", "shape")

    @overload
    def __init__(self, array: Mesh[M_co, N_co, T_co]) -> None: ...
    @overload
    def __init__(self, array: Matrix[M_co, N_co, T_co]) -> None: ...
    @overload
    def __init__(self, array: Iterable[T_co] = (), shape: Optional[tuple[M_co, N_co]] = None) -> None: ...

    def __init__(self, array=(), shape=None):
        self.data: Mesh[M_co, N_co, T_co]  # type: ignore
        if isinstance(array, Mesh):
            self.data = array
        elif isinstance(array, Matrix):
            self.data = array.data
        else:
            self.data = Grid(array, shape)

    def __repr__(self) -> str:
        """Return a canonical representation of the matrix"""
        array_values = ", ".join(map(repr, self))
        shape_values = ", ".join(map(repr, self.shape))
        return f"{self.__class__.__name__}(array=({array_values}), shape=({shape_values}))"

    def __eq__(self, other: object) -> bool:
        """Return true if the two matrices are equal, otherwise false"""
        if self is other:
            return True
        if isinstance(other, Matrix):
            return self.data == other.data
        return NotImplemented

    def __hash__(self) -> int:
        """Return a hash of the matrix"""
        return hash(self.data)

    @overload
    def __getitem__(self, key: int) -> T_co: ...
    @overload
    def __getitem__(self, key: slice) -> Matrix[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> Matrix[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> Matrix[Any, Literal[1], T_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> Matrix[Any, Any, T_co]: ...

    def __getitem__(self, key):
        """Return the value or sub-matrix corresponding to ``key``"""
        result = self.data[key]
        if isinstance(key, slice):
            return Matrix(result)
        if isinstance(key, tuple) and (isinstance(key[0], slice) or isinstance(key[1], slice)):
            return Matrix(result)
        return result

    def __iter__(self) -> Iterator[T_co]:
        """Return an iterator over the values of the matrix in row-major order"""
        return iter(self.data)

    def __reversed__(self) -> Iterator[T_co]:
        """Return an iterator over the values of the matrix in reverse
        row-major order
        """
        return reversed(self.data)

    def __contains__(self, value: object) -> bool:
        """Return true if the matrix contains ``value``, otherwise false"""
        return value in self.data

    def __deepcopy__(self, memo=None) -> Self:
        """Return the matrix"""
        return self

    __copy__ = __deepcopy__

    @classmethod
    def from_nesting(cls, nesting: Iterable[Iterable[T_co]]) -> Self:
        """Construct a matrix from a singly-nested iterable, using the
        shallowest iterable's length to deduce the number of rows, and the
        nested iterables' length to deduce the number of columns

        Raises ``ValueError`` if the length of the nested iterables is
        inconsistent (debug-only).
        """
        array: list[T_co] = []

        nrows = 0
        ncols = 0

        rows = iter(nesting)
        try:
            row = next(rows)
        except StopIteration:
            return cls(array, (nrows, ncols))
        else:
            array.extend(row)

        nrows = 1
        ncols = len(array)

        if __debug__:
            for row in rows:
                n = 0
                for val in row:
                    array.append(val)
                    n += 1
                if n != ncols:
                    raise ValueError(f"row at index {nrows} has length {n}, but precedent rows have length {ncols}")
                nrows += 1
        else:
            for row in rows:
                array.extend(row)
                nrows += 1

        return cls(array, (nrows, ncols))

    def transpose(self) -> Matrix[N_co, M_co, T_co]:
        """Return a transposed view of the matrix"""
        return Matrix(self.data.transpose())

    def flip(self, *, by: Rule = Rule.ROW) -> Matrix[M_co, N_co, T_co]:
        """Return a flipped view of the matrix"""
        return Matrix(self.data.flip(by=by))

    @overload
    def rotate(self, n: EvenNumber) -> Matrix[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> Matrix[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Union[Matrix[M_co, N_co, T_co], Matrix[N_co, M_co, T_co]]: ...
    @overload
    def rotate(self) -> Matrix[N_co, M_co, T_co]: ...

    def rotate(self, n=1):
        """Return a rotated view of the matrix

        Rotates the matrix ``n`` times in increments of 90 degrees. The matrix
        is rotated counter-clockwise if ``n`` is positive, and clockwise if
        ``n`` is negative. All integers are accepted, but many (particularly
        those outside of a small range near zero) will lose typing precision.
        """
        return Matrix(self.data.rotate(n))

    def reverse(self) -> Matrix[M_co, N_co, T_co]:
        """Return a reversed view of the matrix"""
        return self.rotate(2)

    def materialize(self) -> Matrix[M_co, N_co, T_co]:
        """Return a materialized copy of the matrix

        Certain methods internally produce a view onto an existing sequence to
        preserve memory. As views "stack" onto one another, access times can
        become slower.

        This method addresses said issue by shallowly placing the elements into
        a new sequence - a process that we call "materialization". The
        resulting matrix instance will have access times identical to that of
        an instance created from an array-and-shape pairing, but note that this
        may consume significant amounts of memory (depending on the size of the
        matrix).

        If the matrix does not store a kind of view, this method returns a
        matrix that is semantically equivalent to the original. We call such
        matrices "materialized", as they store a sequence, or reference to a
        sequence, whose elements already exist in the desired arrangement.
        """
        return Matrix(self.data.materialize())

    @overload
    def n(self, by: Literal[Rule.ROW]) -> M_co: ...
    @overload
    def n(self, by: Literal[Rule.COL]) -> N_co: ...
    @overload
    def n(self, by: Rule) -> Union[M_co, N_co]: ...

    def n(self, by):
        """Return the dimension corresponding to the given ``Rule``"""
        return self.data.n(by)

    def values(self, *, by: Rule = Rule.ROW, reverse: bool = False) -> Iterator[T_co]:
        """Return an iterator over the values of the matrix in row or
        column-major order
        """
        return self.data.values(by=by, reverse=reverse)

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[Matrix[Literal[1], N_co, T_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[Matrix[M_co, Literal[1], T_co]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[Matrix[Any, Any, T_co]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[Matrix[Literal[1], N_co, T_co]]: ...

    def slices(self, *, by=Rule.ROW, reverse=False):
        """Return an iterator over the rows or columns of the matrix"""
        return map(Matrix, self.data.slices(by=by, reverse=reverse))

    @overload
    def equal(self, other: Matrix[M_co, N_co, object]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def equal(self, other: object) -> IntegerMatrix[M_co, N_co, bool]: ...

    def equal(self, other):
        """Return element-wise ``a == b``"""
        if isinstance(other, Matrix):
            b = other
        else:
            b = itertools.repeat(other)
        a = self
        return IntegerMatrix(
            array=map(operator.__eq__, a, b),
            shape=a.shape,
        )

    @overload
    def not_equal(self, other: Matrix[M_co, N_co, object]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def not_equal(self, other: object) -> IntegerMatrix[M_co, N_co, bool]: ...

    def not_equal(self, other):
        """Return element-wise ``a != b``"""
        if isinstance(other, Matrix):
            b = other
        else:
            b = itertools.repeat(other)
        a = self
        return IntegerMatrix(
            array=map(operator.__ne__, a, b),
            shape=a.shape,
        )


class ComplexMatrix(Matrix[M_co, N_co, C_co]):

    __slots__ = ()

    @overload
    def __getitem__(self, key: int) -> C_co: ...
    @overload
    def __getitem__(self, key: slice) -> ComplexMatrix[Literal[1], Any, C_co]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> C_co: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> ComplexMatrix[Literal[1], Any, C_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> ComplexMatrix[Any, Literal[1], C_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> ComplexMatrix[Any, Any, C_co]: ...

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, slice):
            return ComplexMatrix(result)
        if isinstance(key, tuple) and (isinstance(key[0], slice) or isinstance(key[1], slice)):
            return ComplexMatrix(result)
        return result

    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, int], other: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, float], other: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, complex], other: ComplexMatrix[M_co, N_co, complex]) -> ComplexMatrix[M_co, N_co, complex]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, int], other: int) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, float], other: float) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __add__(self, other):
        """Return element-wise ``a + b``"""
        if isinstance(other, ComplexMatrix):
            b = other
        elif isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return ComplexMatrix(
            array=map(operator.__add__, a, b),
            shape=a.shape,
        )

    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, int], other: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, float], other: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, complex], other: ComplexMatrix[M_co, N_co, complex]) -> ComplexMatrix[M_co, N_co, complex]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, int], other: int) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, float], other: float) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __sub__(self, other):
        """Return element-wise ``a - b``"""
        if isinstance(other, ComplexMatrix):
            b = other
        elif isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return ComplexMatrix(
            array=map(operator.__sub__, a, b),
            shape=a.shape,
        )

    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, int], other: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, float], other: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, complex], other: ComplexMatrix[M_co, N_co, complex]) -> ComplexMatrix[M_co, N_co, complex]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, int], other: int) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, float], other: float) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __mul__(self, other):
        """Return element-wise ``a * b``"""
        if isinstance(other, ComplexMatrix):
            b = other
        elif isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return ComplexMatrix(
            array=map(operator.__mul__, a, b),
            shape=a.shape,
        )

    @overload
    def __matmul__(self: ComplexMatrix[M_co, N_co, int], other: ComplexMatrix[N_co, P_co, int]) -> ComplexMatrix[M_co, P_co, int]: ...
    @overload
    def __matmul__(self: ComplexMatrix[M_co, N_co, float], other: ComplexMatrix[N_co, P_co, float]) -> ComplexMatrix[M_co, P_co, float]: ...
    @overload
    def __matmul__(self: ComplexMatrix[M_co, N_co, complex], other: ComplexMatrix[N_co, P_co, complex]) -> ComplexMatrix[M_co, P_co, complex]: ...

    def __matmul__(self, other):
        """Return the matrix product"""
        if not isinstance(other, ComplexMatrix):
            return NotImplemented

        a =  self
        b = other

        m, n = a.shape
        p, q = b.shape

        if not n:
            return ComplexMatrix((0,) * (m * q), (m, q))

        def matrix_product(a, b):

            def vector_product(a, b):
                return sum(map(operator.mul, a, b))

            get_a = a.__getitem__
            get_b = b.__getitem__

            mn = m * n
            pq = p * q

            ix = range(0, mn, n)
            jx = range(0,  q, 1)
            for i in ix:
                kx = range(i, i + n, 1)
                for j in jx:
                    lx = range(j, j + pq, q)
                    yield vector_product(
                        map(get_a, kx),
                        map(get_b, lx),
                    )

        return ComplexMatrix(matrix_product(a, b), (m, q))

    @overload
    def __truediv__(self: ComplexMatrix[M_co, N_co, float], other: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __truediv__(self: ComplexMatrix[M_co, N_co, complex], other: ComplexMatrix[M_co, N_co, complex]) -> ComplexMatrix[M_co, N_co, complex]: ...
    @overload
    def __truediv__(self: ComplexMatrix[M_co, N_co, float], other: float) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __truediv__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __truediv__(self, other):
        """Return element-wise ``a / b``"""
        if isinstance(other, ComplexMatrix):
            b = other
        elif isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return ComplexMatrix(
            array=map(operator.__truediv__, a, b),
            shape=a.shape,
        )

    @overload
    def __radd__(self: ComplexMatrix[M_co, N_co, int], other: int) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __radd__(self: ComplexMatrix[M_co, N_co, float], other: float) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __radd__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __radd__(self, other):
        """Return element-wise ``b + a``"""
        if isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return ComplexMatrix(
            array=map(operator.__add__, b, a),
            shape=a.shape,
        )

    @overload
    def __rsub__(self: ComplexMatrix[M_co, N_co, int], other: int) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __rsub__(self: ComplexMatrix[M_co, N_co, float], other: float) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __rsub__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __rsub__(self, other):
        """Return element-wise ``b - a``"""
        if isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return ComplexMatrix(
            array=map(operator.__sub__, b, a),
            shape=a.shape,
        )

    @overload
    def __rmul__(self: ComplexMatrix[M_co, N_co, int], other: int) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __rmul__(self: ComplexMatrix[M_co, N_co, float], other: float) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __rmul__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __rmul__(self, other):
        """Return element-wise ``b * a``"""
        if isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return ComplexMatrix(
            array=map(operator.__mul__, b, a),
            shape=a.shape,
        )

    @overload
    def __rtruediv__(self: ComplexMatrix[M_co, N_co, float], other: float) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __rtruediv__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __rtruediv__(self, other):
        """Return element-wise ``b / a``"""
        if isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return ComplexMatrix(
            array=map(operator.__truediv__, b, a),
            shape=a.shape,
        )

    @overload
    def __neg__(self: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __neg__(self: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __neg__(self: ComplexMatrix[M_co, N_co, complex]) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __neg__(self):
        """Return element-wise ``-a``"""
        a = self
        return ComplexMatrix(
            array=map(operator.__neg__, a),
            shape=a.shape,
        )

    @overload
    def __pos__(self: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __pos__(self: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __pos__(self: ComplexMatrix[M_co, N_co, complex]) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __pos__(self):
        """Return element-wise ``+a``"""
        return self

    @overload
    def __abs__(self: ComplexMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __abs__(self: ComplexMatrix[M_co, N_co, complex]) -> RealMatrix[M_co, N_co, float]: ...

    def __abs__(self):
        """Return element-wise ``abs(a)``"""
        a = self
        return RealMatrix(
            array=map(abs, a),
            shape=a.shape,
        )

    def transpose(self) -> ComplexMatrix[N_co, M_co, C_co]:
        return ComplexMatrix(super().transpose())

    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrix[M_co, N_co, C_co]:
        return ComplexMatrix(super().flip(by=by))

    @overload
    def rotate(self, n: EvenNumber) -> ComplexMatrix[M_co, N_co, C_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> ComplexMatrix[N_co, M_co, C_co]: ...
    @overload
    def rotate(self, n: int) -> Union[ComplexMatrix[M_co, N_co, C_co], ComplexMatrix[N_co, M_co, C_co]]: ...
    @overload
    def rotate(self) -> ComplexMatrix[N_co, M_co, C_co]: ...

    def rotate(self, n=1):
        return ComplexMatrix(super().rotate(n))

    def reverse(self) -> ComplexMatrix[M_co, N_co, C_co]:
        return ComplexMatrix(super().reverse())

    def materialize(self) -> ComplexMatrix[M_co, N_co, C_co]:
        return ComplexMatrix(super().materialize())

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[ComplexMatrix[Literal[1], N_co, C_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[ComplexMatrix[M_co, Literal[1], C_co]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[ComplexMatrix[Any, Any, C_co]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[ComplexMatrix[Literal[1], N_co, C_co]]: ...

    def slices(self, *, by=Rule.ROW, reverse=False):
        return map(ComplexMatrix, super().slices(by=by, reverse=reverse))

    @overload
    def conjugate(self: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def conjugate(self: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def conjugate(self: ComplexMatrix[M_co, N_co, complex]) -> ComplexMatrix[M_co, N_co, complex]: ...

    def conjugate(self):
        """Return element-wise ``a.conjugate()``"""
        a = self
        return ComplexMatrix(
            array=map(lambda x: x.conjugate(), a),
            shape=a.shape,
        )

    @overload
    def transjugate(self: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[N_co, M_co, int]: ...
    @overload
    def transjugate(self: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[N_co, M_co, float]: ...
    @overload
    def transjugate(self: ComplexMatrix[M_co, N_co, complex]) -> ComplexMatrix[N_co, M_co, complex]: ...

    def transjugate(self):
        """Return the conjugate transpose

        The returned matrix may or may not be a view depending on the active
        class.
        """
        return self.transpose().conjugate()


class RealMatrix(ComplexMatrix[M_co, N_co, R_co]):

    __slots__ = ()

    def __lt__(self, other: RealMatrix[Any, Any, float]) -> bool:
        """Return true if lexicographic ``a < b``, otherwise false"""
        if isinstance(other, RealMatrix):
            return self.compare(other) < 0
        return NotImplemented

    def __le__(self, other: RealMatrix[Any, Any, float]) -> bool:
        """Return true if lexicographic ``a <= b``, otherwise false"""
        if isinstance(other, RealMatrix):
            return self.compare(other) <= 0
        return NotImplemented

    def __gt__(self, other: RealMatrix[Any, Any, float]) -> bool:
        """Return true if lexicographic ``a > b``, otherwise false"""
        if isinstance(other, RealMatrix):
            return self.compare(other) > 0
        return NotImplemented

    def __ge__(self, other: RealMatrix[Any, Any, float]) -> bool:
        """Return true if lexicographic ``a >= b``, otherwise false"""
        if isinstance(other, RealMatrix):
            return self.compare(other) >= 0
        return NotImplemented

    @overload
    def __getitem__(self, key: int) -> R_co: ...
    @overload
    def __getitem__(self, key: slice) -> RealMatrix[Literal[1], Any, R_co]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> R_co: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> RealMatrix[Literal[1], Any, R_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> RealMatrix[Any, Literal[1], R_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> RealMatrix[Any, Any, R_co]: ...

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, slice):
            return RealMatrix(result)
        if isinstance(key, tuple) and (isinstance(key[0], slice) or isinstance(key[1], slice)):
            return RealMatrix(result)
        return result

    @overload  # type: ignore[override]
    def __add__(self: RealMatrix[M_co, N_co, int], other: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __add__(self: RealMatrix[M_co, N_co, float], other: RealMatrix[M_co, N_co, float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, int], other: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, float], other: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, complex], other: ComplexMatrix[M_co, N_co, complex]) -> ComplexMatrix[M_co, N_co, complex]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, int], other: int) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __add__(self, other):
        result = super().__add__(other)
        if isinstance(other, (RealMatrix, float, int)):
            return RealMatrix(result)
        return result

    @overload  # type: ignore[override]
    def __sub__(self: RealMatrix[M_co, N_co, int], other: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __sub__(self: RealMatrix[M_co, N_co, float], other: RealMatrix[M_co, N_co, float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, int], other: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, float], other: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, complex], other: ComplexMatrix[M_co, N_co, complex]) -> ComplexMatrix[M_co, N_co, complex]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, int], other: int) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __sub__(self, other):
        result = super().__sub__(other)
        if isinstance(other, (RealMatrix, float, int)):
            return RealMatrix(result)
        return result

    @overload  # type: ignore[override]
    def __mul__(self: RealMatrix[M_co, N_co, int], other: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __mul__(self: RealMatrix[M_co, N_co, float], other: RealMatrix[M_co, N_co, float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, int], other: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, float], other: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, complex], other: ComplexMatrix[M_co, N_co, complex]) -> ComplexMatrix[M_co, N_co, complex]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, int], other: int) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __mul__(self, other):
        result = super().__mul__(other)
        if isinstance(other, (RealMatrix, float, int)):
            return RealMatrix(result)
        return result

    @overload  # type: ignore[override]
    def __matmul__(self: RealMatrix[M_co, N_co, int], other: RealMatrix[N_co, P_co, int]) -> RealMatrix[M_co, P_co, int]: ...
    @overload
    def __matmul__(self: RealMatrix[M_co, N_co, float], other: RealMatrix[N_co, P_co, float]) -> RealMatrix[M_co, P_co, float]: ...
    @overload
    def __matmul__(self: ComplexMatrix[M_co, N_co, int], other: ComplexMatrix[N_co, P_co, int]) -> ComplexMatrix[M_co, P_co, int]: ...
    @overload
    def __matmul__(self: ComplexMatrix[M_co, N_co, float], other: ComplexMatrix[N_co, P_co, float]) -> ComplexMatrix[M_co, P_co, float]: ...
    @overload
    def __matmul__(self: ComplexMatrix[M_co, N_co, complex], other: ComplexMatrix[N_co, P_co, complex]) -> ComplexMatrix[M_co, P_co, complex]: ...

    def __matmul__(self, other):
        result = super().__matmul__(other)
        if isinstance(other, RealMatrix):
            return RealMatrix(result)
        return result

    @overload  # type: ignore[override]
    def __truediv__(self: RealMatrix[M_co, N_co, float], other: RealMatrix[M_co, N_co, float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __truediv__(self: ComplexMatrix[M_co, N_co, float], other: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __truediv__(self: ComplexMatrix[M_co, N_co, complex], other: ComplexMatrix[M_co, N_co, complex]) -> ComplexMatrix[M_co, N_co, complex]: ...
    @overload
    def __truediv__(self: ComplexMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __truediv__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __truediv__(self, other):
        result = super().__truediv__(other)
        if isinstance(other, (RealMatrix, float, int)):
            return RealMatrix(result)
        return result

    @overload
    def __floordiv__(self: RealMatrix[M_co, N_co, int], other: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __floordiv__(self: RealMatrix[M_co, N_co, float], other: RealMatrix[M_co, N_co, float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __floordiv__(self: RealMatrix[M_co, N_co, int], other: int) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __floordiv__(self: RealMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...

    def __floordiv__(self, other):
        """Return element-wise ``a // b``"""
        if isinstance(other, RealMatrix):
            b = other
        elif isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return RealMatrix(
            array=map(operator.__floordiv__, a, b),
            shape=a.shape,
        )

    @overload
    def __mod__(self: RealMatrix[M_co, N_co, int], other: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __mod__(self: RealMatrix[M_co, N_co, float], other: RealMatrix[M_co, N_co, float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __mod__(self: RealMatrix[M_co, N_co, int], other: int) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __mod__(self: RealMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...

    def __mod__(self, other):
        """Return element-wise ``a % b``"""
        if isinstance(other, RealMatrix):
            b = other
        elif isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return RealMatrix(
            array=map(operator.__mod__, a, b),
            shape=a.shape,
        )

    @overload
    def __divmod__(self: RealMatrix[M_co, N_co, int], other: RealMatrix[M_co, N_co, int]) -> tuple[RealMatrix[M_co, N_co, int], RealMatrix[M_co, N_co, int]]: ...
    @overload
    def __divmod__(self: RealMatrix[M_co, N_co, float], other: RealMatrix[M_co, N_co, float]) -> tuple[RealMatrix[M_co, N_co, float], RealMatrix[M_co, N_co, float]]: ...
    @overload
    def __divmod__(self: RealMatrix[M_co, N_co, int], other: int) -> tuple[RealMatrix[M_co, N_co, int], RealMatrix[M_co, N_co, int]]: ...
    @overload
    def __divmod__(self: RealMatrix[M_co, N_co, float], other: float) -> tuple[RealMatrix[M_co, N_co, float], RealMatrix[M_co, N_co, float]]: ...

    def __divmod__(self, other):
        """Return element-wise ``divmod(a, b)``"""
        if isinstance(other, RealMatrix):
            b = other
        elif isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        c, d = itertools.tee(map(divmod, a, b))
        return (
            RealMatrix(
                array=map(operator.itemgetter(0), c),
                shape=a.shape,
            ),
            RealMatrix(
                array=map(operator.itemgetter(1), d),
                shape=a.shape,
            ),
        )

    @overload
    def __radd__(self: ComplexMatrix[M_co, N_co, int], other: int) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __radd__(self: ComplexMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __radd__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __radd__(self, other):
        result = super().__radd__(other)
        if isinstance(other, (float, int)):
            return RealMatrix(result)
        return result

    @overload
    def __rsub__(self: ComplexMatrix[M_co, N_co, int], other: int) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __rsub__(self: ComplexMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __rsub__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __rsub__(self, other):
        result = super().__rsub__(other)
        if isinstance(other, (float, int)):
            return RealMatrix(result)
        return result

    @overload
    def __rmul__(self: ComplexMatrix[M_co, N_co, int], other: int) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __rmul__(self: ComplexMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __rmul__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __rmul__(self, other):
        result = super().__rmul__(other)
        if isinstance(other, (float, int)):
            return RealMatrix(result)
        return result

    @overload
    def __rtruediv__(self: ComplexMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __rtruediv__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __rtruediv__(self, other):
        result = super().__rtruediv__(other)
        if isinstance(other, (float, int)):
            return RealMatrix(result)
        return result

    @overload
    def __rfloordiv__(self: RealMatrix[M_co, N_co, int], other: int) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __rfloordiv__(self: RealMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...

    def __rfloordiv__(self, other):
        """Return element-wise ``b // a``"""
        if isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return RealMatrix(
            array=map(operator.__floordiv__, b, a),
            shape=a.shape,
        )

    @overload
    def __rmod__(self: RealMatrix[M_co, N_co, int], other: int) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __rmod__(self: RealMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...

    def __rmod__(self, other):
        """Return element-wise ``b % a``"""
        if isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return RealMatrix(
            array=map(operator.__mod__, b, a),
            shape=a.shape,
        )

    @overload
    def __rdivmod__(self: RealMatrix[M_co, N_co, int], other: int) -> tuple[RealMatrix[M_co, N_co, int], RealMatrix[M_co, N_co, int]]: ...
    @overload
    def __rdivmod__(self: RealMatrix[M_co, N_co, float], other: float) -> tuple[RealMatrix[M_co, N_co, float], RealMatrix[M_co, N_co, float]]: ...

    def __rdivmod__(self, other):
        """Return element-wise ``divmod(b, a)``"""
        if isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        c, d = itertools.tee(map(divmod, b, a))
        return (
            RealMatrix(
                array=map(operator.itemgetter(0), c),
                shape=a.shape,
            ),
            RealMatrix(
                array=map(operator.itemgetter(1), d),
                shape=a.shape,
            ),
        )

    @overload
    def __neg__(self: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __neg__(self: RealMatrix[M_co, N_co, float]) -> RealMatrix[M_co, N_co, float]: ...

    def __neg__(self):
        return RealMatrix(super().__neg__())

    @overload
    def __pos__(self: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __pos__(self: RealMatrix[M_co, N_co, float]) -> RealMatrix[M_co, N_co, float]: ...

    def __pos__(self):
        return RealMatrix(super().__pos__())

    def transpose(self) -> RealMatrix[N_co, M_co, R_co]:
        return RealMatrix(super().transpose())

    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrix[M_co, N_co, R_co]:
        return RealMatrix(super().flip(by=by))

    @overload
    def rotate(self, n: EvenNumber) -> RealMatrix[M_co, N_co, R_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> RealMatrix[N_co, M_co, R_co]: ...
    @overload
    def rotate(self, n: int) -> Union[RealMatrix[M_co, N_co, R_co], RealMatrix[N_co, M_co, R_co]]: ...
    @overload
    def rotate(self) -> RealMatrix[N_co, M_co, R_co]: ...

    def rotate(self, n=1):
        return RealMatrix(super().rotate(n))

    def reverse(self) -> RealMatrix[M_co, N_co, R_co]:
        return RealMatrix(super().reverse())

    def materialize(self) -> RealMatrix[M_co, N_co, R_co]:
        return RealMatrix(super().materialize())

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[RealMatrix[Literal[1], N_co, R_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[RealMatrix[M_co, Literal[1], R_co]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[RealMatrix[Any, Any, R_co]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[RealMatrix[Literal[1], N_co, R_co]]: ...

    def slices(self, *, by=Rule.ROW, reverse=False):
        return map(RealMatrix, super().slices(by=by, reverse=reverse))

    @overload
    def conjugate(self: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def conjugate(self: RealMatrix[M_co, N_co, float]) -> RealMatrix[M_co, N_co, float]: ...

    def conjugate(self):
        return self

    @overload
    def transjugate(self: RealMatrix[M_co, N_co, int]) -> RealMatrix[N_co, M_co, int]: ...
    @overload
    def transjugate(self: RealMatrix[M_co, N_co, float]) -> RealMatrix[N_co, M_co, float]: ...

    def transjugate(self):
        return RealMatrix(super().transjugate())

    def compare(self, other: RealMatrix[Any, Any, float]) -> Literal[-1, 0, 1]:
        """Return literal ``-1``, ``0``, or ``+1`` if the matrix
        lexicographically compares less than, equal, or greater than ``other``,
        respectively

        Matrices are lexicographically compared values first, shapes second -
        similar to how built-in sequences compare values first, lengths second.
        """
        def inner(a: Iterable[float], b: Iterable[float]) -> Literal[-1, 0, 1]:
            if a is b:
                return 0
            for x, y in zip(a, b):
                if x is y or x == y:
                    continue
                return -1 if x < y else 1
            return 0
        return inner(self, other) or inner(self.shape, other.shape)

    @overload
    def lesser(self, other: RealMatrix[M_co, N_co, float]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def lesser(self, other: float) -> IntegerMatrix[M_co, N_co, bool]: ...

    def lesser(self, other):
        """Return element-wise ``a < b``"""
        if isinstance(other, RealMatrix):
            b = other
        else:
            b = itertools.repeat(other)
        a = self
        return IntegerMatrix(
            array=map(operator.__lt__, a, b),
            shape=a.shape,
        )

    @overload
    def lesser_equal(self, other: RealMatrix[M_co, N_co, float]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def lesser_equal(self, other: float) -> IntegerMatrix[M_co, N_co, bool]: ...

    def lesser_equal(self, other):
        """Return element-wise ``a <= b``"""
        if isinstance(other, RealMatrix):
            b = other
        else:
            b = itertools.repeat(other)
        a = self
        return IntegerMatrix(
            array=map(operator.__le__, a, b),
            shape=a.shape,
        )

    @overload
    def greater(self, other: RealMatrix[M_co, N_co, float]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def greater(self, other: float) -> IntegerMatrix[M_co, N_co, bool]: ...

    def greater(self, other):
        """Return element-wise ``a > b``"""
        if isinstance(other, RealMatrix):
            b = other
        else:
            b = itertools.repeat(other)
        a = self
        return IntegerMatrix(
            array=map(operator.__gt__, a, b),
            shape=a.shape,
        )

    @overload
    def greater_equal(self, other: RealMatrix[M_co, N_co, float]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def greater_equal(self, other: float) -> IntegerMatrix[M_co, N_co, bool]: ...

    def greater_equal(self, other):
        """Return element-wise ``a >= b``"""
        if isinstance(other, RealMatrix):
            b = other
        else:
            b = itertools.repeat(other)
        a = self
        return IntegerMatrix(
            array=map(operator.__ge__, a, b),
            shape=a.shape,
        )


class IntegerMatrix(RealMatrix[M_co, N_co, I_co]):

    __slots__ = ()

    @overload
    def __getitem__(self, key: int) -> I_co: ...
    @overload
    def __getitem__(self, key: slice) -> IntegerMatrix[Literal[1], Any, I_co]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> I_co: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> IntegerMatrix[Literal[1], Any, I_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> IntegerMatrix[Any, Literal[1], I_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> IntegerMatrix[Any, Any, I_co]: ...

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, slice):
            return IntegerMatrix(result)
        if isinstance(key, tuple) and (isinstance(key[0], slice) or isinstance(key[1], slice)):
            return IntegerMatrix(result)
        return result

    @overload  # type: ignore[override]
    def __add__(self: IntegerMatrix[M_co, N_co, int], other: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __add__(self: RealMatrix[M_co, N_co, int], other: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __add__(self: RealMatrix[M_co, N_co, float], other: RealMatrix[M_co, N_co, float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, int], other: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, float], other: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, complex], other: ComplexMatrix[M_co, N_co, complex]) -> ComplexMatrix[M_co, N_co, complex]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __add__(self, other):
        result = super().__add__(other)
        if isinstance(other, (IntegerMatrix, int)):
            return IntegerMatrix(result)
        return result

    @overload  # type: ignore[override]
    def __sub__(self: IntegerMatrix[M_co, N_co, int], other: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __sub__(self: RealMatrix[M_co, N_co, int], other: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __sub__(self: RealMatrix[M_co, N_co, float], other: RealMatrix[M_co, N_co, float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, int], other: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, float], other: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, complex], other: ComplexMatrix[M_co, N_co, complex]) -> ComplexMatrix[M_co, N_co, complex]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __sub__(self, other):
        result = super().__sub__(other)
        if isinstance(other, (IntegerMatrix, int)):
            return IntegerMatrix(result)
        return result

    @overload  # type: ignore[override]
    def __mul__(self: IntegerMatrix[M_co, N_co, int], other: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __mul__(self: RealMatrix[M_co, N_co, int], other: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __mul__(self: RealMatrix[M_co, N_co, float], other: RealMatrix[M_co, N_co, float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, int], other: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[M_co, N_co, int]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, float], other: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, complex], other: ComplexMatrix[M_co, N_co, complex]) -> ComplexMatrix[M_co, N_co, complex]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __mul__(self, other):
        result = super().__mul__(other)
        if isinstance(other, (IntegerMatrix, int)):
            return IntegerMatrix(result)
        return result

    @overload  # type: ignore[override]
    def __matmul__(self: IntegerMatrix[M_co, N_co, int], other: IntegerMatrix[N_co, P_co, int]) -> IntegerMatrix[M_co, P_co, int]: ...
    @overload
    def __matmul__(self: RealMatrix[M_co, N_co, int], other: RealMatrix[N_co, P_co, int]) -> RealMatrix[M_co, P_co, int]: ...
    @overload
    def __matmul__(self: RealMatrix[M_co, N_co, float], other: RealMatrix[N_co, P_co, float]) -> RealMatrix[M_co, P_co, float]: ...
    @overload
    def __matmul__(self: ComplexMatrix[M_co, N_co, int], other: ComplexMatrix[N_co, P_co, int]) -> ComplexMatrix[M_co, P_co, int]: ...
    @overload
    def __matmul__(self: ComplexMatrix[M_co, N_co, float], other: ComplexMatrix[N_co, P_co, float]) -> ComplexMatrix[M_co, P_co, float]: ...
    @overload
    def __matmul__(self: ComplexMatrix[M_co, N_co, complex], other: ComplexMatrix[N_co, P_co, complex]) -> ComplexMatrix[M_co, P_co, complex]: ...

    def __matmul__(self, other):
        result = super().__matmul__(other)
        if isinstance(other, IntegerMatrix):
            return IntegerMatrix(result)
        return result

    @overload  # type: ignore[override]
    def __floordiv__(self: IntegerMatrix[M_co, N_co, int], other: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __floordiv__(self: RealMatrix[M_co, N_co, int], other: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __floordiv__(self: RealMatrix[M_co, N_co, float], other: RealMatrix[M_co, N_co, float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __floordiv__(self: RealMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __floordiv__(self: RealMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...

    def __floordiv__(self, other):
        result = super().__floordiv__(other)
        if isinstance(other, (IntegerMatrix, int)):
            return IntegerMatrix(result)
        return result

    @overload  # type: ignore[override]
    def __mod__(self: IntegerMatrix[M_co, N_co, int], other: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __mod__(self: RealMatrix[M_co, N_co, int], other: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...
    @overload
    def __mod__(self: RealMatrix[M_co, N_co, float], other: RealMatrix[M_co, N_co, float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __mod__(self: RealMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __mod__(self: RealMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...

    def __mod__(self, other):
        result = super().__mod__(other)
        if isinstance(other, (IntegerMatrix, int)):
            return IntegerMatrix(result)
        return result

    @overload  # type: ignore[override]
    def __divmod__(self: IntegerMatrix[M_co, N_co, int], other: IntegerMatrix[M_co, N_co, int]) -> tuple[IntegerMatrix[M_co, N_co, int], IntegerMatrix[M_co, N_co, int]]: ...
    @overload
    def __divmod__(self: RealMatrix[M_co, N_co, int], other: RealMatrix[M_co, N_co, int]) -> tuple[RealMatrix[M_co, N_co, int], RealMatrix[M_co, N_co, int]]: ...
    @overload
    def __divmod__(self: RealMatrix[M_co, N_co, float], other: RealMatrix[M_co, N_co, float]) -> tuple[RealMatrix[M_co, N_co, float], RealMatrix[M_co, N_co, float]]: ...
    @overload
    def __divmod__(self: RealMatrix[M_co, N_co, int], other: int) -> tuple[IntegerMatrix[M_co, N_co, int], IntegerMatrix[M_co, N_co, int]]: ...
    @overload
    def __divmod__(self: RealMatrix[M_co, N_co, float], other: float) -> tuple[RealMatrix[M_co, N_co, float], RealMatrix[M_co, N_co, float]]: ...

    def __divmod__(self, other):
        result = super().__divmod__(other)
        if isinstance(other, (IntegerMatrix, int)):
            return tuple(map(IntegerMatrix, result))
        return result

    @overload
    def __lshift__(self: IntegerMatrix[M_co, N_co, int], other: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __lshift__(self: IntegerMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...

    def __lshift__(self, other):
        """Return element-wise ``a << b``"""
        if isinstance(other, IntegerMatrix):
            b = other
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return IntegerMatrix(
            array=map(operator.__lshift__, a, b),
            shape=a.shape,
        )

    @overload
    def __rshift__(self: IntegerMatrix[M_co, N_co, int], other: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __rshift__(self: IntegerMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...

    def __rshift__(self, other):
        """Return element-wise ``a >> b``"""
        if isinstance(other, IntegerMatrix):
            b = other
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return IntegerMatrix(
            array=map(operator.__rshift__, a, b),
            shape=a.shape,
        )

    @overload
    def __and__(self: IntegerMatrix[M_co, N_co, bool], other: IntegerMatrix[M_co, N_co, bool]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __and__(self: IntegerMatrix[M_co, N_co, int], other: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __and__(self: IntegerMatrix[M_co, N_co, bool], other: bool) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __and__(self: IntegerMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...

    def __and__(self, other):
        """Return element-wise ``a & b``"""
        if isinstance(other, IntegerMatrix):
            b = other
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return IntegerMatrix(
            array=map(operator.__and__, a, b),
            shape=a.shape,
        )

    @overload
    def __xor__(self: IntegerMatrix[M_co, N_co, bool], other: IntegerMatrix[M_co, N_co, bool]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __xor__(self: IntegerMatrix[M_co, N_co, int], other: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __xor__(self: IntegerMatrix[M_co, N_co, bool], other: bool) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __xor__(self: IntegerMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...

    def __xor__(self, other):
        """Return element-wise ``a ^ b``"""
        if isinstance(other, IntegerMatrix):
            b = other
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return IntegerMatrix(
            array=map(operator.__xor__, a, b),
            shape=a.shape,
        )

    @overload
    def __or__(self: IntegerMatrix[M_co, N_co, bool], other: IntegerMatrix[M_co, N_co, bool]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __or__(self: IntegerMatrix[M_co, N_co, int], other: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __or__(self: IntegerMatrix[M_co, N_co, bool], other: bool) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __or__(self: IntegerMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...

    def __or__(self, other):
        """Return element-wise ``a | b``"""
        if isinstance(other, IntegerMatrix):
            b = other
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return IntegerMatrix(
            array=map(operator.__or__, a, b),
            shape=a.shape,
        )

    @overload
    def __radd__(self: ComplexMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __radd__(self: ComplexMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __radd__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __radd__(self, other):
        result = super().__radd__(other)
        if isinstance(other, int):
            return IntegerMatrix(result)
        return result

    @overload
    def __rsub__(self: ComplexMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __rsub__(self: ComplexMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __rsub__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __rsub__(self, other):
        result = super().__rsub__(other)
        if isinstance(other, int):
            return IntegerMatrix(result)
        return result

    @overload
    def __rmul__(self: ComplexMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __rmul__(self: ComplexMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __rmul__(self: ComplexMatrix[M_co, N_co, complex], other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __rmul__(self, other):
        result = super().__rmul__(other)
        if isinstance(other, int):
            return IntegerMatrix(result)
        return result

    @overload
    def __rfloordiv__(self: RealMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __rfloordiv__(self: RealMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...

    def __rfloordiv__(self, other):
        result = super().__rfloordiv__(other)
        if isinstance(other, int):
            return IntegerMatrix(result)
        return result

    @overload
    def __rmod__(self: RealMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __rmod__(self: RealMatrix[M_co, N_co, float], other: float) -> RealMatrix[M_co, N_co, float]: ...

    def __rmod__(self, other):
        result = super().__rmod__(other)
        if isinstance(other, int):
            return IntegerMatrix(result)
        return result

    @overload
    def __rdivmod__(self: RealMatrix[M_co, N_co, int], other: int) -> tuple[IntegerMatrix[M_co, N_co, int], IntegerMatrix[M_co, N_co, int]]: ...
    @overload
    def __rdivmod__(self: RealMatrix[M_co, N_co, float], other: float) -> tuple[RealMatrix[M_co, N_co, float], RealMatrix[M_co, N_co, float]]: ...

    def __rdivmod__(self, other):
        result = super().__rdivmod__(other)
        if isinstance(other, int):
            return tuple(map(IntegerMatrix, result))
        return result

    def __rlshift__(self: IntegerMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]:
        """Return element-wise ``b << a``"""
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return IntegerMatrix(
            array=map(operator.__lshift__, b, a),
            shape=a.shape,
        )

    def __rrshift__(self: IntegerMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]:
        """Return element-wise ``b >> a``"""
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return IntegerMatrix(
            array=map(operator.__rshift__, b, a),
            shape=a.shape,
        )

    @overload
    def __rand__(self: IntegerMatrix[M_co, N_co, bool], other: bool) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __rand__(self: IntegerMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...

    def __rand__(self, other):
        """Return element-wise ``b & a``"""
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return IntegerMatrix(
            array=map(operator.__and__, b, a),
            shape=a.shape,
        )

    @overload
    def __rxor__(self: IntegerMatrix[M_co, N_co, bool], other: bool) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __rxor__(self: IntegerMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...

    def __rxor__(self, other):
        """Return element-wise ``b ^ a``"""
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return IntegerMatrix(
            array=map(operator.__xor__, b, a),
            shape=a.shape,
        )

    @overload
    def __ror__(self: IntegerMatrix[M_co, N_co, bool], other: bool) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __ror__(self: IntegerMatrix[M_co, N_co, int], other: int) -> IntegerMatrix[M_co, N_co, int]: ...

    def __ror__(self, other):
        """Return element-wise ``b | a``"""
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        a = self
        return IntegerMatrix(
            array=map(operator.__or__, b, a),
            shape=a.shape,
        )

    def __neg__(self: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[M_co, N_co, int]:
        return IntegerMatrix(super().__neg__())

    def __pos__(self: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[M_co, N_co, int]:
        return IntegerMatrix(super().__pos__())

    def __abs__(self: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[M_co, N_co, int]:
        return IntegerMatrix(super().__abs__())

    @overload
    def __invert__(self: IntegerMatrix[M_co, N_co, bool]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __invert__(self: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[M_co, N_co, int]: ...

    def __invert__(self):
        """Return element-wise ``~a``"""
        a = self
        return IntegerMatrix(
            array=map(operator.__invert__, a),
            shape=a.shape,
        )

    def transpose(self) -> IntegerMatrix[N_co, M_co, I_co]:
        return IntegerMatrix(super().transpose())

    def flip(self, *, by: Rule = Rule.ROW) -> IntegerMatrix[M_co, N_co, I_co]:
        return IntegerMatrix(super().flip(by=by))

    @overload
    def rotate(self, n: EvenNumber) -> IntegerMatrix[M_co, N_co, I_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> IntegerMatrix[N_co, M_co, I_co]: ...
    @overload
    def rotate(self, n: int) -> Union[IntegerMatrix[M_co, N_co, I_co], IntegerMatrix[N_co, M_co, I_co]]: ...
    @overload
    def rotate(self) -> IntegerMatrix[N_co, M_co, I_co]: ...

    def rotate(self, n=1):
        return IntegerMatrix(super().rotate(n))

    def reverse(self) -> IntegerMatrix[M_co, N_co, I_co]:
        return IntegerMatrix(super().reverse())

    def materialize(self) -> IntegerMatrix[M_co, N_co, I_co]:
        return IntegerMatrix(super().materialize())

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[IntegerMatrix[Literal[1], N_co, I_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[IntegerMatrix[M_co, Literal[1], I_co]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[IntegerMatrix[Any, Any, I_co]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[IntegerMatrix[Literal[1], N_co, I_co]]: ...

    def slices(self, *, by=Rule.ROW, reverse=False):
        return map(IntegerMatrix, super().slices(by=by, reverse=reverse))

    def conjugate(self: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[M_co, N_co, int]:
        return IntegerMatrix(super().conjugate())

    def transjugate(self: IntegerMatrix[M_co, N_co, int]) -> IntegerMatrix[N_co, M_co, int]:
        return IntegerMatrix(super().transjugate())
