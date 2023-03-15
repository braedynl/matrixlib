from __future__ import annotations

import itertools
import operator
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Generic, Literal, Optional, TypeVar, Union, overload

from typing_extensions import Self

from .abc import Shaped
from .rule import Rule
from .utilities import AbstractGrid, EvenNumber, Grid, OddNumber

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)
P_co = TypeVar("P_co", covariant=True, bound=int)

C_co = TypeVar("C_co", covariant=True, bound=complex)
R_co = TypeVar("R_co", covariant=True, bound=float)
I_co = TypeVar("I_co", covariant=True, bound=int)


class Matrix(Shaped[M_co, N_co], Sequence[T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("_grid",)
    __match_args__ = ("array", "shape")

    @overload
    def __new__(cls, array: AbstractGrid[M_co, N_co, T_co]) -> Self: ...
    @overload
    def __new__(cls, array: Matrix[M_co, N_co, T_co]) -> Self: ...
    @overload
    def __new__(cls, array: Iterable[T_co] = (), shape: Optional[tuple[M_co, N_co]] = None) -> Self: ...

    def __new__(cls, array=(), shape=None):
        if type(array) is cls:
            return array

        self = super().__new__(cls)

        if isinstance(array, AbstractGrid):
            self._grid = array
        elif isinstance(array, Matrix):
            self._grid = array._grid
        else:
            self._grid = Grid(array, shape)

        return self

    def __repr__(self) -> str:
        """Return a canonical representation of the matrix"""
        temp = self.materialize()
        return f"{self.__class__.__name__}(array={temp.array!r}, shape={temp.shape!r})"

    def __eq__(self, other: object) -> bool:
        """Return true if the two matrices are equal, otherwise false"""
        if self is other:
            return True
        if isinstance(other, Matrix):
            return self.grid == other.grid
        return NotImplemented

    def __hash__(self) -> int:
        """Return a hash of the matrix"""
        return hash(self.grid)

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
        value = self.grid[key]
        if isinstance(value, AbstractGrid):
            return Matrix(value)
        return value

    def __iter__(self) -> Iterator[T_co]:
        """Return an iterator over the values of the matrix in row-major order"""
        return iter(self.grid)

    def __reversed__(self) -> Iterator[T_co]:
        """Return an iterator over the values of the matrix in reverse
        row-major order
        """
        return reversed(self.grid)

    def __contains__(self, value: object) -> bool:
        """Return true if the matrix contains ``value``, otherwise false"""
        return value in self.grid

    def __deepcopy__(self, memo=None) -> Self:
        """Return the matrix"""
        return self

    __copy__ = __deepcopy__

    @property
    def array(self) -> Sequence[T_co]:
        return self.grid.array

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self.grid.shape

    @property
    def grid(self) -> AbstractGrid[M_co, N_co, T_co]:
        return self._grid  # type: ignore[attr-defined]

    def transpose(self) -> Matrix[N_co, M_co, T_co]:
        """Return a transposed view of the matrix"""
        return Matrix(self.grid.transpose())

    def flip(self, *, by: Rule = Rule.ROW) -> Matrix[M_co, N_co, T_co]:
        """Return a flipped view of the matrix"""
        return Matrix(self.grid.flip(by=by))

    @overload
    def rotate(self, n: EvenNumber) -> Matrix[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> Matrix[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Union[Matrix[M_co, N_co, T_co], Matrix[N_co, M_co, T_co]]: ...
    @overload
    def rotate(self) -> Matrix[N_co, M_co, T_co]: ...

    def rotate(self, n=1):
        """Return a rotated view of the matrix"""
        return Matrix(self.grid.rotate(n))

    def reverse(self) -> Matrix[M_co, N_co, T_co]:
        """Return a reversed view of the matrix"""
        return self.rotate(2)

    def materialize(self) -> Matrix[M_co, N_co, T_co]:
        """Return a materialized copy of the matrix

        Certain methods may, internally, produce a view onto an existing
        sequence to preserve memory. As views "stack" onto one another, access
        times can become slower.

        This method addresses said issue by copying the elements into a new
        sequence - a process that we call "materialization". The resulting
        matrix instance will have access times identical to that of an instance
        created from an array-and-shape pairing, but note that this may consume
        significant amounts of memory (depending on the size of the matrix).

        If the matrix does not store a kind of view, this method returns a
        matrix that is semantically equivalent to the original. We call such
        matrices "materialized", as they store a sequence, or reference to a
        sequence, whose elements already exist in the desired arrangement.
        """
        return Matrix(self.grid.materialize())

    @overload
    def n(self, by: Literal[Rule.ROW]) -> M_co: ...
    @overload
    def n(self, by: Literal[Rule.COL]) -> N_co: ...
    @overload
    def n(self, by: Rule) -> Union[M_co, N_co]: ...

    def n(self, by):
        """Return the dimension corresponding to the given ``Rule``"""
        return self.grid.n(by)

    def values(self, *, by: Rule = Rule.ROW, reverse: bool = False) -> Iterator[T_co]:
        """Return an iterator over the values of the matrix"""
        return self.grid.values(by=by, reverse=reverse)

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
        return map(Matrix, self.grid.slices(by=by, reverse=reverse))

    @overload
    def equal(self, other: Matrix[M_co, N_co, object]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def equal(self, other: object) -> IntegerMatrix[M_co, N_co, bool]: ...

    def equal(self, other):
        """Return element-wise ``a == b``"""
        if isinstance(other, Matrix):
            b = iter(other)
        else:
            b = itertools.repeat(other)
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__eq__, a, b),
            shape=shape,
        )

    @overload
    def not_equal(self, other: Matrix[M_co, N_co, object]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def not_equal(self, other: object) -> IntegerMatrix[M_co, N_co, bool]: ...

    def not_equal(self, other):
        """Return element-wise ``a != b``"""
        if isinstance(other, Matrix):
            b = iter(other)
        else:
            b = itertools.repeat(other)
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__ne__, a, b),
            shape=shape,
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
        value = super().__getitem__(key)
        if isinstance(value, Matrix):
            return ComplexMatrix(value)
        return value

    def __add__(self, other: Union[ComplexMatrix[M_co, N_co, complex], complex]) -> ComplexMatrix[M_co, N_co, complex]:
        if isinstance(other, ComplexMatrix):
            b = iter(other)
        elif isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return ComplexMatrix(
            array=map(operator.__add__, a, b),
            shape=shape,
        )

    def __sub__(self, other: Union[ComplexMatrix[M_co, N_co, complex], complex]) -> ComplexMatrix[M_co, N_co, complex]:
        if isinstance(other, ComplexMatrix):
            b = iter(other)
        elif isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return ComplexMatrix(
            array=map(operator.__sub__, a, b),
            shape=shape,
        )

    def __mul__(self, other: Union[ComplexMatrix[M_co, N_co, complex], complex]) -> ComplexMatrix[M_co, N_co, complex]:
        if isinstance(other, ComplexMatrix):
            b = iter(other)
        elif isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return ComplexMatrix(
            array=map(operator.__mul__, a, b),
            shape=shape,
        )

    def __matmul__(self, other: ComplexMatrix[N_co, P_co, complex]) -> ComplexMatrix[M_co, P_co, complex]:
        m, n =  self.shape
        p, q = other.shape

        if n != p:
            raise ValueError(f"incompatible shapes, ({n = }) != ({p = })")

        if n:
            get_lhs =  self.__getitem__
            get_rhs = other.__getitem__

            def sum_prod(a, b):
                return sum(map(operator.mul, a, b))

            mn = m * n
            pq = p * q

            ix = range(0, mn, n)
            jx = range(0,  q, 1)

            array = tuple(  # XXX: allow Grid to hold any sequence type? list comp for potential speed-up
                sum_prod(
                    map(get_lhs, range(i, i +  n, 1)),
                    map(get_rhs, range(j, j + pq, q)),
                )
                for i in ix
                for j in jx
            )

        else:
            array = (0,) * (m * q)

        shape = (m, q)

        return ComplexMatrix(array, shape)  # TODO: overrides

    def __truediv__(self, other: Union[ComplexMatrix[M_co, N_co, complex], complex]) -> ComplexMatrix[M_co, N_co, complex]:
        if isinstance(other, ComplexMatrix):
            b = iter(other)
        elif isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return ComplexMatrix(
            array=map(operator.__truediv__, a, b),
            shape=shape,
        )

    def __radd__(self, other: complex) -> ComplexMatrix[M_co, N_co, complex]:
        if isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return ComplexMatrix(
            array=map(operator.__add__, b, a),
            shape=shape,
        )

    def __rsub__(self, other: complex) -> ComplexMatrix[M_co, N_co, complex]:
        if isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return ComplexMatrix(
            array=map(operator.__sub__, b, a),
            shape=shape,
        )

    def __rmul__(self, other: complex) -> ComplexMatrix[M_co, N_co, complex]:
        if isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return ComplexMatrix(
            array=map(operator.__mul__, b, a),
            shape=shape,
        )

    def __rtruediv__(self, other: complex) -> ComplexMatrix[M_co, N_co, complex]:
        if isinstance(other, (complex, float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return ComplexMatrix(
            array=map(operator.__truediv__, b, a),
            shape=shape,
        )

    @overload
    def __neg__(self: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[M_co, N_co, int]: ...  # type: ignore[misc]
    @overload
    def __neg__(self: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...  # type: ignore[misc]
    @overload
    def __neg__(self) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __neg__(self):
        shape = self.shape
        a = iter(self)
        return ComplexMatrix(
            array=map(operator.__neg__, a),
            shape=shape,
        )

    @overload
    def __pos__(self: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[M_co, N_co, int]: ...  # type: ignore[misc]
    @overload
    def __pos__(self: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...  # type: ignore[misc]
    @overload
    def __pos__(self) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __pos__(self):
        shape = self.shape
        a = iter(self)
        return ComplexMatrix(
            array=map(operator.__pos__, a),
            shape=shape,
        )

    @overload
    def __abs__(self: ComplexMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...  # type: ignore[misc]
    @overload
    def __abs__(self: ComplexMatrix[M_co, N_co, float]) -> RealMatrix[M_co, N_co, float]: ...  # type: ignore[misc]
    @overload
    def __abs__(self) -> RealMatrix[M_co, N_co, float]: ...

    def __abs__(self):
        shape = self.shape
        a = iter(self)
        return RealMatrix(
            array=map(abs, a),
            shape=shape,
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
    def conjugate(self: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[M_co, N_co, int]: ...  # type: ignore[misc]
    @overload
    def conjugate(self: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[M_co, N_co, float]: ...  # type: ignore[misc]
    @overload
    def conjugate(self) -> ComplexMatrix[M_co, N_co, complex]: ...

    def conjugate(self):
        shape = self.shape
        a = iter(self)
        return ComplexMatrix(
            array=map(lambda x: x.conjugate(), a),
            shape=shape,
        )

    @overload
    def transjugate(self: ComplexMatrix[M_co, N_co, int]) -> ComplexMatrix[N_co, M_co, int]: ...  # type: ignore[misc]
    @overload
    def transjugate(self: ComplexMatrix[M_co, N_co, float]) -> ComplexMatrix[N_co, M_co, float]: ...  # type: ignore[misc]
    @overload
    def transjugate(self) -> ComplexMatrix[N_co, M_co, complex]: ...

    def transjugate(self):
        return self.transpose().conjugate()


class RealMatrix(ComplexMatrix[M_co, N_co, R_co]):

    __slots__ = ()

    def __lt__(self, other: RealMatrix) -> bool:
        """Return true if lexicographic ``a < b``, otherwise false"""
        if isinstance(other, RealMatrix):
            return self.compare(other) < 0
        return NotImplemented

    def __le__(self, other: RealMatrix) -> bool:
        """Return true if lexicographic ``a <= b``, otherwise false"""
        if isinstance(other, RealMatrix):
            return self.compare(other) <= 0
        return NotImplemented

    def __gt__(self, other: RealMatrix) -> bool:
        """Return true if lexicographic ``a > b``, otherwise false"""
        if isinstance(other, RealMatrix):
            return self.compare(other) > 0
        return NotImplemented

    def __ge__(self, other: RealMatrix) -> bool:
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
        value = super().__getitem__(key)
        if isinstance(value, ComplexMatrix):
            return RealMatrix(value)
        return value

    @overload
    def __add__(self, other: Union[RealMatrix[M_co, N_co, float], float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __add__(self, other: Union[ComplexMatrix[M_co, N_co, complex], complex]) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __add__(self, other):
        if isinstance(other, RealMatrix):
            b = iter(other)
        elif isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return super().__add__(other)
        shape = self.shape
        a = iter(self)
        return RealMatrix(
            array=map(operator.__add__, a, b),
            shape=shape,
        )

    @overload
    def __sub__(self, other: Union[RealMatrix[M_co, N_co, float], float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __sub__(self, other: Union[ComplexMatrix[M_co, N_co, complex], complex]) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __sub__(self, other):
        if isinstance(other, RealMatrix):
            b = iter(other)
        elif isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return super().__sub__(other)
        shape = self.shape
        a = iter(self)
        return RealMatrix(
            array=map(operator.__sub__, a, b),
            shape=shape,
        )

    @overload
    def __mul__(self, other: Union[RealMatrix[M_co, N_co, float], float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __mul__(self, other: Union[ComplexMatrix[M_co, N_co, complex], complex]) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __mul__(self, other):
        if isinstance(other, RealMatrix):
            b = iter(other)
        elif isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return super().__mul__(other)
        shape = self.shape
        a = iter(self)
        return RealMatrix(
            array=map(operator.__mul__, a, b),
            shape=shape,
        )

    @overload
    def __truediv__(self, other: Union[RealMatrix[M_co, N_co, float], float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __truediv__(self, other: Union[ComplexMatrix[M_co, N_co, complex], complex]) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __truediv__(self, other):
        if isinstance(other, RealMatrix):
            b = iter(other)
        elif isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return super().__truediv__(other)
        shape = self.shape
        a = iter(self)
        return RealMatrix(
            array=map(operator.__truediv__, a, b),
            shape=shape,
        )

    def __floordiv__(self, other: Union[RealMatrix[M_co, N_co, float], float]) -> RealMatrix[M_co, N_co, float]:
        if isinstance(other, RealMatrix):
            b = iter(other)
        elif isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return RealMatrix(
            array=map(operator.__floordiv__, a, b),
            shape=shape,
        )

    def __mod__(self, other: Union[RealMatrix[M_co, N_co, float], float]) -> RealMatrix[M_co, N_co, float]:
        if isinstance(other, RealMatrix):
            b = iter(other)
        elif isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return RealMatrix(
            array=map(operator.__mod__, a, b),
            shape=shape,
        )

    def __divmod__(self, other: Union[RealMatrix[M_co, N_co, float], float]) -> tuple[RealMatrix[M_co, N_co, float], RealMatrix[M_co, N_co, float]]:
        if isinstance(other, RealMatrix):
            b = iter(other)
        elif isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        c, d = itertools.tee(map(divmod, a, b))  # type: ignore[arg-type]
        return (
            RealMatrix(
                array=map(operator.itemgetter(0), c),  # type: ignore[arg-type]
                shape=shape,
            ),
            RealMatrix(
                array=map(operator.itemgetter(1), d),  # type: ignore[arg-type]
                shape=shape,
            ),
        )

    @overload
    def __radd__(self, other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __radd__(self, other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __radd__(self, other):
        if isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return super().__radd__(other)
        shape = self.shape
        a = iter(self)
        return RealMatrix(
            array=map(operator.__add__, b, a),
            shape=shape,
        )

    @overload
    def __rsub__(self, other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __rsub__(self, other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __rsub__(self, other):
        if isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return super().__rsub__(other)
        shape = self.shape
        a = iter(self)
        return RealMatrix(
            array=map(operator.__sub__, b, a),
            shape=shape,
        )

    @overload
    def __rmul__(self, other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return super().__rmul__(other)
        shape = self.shape
        a = iter(self)
        return RealMatrix(
            array=map(operator.__mul__, b, a),
            shape=shape,
        )

    @overload
    def __rtruediv__(self, other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __rtruediv__(self, other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __rtruediv__(self, other):
        if isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return super().__rtruediv__(other)
        shape = self.shape
        a = iter(self)
        return RealMatrix(
            array=map(operator.__truediv__, b, a),
            shape=shape,
        )

    def __rfloordiv__(self, other: float) -> RealMatrix[M_co, N_co, float]:
        if isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return RealMatrix(
            array=map(operator.__floordiv__, b, a),
            shape=shape,
        )

    def __rmod__(self, other: float) -> RealMatrix[M_co, N_co, float]:
        if isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return RealMatrix(
            array=map(operator.__mod__, b, a),
            shape=shape,
        )

    def __rdivmod__(self, other: float) -> tuple[RealMatrix[M_co, N_co, float], RealMatrix[M_co, N_co, float]]:
        if isinstance(other, (float, int)):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        c, d = itertools.tee(map(divmod, b, a))  # type: ignore[arg-type]
        return (
            RealMatrix(
                array=map(operator.itemgetter(0), c),  # type: ignore[arg-type]
                shape=shape,
            ),
            RealMatrix(
                array=map(operator.itemgetter(1), d),  # type: ignore[arg-type]
                shape=shape,
            ),
        )

    @overload
    def __neg__(self: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...  # type: ignore[misc]
    @overload
    def __neg__(self) -> RealMatrix[M_co, N_co, float]: ...

    def __neg__(self):
        return RealMatrix(super().__neg__())

    @overload
    def __pos__(self: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...  # type: ignore[misc]
    @overload
    def __pos__(self) -> RealMatrix[M_co, N_co, float]: ...

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
    def conjugate(self: RealMatrix[M_co, N_co, int]) -> RealMatrix[M_co, N_co, int]: ...  # type: ignore[misc]
    @overload
    def conjugate(self) -> RealMatrix[M_co, N_co, float]: ...

    def conjugate(self):
        return RealMatrix(super().conjugate())

    @overload
    def transjugate(self: RealMatrix[M_co, N_co, int]) -> RealMatrix[N_co, M_co, int]: ...  # type: ignore[misc]
    @overload
    def transjugate(self) -> RealMatrix[N_co, M_co, float]: ...

    def transjugate(self):
        return RealMatrix(super().transjugate())

    def compare(self, other: RealMatrix) -> Literal[-1, 0, 1]:
        """Return literal -1, 0, or +1 if the matrix lexicographically compares
        less than, equal, or greater than ``other``, respectively

        Matrices are lexicographically compared values first, shapes second -
        similar to how built-in sequences compare values first, lengths second.
        """
        def inner(a, b):
            if a is b:
                return 0
            for x, y in zip(a, b):
                if x is y or x == y:
                    continue
                return -1 if x < y else 1
            return 0
        return inner(self, other) or inner(self.shape, other.shape)


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
        value = super().__getitem__(key)
        if isinstance(value, RealMatrix):
            return IntegerMatrix(value)
        return value

    @overload  # type: ignore[override]
    def __add__(self, other: Union[IntegerMatrix[M_co, N_co, int], int]) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __add__(self, other: Union[RealMatrix[M_co, N_co, float], float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __add__(self, other: Union[ComplexMatrix[M_co, N_co, complex], complex]) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __add__(self, other):
        if isinstance(other, IntegerMatrix):
            b = iter(other)
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return super().__add__(other)
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__add__, a, b),
            shape=shape,
        )

    @overload  # type: ignore[override]
    def __sub__(self, other: Union[IntegerMatrix[M_co, N_co, int], int]) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __sub__(self, other: Union[RealMatrix[M_co, N_co, float], float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __sub__(self, other: Union[ComplexMatrix[M_co, N_co, complex], complex]) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __sub__(self, other):
        if isinstance(other, IntegerMatrix):
            b = iter(other)
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return super().__sub__(other)
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__sub__, a, b),
            shape=shape,
        )

    @overload  # type: ignore[override]
    def __mul__(self, other: Union[IntegerMatrix[M_co, N_co, int], int]) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __mul__(self, other: Union[RealMatrix[M_co, N_co, float], float]) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __mul__(self, other: Union[ComplexMatrix[M_co, N_co, complex], complex]) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __mul__(self, other):
        if isinstance(other, IntegerMatrix):
            b = iter(other)
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return super().__mul__(other)
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__mul__, a, b),
            shape=shape,
        )

    @overload
    def __floordiv__(self, other: Union[IntegerMatrix[M_co, N_co, int], int]) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __floordiv__(self, other: Union[RealMatrix[M_co, N_co, float], float]) -> RealMatrix[M_co, N_co, float]: ...

    def __floordiv__(self, other):
        if isinstance(other, IntegerMatrix):
            b = iter(other)
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return super().__floordiv__(other)
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__floordiv__, a, b),
            shape=shape,
        )

    @overload
    def __mod__(self, other: Union[IntegerMatrix[M_co, N_co, int], int]) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __mod__(self, other: Union[RealMatrix[M_co, N_co, float], float]) -> RealMatrix[M_co, N_co, float]: ...

    def __mod__(self, other):
        if isinstance(other, IntegerMatrix):
            b = iter(other)
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return super().__mod__(other)
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__mod__, a, b),
            shape=shape,
        )

    @overload
    def __divmod__(self, other: Union[IntegerMatrix[M_co, N_co, int], int]) -> tuple[IntegerMatrix[M_co, N_co, int], IntegerMatrix[M_co, N_co, int]]: ...
    @overload
    def __divmod__(self, other: Union[RealMatrix[M_co, N_co, float], float]) -> tuple[RealMatrix[M_co, N_co, float], RealMatrix[M_co, N_co, float]]: ...

    def __divmod__(self, other):
        if isinstance(other, IntegerMatrix):
            b = iter(other)
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return super().__divmod__(other)
        shape = self.shape
        a = iter(self)
        c, d = itertools.tee(map(divmod, a, b))
        return (
            IntegerMatrix(
                array=map(operator.itemgetter(0), c),
                shape=shape,
            ),
            IntegerMatrix(
                array=map(operator.itemgetter(1), d),
                shape=shape,
            ),
        )

    def __lshift__(self, other: Union[IntegerMatrix[M_co, N_co, int], int]) -> IntegerMatrix[M_co, N_co, int]:
        if isinstance(other, IntegerMatrix):
            b = iter(other)
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__lshift__, a, b),
            shape=shape,
        )

    def __rshift__(self, other: Union[IntegerMatrix[M_co, N_co, int], int]) -> IntegerMatrix[M_co, N_co, int]:
        if isinstance(other, IntegerMatrix):
            b = iter(other)
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__rshift__, a, b),
            shape=shape,
        )

    @overload
    def __and__(self: IntegerMatrix[M_co, N_co, bool], other: Union[IntegerMatrix[M_co, N_co, bool], bool]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __and__(self, other: Union[IntegerMatrix[M_co, N_co, int], int]) -> IntegerMatrix[M_co, N_co, int]: ...

    def __and__(self, other):
        if isinstance(other, IntegerMatrix):
            b = iter(other)
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__and__, a, b),
            shape=shape,
        )

    @overload
    def __xor__(self: IntegerMatrix[M_co, N_co, bool], other: Union[IntegerMatrix[M_co, N_co, bool], bool]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __xor__(self, other: Union[IntegerMatrix[M_co, N_co, int], int]) -> IntegerMatrix[M_co, N_co, int]: ...

    def __xor__(self, other):
        if isinstance(other, IntegerMatrix):
            b = iter(other)
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__xor__, a, b),
            shape=shape,
        )

    @overload
    def __or__(self: IntegerMatrix[M_co, N_co, bool], other: Union[IntegerMatrix[M_co, N_co, bool], bool]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __or__(self, other: Union[IntegerMatrix[M_co, N_co, int], int]) -> IntegerMatrix[M_co, N_co, int]: ...

    def __or__(self, other):
        if isinstance(other, IntegerMatrix):
            b = iter(other)
        elif isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__or__, a, b),
            shape=shape,
        )

    @overload  # type: ignore[override]
    def __radd__(self, other: int) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __radd__(self, other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __radd__(self, other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __radd__(self, other):
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return super().__radd__(other)
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__add__, b, a),
            shape=shape,
        )

    @overload  # type: ignore[override]
    def __rsub__(self, other: int) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __rsub__(self, other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __rsub__(self, other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __rsub__(self, other):
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return super().__rsub__(other)
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__sub__, b, a),
            shape=shape,
        )

    @overload  # type: ignore[override]
    def __rmul__(self, other: int) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __rmul__(self, other: float) -> RealMatrix[M_co, N_co, float]: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexMatrix[M_co, N_co, complex]: ...

    def __rmul__(self, other):
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return super().__rmul__(other)
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__mul__, b, a),
            shape=shape,
        )

    @overload
    def __rfloordiv__(self, other: int) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __rfloordiv__(self, other: float) -> RealMatrix[M_co, N_co, float]: ...

    def __rfloordiv__(self, other):
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return super().__rfloordiv__(other)
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__floordiv__, b, a),
            shape=shape,
        )

    @overload
    def __rmod__(self, other: int) -> IntegerMatrix[M_co, N_co, int]: ...
    @overload
    def __rmod__(self, other: float) -> RealMatrix[M_co, N_co, float]: ...

    def __rmod__(self, other):
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return super().__rmod__(other)
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__mod__, b, a),
            shape=shape,
        )

    @overload
    def __rdivmod__(self, other: int) -> tuple[IntegerMatrix[M_co, N_co, int], IntegerMatrix[M_co, N_co, int]]: ...
    @overload
    def __rdivmod__(self, other: float) -> tuple[RealMatrix[M_co, N_co, float], RealMatrix[M_co, N_co, float]]: ...

    def __rdivmod__(self, other):
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return super().__rdivmod__(other)
        shape = self.shape
        a = iter(self)
        c, d = itertools.tee(map(divmod, b, a))
        return (
            IntegerMatrix(
                array=map(operator.itemgetter(0), c),
                shape=shape,
            ),
            IntegerMatrix(
                array=map(operator.itemgetter(1), d),
                shape=shape,
            ),
        )

    def __rlshift__(self, other: int) -> IntegerMatrix[M_co, N_co, int]:
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__lshift__, b, a),
            shape=shape,
        )

    def __rrshift__(self, other: int) -> IntegerMatrix[M_co, N_co, int]:
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__rshift__, b, a),
            shape=shape,
        )

    @overload
    def __rxor__(self: IntegerMatrix[M_co, N_co, bool], other: bool) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __rxor__(self, other: int) -> IntegerMatrix[M_co, N_co, int]: ...

    def __rxor__(self, other):
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__xor__, b, a),
            shape=shape,
        )

    @overload
    def __rand__(self: IntegerMatrix[M_co, N_co, bool], other: bool) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __rand__(self, other: int) -> IntegerMatrix[M_co, N_co, int]: ...

    def __rand__(self, other):
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__and__, b, a),
            shape=shape,
        )

    @overload
    def __ror__(self: IntegerMatrix[M_co, N_co, bool], other: bool) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __ror__(self, other: int) -> IntegerMatrix[M_co, N_co, int]: ...

    def __ror__(self, other):
        if isinstance(other, int):
            b = itertools.repeat(other)
        else:
            return NotImplemented
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__or__, b, a),
            shape=shape,
        )

    def __neg__(self) -> IntegerMatrix[M_co, N_co, int]:
        return IntegerMatrix(super().__neg__())

    def __pos__(self) -> IntegerMatrix[M_co, N_co, int]:
        return IntegerMatrix(super().__pos__())

    def __abs__(self) -> IntegerMatrix[M_co, N_co, int]:
        return IntegerMatrix(super().__abs__())

    @overload
    def __invert__(self: IntegerMatrix[M_co, N_co, bool]) -> IntegerMatrix[M_co, N_co, bool]: ...
    @overload
    def __invert__(self) -> IntegerMatrix[M_co, N_co, int]: ...

    def __invert__(self):
        shape = self.shape
        a = iter(self)
        return IntegerMatrix(
            array=map(operator.__invert__, a),
            shape=shape,
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

    def conjugate(self) -> IntegerMatrix[M_co, N_co, int]:
        return IntegerMatrix(super().conjugate())

    def transjugate(self) -> IntegerMatrix[N_co, M_co, int]:
        return IntegerMatrix(super().transjugate())
