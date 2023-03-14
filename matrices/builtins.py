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


class Matrix(Shaped[M_co, N_co], Sequence[T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("_grid",)

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
        matrix = self.materialize()
        return f"{self.__class__.__name__}(array={matrix.array!r}, shape={matrix.shape!r})"

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
            return self.__class__(value)
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
        return self.__class__(self.grid.transpose())

    def flip(self, *, by: Rule = Rule.ROW) -> Matrix[M_co, N_co, T_co]:
        """Return a flipped view of the matrix"""
        return self.__class__(self.grid.flip(by=by))

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
        return self.__class__(self.grid.rotate(n))

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
        return self.__class__(self.grid.materialize())

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
        return map(self.__class__, self.grid.slices(by=by, reverse=reverse))

    @overload
    def equal(self, other: Matrix[M_co, N_co, object]) -> Matrix[M_co, N_co, bool]: ...
    @overload
    def equal(self, other: object) -> Matrix[M_co, N_co, bool]: ...

    def equal(self, other):
        """Return element-wise ``a == b``"""
        if isinstance(other, Matrix):
            b = iter(other)
        else:
            b = itertools.repeat(other)
        a = iter(self)
        return Matrix(
            array=map(operator.__eq__, a, b),
            shape=self.shape,
        )

    @overload
    def not_equal(self, other: Matrix[M_co, N_co, object]) -> Matrix[M_co, N_co, bool]: ...
    @overload
    def not_equal(self, other: object) -> Matrix[M_co, N_co, bool]: ...

    def not_equal(self, other):
        """Return element-wise ``a != b``"""
        if isinstance(other, Matrix):
            b = iter(other)
        else:
            b = itertools.repeat(other)
        a = iter(self)
        return Matrix(
            array=map(operator.__ne__, a, b),
            shape=self.shape,
        )
