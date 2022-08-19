from __future__ import annotations

import copy
import enum
import functools
import itertools
import operator
import reprlib
import typing
from collections.abc import Callable, Iterable, Iterator, Sequence
from enum import Enum
from io import StringIO
from typing import Any, Literal, NamedTuple, ParamSpec, Type, TypeVar

T = TypeVar("T")
R = TypeVar("R")

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
Tx = TypeVar("Tx")

P = ParamSpec("P")


class Shape(NamedTuple):
    """The dimensions of a matrix

    A named tuple of two integer fields, `nrows` and `ncols`.

    Note that there is no validation of arguments on construction - the limits
    of valid dimensions are left up to the matrix implementation.
    """

    nrows: int
    ncols: int

    @property
    def size(self) -> int:
        """The product of the number of rows and columns"""
        return self.nrows * self.ncols

    def __str__(self) -> str:
        """Return a string representation of the shape"""
        return f"{self.nrows} × {self.ncols}"


class Rule(Enum):
    """The direction by which to operate within a matrix

    An enum of two integer-valued members, `ROW` and `COL`.

    The values of both members are usable as an index that obtains the
    corresponding dimension from a shape (or any two-element sequence).
    """

    ROW: int = 0
    COL: int = 1

    @property
    def true_name(self) -> str:
        """The "true" unabbreviated name of the rule

        Typically used for error messages.
        """
        return "column" if self is Rule.COL else "row"


def binary_reverse(func: Callable[[Any, Any], R]) -> Callable[[Any, Any], R]:
    """Return a wrapper of a given binary function that reverses its incoming
    arguments
    """

    def wrapper(x: Any, y: Any) -> R:
        return func(y, x)

    return wrapper


def logical_and(a: Any, b: Any, /) -> bool:
    """Return the logical and of two objects"""
    return not not (a and b)


def logical_not(a: Any, /) -> bool:
    """Return the logical not of an object"""
    return not a


def logical_or(a: Any, b: Any, /) -> bool:
    """Return the logical or of two objects"""
    return not not (a or b)


def logical_xor(a: Any, b: Any, /) -> bool:
    """Return the logical xor of two objects"""
    return not not ((a and not b) or (not a and b))


class Key(Enum):
    """Enum used to signal the multiplicity of a key, as returned by indices()"""

    ONE  = enum.auto()
    MANY = enum.auto()

@typing.overload
def indices(
    key: tuple[int, int],
    nrows: int,
    ncols: int,
) -> tuple[Literal[Key.ONE], Literal[Key.ONE], int, int]:
    pass

@typing.overload
def indices(
    key: tuple[int, slice],
    nrows: int,
    ncols: int,
) -> tuple[Literal[Key.ONE], Literal[Key.MANY], int, range]:
    pass

@typing.overload
def indices(
    key: tuple[slice, int],
    nrows: int,
    ncols: int,
) -> tuple[Literal[Key.MANY], Literal[Key.ONE], range, int]:
    pass

@typing.overload
def indices(
    key: tuple[slice, slice],
    nrows: int,
    ncols: int,
) -> tuple[Literal[Key.MANY], Literal[Key.MANY], range, range]:
    pass

def indices(key, nrows, ncols):
    """Parse a tuple key, dividing it into a new tuple that maps the sub-key
    types and the (safe) indices they correspond to
    """
    try:
        rowkey, colkey = key
    except TypeError:
        raise TypeError("indices must be an integer, slice, or tuple of slices and/or integers") from None

    def index(subkey, n, by):
        if isinstance(subkey, int):
            k = subkey + (n if subkey < 0 else 0)
            if not (0 <= k < n):
                raise IndexError(f"{by.true_name} index out of range: there are {n} {by.true_name}s but index was {subkey!r}")
            return (Key.ONE, k)
        if isinstance(subkey, slice):
            k = range(*subkey.indices(n))
            return (Key.MANY, k)
        raise TypeError("tuple index must contain integers and/or slices")

    m, rk = index(rowkey, nrows, by=Rule.ROW)
    n, ck = index(colkey, ncols, by=Rule.COL)

    return (m, n, rk, ck)


@typing.final
class Matrix(Sequence[T]):
    """A sequence type for manipulating arbitrary data types in both one and
    two dimensions

    The `Matrix` type implements the built-in `collections.abc.Sequence`
    interface. Matrices are mutable, and will perform certain operations in
    place - most will additionally return `self`, allowing for methods to be
    chained.

    This class behaves like a `Sequence[T]`, but interfaces
    `Sequence[Sequence[T]]`-like functionality. Indexing and iteration is
    always done in row-major order unless stated otherwise.
    """

    __slots__ = "data", "nrows", "ncols"

    def __init__(self: Matrix[T], values: Iterable[T], nrows: int, ncols: int) -> None:
        """Construct a matrix from the elements of `values`, interpreting it as
        shape `nrows` × `ncols`
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("dimensions must be non-negative")

        data = list(values)
        if (size := len(data)) != nrows * ncols:
            raise ValueError(f"cannot interpret size {size} iterable as shape {nrows} × {ncols}")

        self.nrows: int = nrows
        self.ncols: int = ncols
        self.data: list[T] = data

    @classmethod
    def new(cls: Type[Matrix], data: list[T], nrows: int, ncols: int) -> Matrix[T]:
        """Construct a matrix directly from a data list and shape

        This method exists primarily for the benefit of matrix-producing
        functions that have "pre-validated" the data and shape. Should be used
        with caution - this method is not marked as internal because its usage
        is not entirely discouraged if you're aware of the dangers.

        The following properties are required to construct a valid matrix:
        - `data` must be a flattened `list`. That is, the elements of the
          matrix must be on the shallowest depth. A nested list would imply a
          matrix that contains `list` instances.
        - `shape` must be a `Shape`, where the product of its dimensions must
          equal `len(data)`.
        """
        self = cls.__new__(cls)
        self.data = data
        self.nrows, self.ncols = nrows, ncols
        return self

    @classmethod
    def fill(cls: Type[Matrix], value: T, nrows: int, ncols: int) -> Matrix[T]:
        """Construct a matrix of shape `nrows` × `ncols`, comprised solely of
        `value`
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("dimensions must be non-negative")
        data = list(itertools.repeat(value, nrows * ncols))
        return cls.new(data, nrows, ncols)

    @classmethod
    def fill_like(cls: Type[Matrix], value: T, other: Matrix) -> Matrix[T]:
        """Construct a matrix of equal shape to `other`, comprised solely of
        `value`
        """
        nrows, ncols = other.shape
        return cls.fill(value, nrows=nrows, ncols=ncols)

    @property
    def shape(self: Matrix[T]) -> Shape:
        """A tuple of the matrix's number of rows and columns"""
        return Shape(self.nrows, self.ncols)

    @property
    def size(self: Matrix[T]) -> int:
        """The product of the number of rows and columns"""
        return self.nrows * self.ncols

    @reprlib.recursive_repr(fillvalue="...")
    def __repr__(self: Matrix[T]) -> str:
        """Return a canonical representation of the matrix"""
        return f"{self.__class__.__name__}({self.data!r}, nrows={self.nrows!r}, ncols={self.ncols!r})"

    def __str__(self: Matrix[T]) -> str:
        """Return a string representation of the matrix

        Elements of the matrix are organized into right-justified fields of
        width 10. If an element's string representation goes beyond this width,
        it will be truncated to 9 characters, using `…` as a suffix. A single
        space is used to delineate elements.

        Customizable string formatting (through `__format__()`) is a planned
        addition. For the time being, custom formatting must be done manually.
        """
        nrows, ncols = self.shape
        items = iter(self)

        max_width = 10
        result = StringIO()

        if nrows and ncols:

            for _ in range(nrows):
                result.write("| ")

                for _ in range(ncols):
                    chars = str(next(items))
                    if len(chars) > max_width:
                        result.write(f"{chars[:max_width - 1]}…")
                    else:
                        result.write(chars.rjust(max_width))
                    result.write(" ")

                result.write("|\n")

        else:
            result.write("Empty matrix ")

        result.write(f"({nrows} × {ncols})")

        return result.getvalue()

    def __lt__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.lt()`"""
        data = list(self.map(operator.lt, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __le__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.le()`"""
        data = list(self.map(operator.le, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __eq__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:  # type: ignore[override]
        """Return the mapping of `operator.eq()`"""
        data = list(self.map(operator.eq, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __ne__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:  # type: ignore[override]
        """Return the mapping of `operator.ne()`"""
        data = list(self.map(operator.ne, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __gt__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.gt()`"""
        data = list(self.map(operator.gt, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __ge__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.ge()`"""
        data = list(self.map(operator.ge, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __call__(self: Matrix[Callable[P, R]], *args: P.args, **kwargs: P.kwargs) -> Matrix[R]:
        """Return a new matrix of the results from calling each element with
        the given arguments
        """
        data = [f(*args, **kwargs) for f in self]
        return Matrix.new(data, self.nrows, self.ncols)

    def __len__(self: Matrix[T]) -> int:
        """Return the matrix's size"""
        return len(self.data)

    @typing.overload
    def __getitem__(self: Matrix[T], key: int) -> T:
        pass

    @typing.overload
    def __getitem__(self: Matrix[T], key: slice) -> Matrix[T]:
        pass

    @typing.overload
    def __getitem__(self: Matrix[T], key: tuple[int, int]) -> T:
        pass

    @typing.overload
    def __getitem__(self: Matrix[T], key: tuple[int, slice]) -> Matrix[T]:
        pass

    @typing.overload
    def __getitem__(self: Matrix[T], key: tuple[slice, int]) -> Matrix[T]:
        pass

    @typing.overload
    def __getitem__(self: Matrix[T], key: tuple[slice, slice]) -> Matrix[T]:
        pass

    def __getitem__(self, key):
        """Return the element or sub-matrix corresponding to `key`

        If `key` is an integer or slice, it is treated as if it is indexing the
        flattened matrix, returning the corresponding value(s). A slice will
        always return a new matrix of shape `(1, N)`, where `N` is the length
        of the slice's range.

        If `key` is a tuple, the first index is applied against the rows, while
        the second is applied against the columns. A tuple of two integers,
        `(i, j)`, will return the element at row `i`, column `j`. All other
        tuple variations will return a new sub-matrix of shape `(M, N)`, where
        `M` is the length of the first slice's range, and `N` is the length of
        the second slice's range - either becomes 1 if the sub-key is an
        integer.
        """
        if isinstance(key, int):
            try:
                res = self.data[key]
            except IndexError:
                raise IndexError(f"index out of range: size is {self.size} but index was {key}") from None
            else:
                return res

        nrows, ncols = self.shape

        if isinstance(key, slice):
            ix = range(*key.indices(nrows * ncols))
            return Matrix.new(
                [self.data[i] for i in ix],
                nrows=1,
                ncols=len(ix),
            )

        match indices(key, nrows, ncols):

            case (Key.ONE, Key.ONE, i, j):
                return self.data[i * ncols + j]

            case (Key.ONE, Key.MANY, i, jx):
                return Matrix.new(
                    [self.data[i * ncols + j] for j in jx],
                    nrows=1,
                    ncols=len(jx),
                )

            case (Key.MANY, Key.ONE, ix, j):
                return Matrix.new(
                    [self.data[i * ncols + j] for i in ix],
                    nrows=len(ix),
                    ncols=1,
                )

            case (Key.MANY, Key.MANY, ix, jx):
                return Matrix.new(
                    [self.data[i * ncols + j] for i in ix for j in jx],
                    nrows=len(ix),
                    ncols=len(jx),
                )

    @typing.overload
    def __setitem__(self: Matrix[T], key: int, value: T) -> None:
        pass

    @typing.overload
    def __setitem__(self: Matrix[T], key: slice, value: Iterable[T] | T) -> None:
        pass

    @typing.overload
    def __setitem__(self: Matrix[T], key: tuple[int, int], value: T) -> None:
        pass

    @typing.overload
    def __setitem__(self: Matrix[T], key: tuple[int, slice], value: Iterable[T] | T) -> None:
        pass

    @typing.overload
    def __setitem__(self: Matrix[T], key: tuple[slice, int], value: Iterable[T] | T) -> None:
        pass

    @typing.overload
    def __setitem__(self: Matrix[T], key: tuple[slice, slice], value: Iterable[T] | T) -> None:
        pass

    def __setitem__(self, key, value):
        """Overwrite the element or sub-matrix corresponding to `key` with
        `value`

        If `key` is an integer or slice, it is treated as if it is indexing
        the flattened matrix, overwriting the corresponding value(s).

        If `key` is a tuple, the first index is applied against the rows, while
        the second is applied against the columns. A tuple of two integers,
        `(i, j)`, will overwrite the element at row `i`, column `j`. All other
        tuple variations will overwrite a sub-matrix of shape `(M, N)`, where
        `M` is the length of the first slice's range, and `N` is the length of
        the second slice's range - either becomes 1 if the sub-key is an
        integer.

        If the key is any variation of a slice, `value` is expected to be a
        matrix of identical size to the key's selected range. If `value` is not
        a matrix, it's assumed to be an element, and will instead be repeated
        across the key's selection.
        """
        if isinstance(key, int):
            try:
                self.data[key] = value
            except IndexError:
                raise IndexError(f"index out of range: size is {self.size} but index was {key}") from None
            else:
                return

        nrows, ncols = self.shape

        def value_iterator(m):
            try:
                it = iter(value)
            except TypeError:
                yield from itertools.repeat(value, m)
            else:
                yield from it

        if isinstance(key, slice):
            ix = range(*key.indices(nrows * ncols))
            values = value_iterator(len(ix))
            for i, x in zip(ix, values):
                self.data[i] = x
            return

        match indices(key, nrows, ncols):

            case (Key.ONE, Key.ONE, i, j):
                self.data[i * ncols + j] = value

            case (Key.ONE, Key.MANY, i, jx):
                values = value_iterator(len(jx))
                for j, x in zip(jx, values):
                    self.data[i * ncols + j] = x

            case (Key.MANY, Key.ONE, ix, j):
                values = value_iterator(len(ix))
                for i, x in zip(ix, values):
                    self.data[i * ncols + j] = x

            case (Key.MANY, Key.MANY, ix, jx):
                values = value_iterator(len(ix) * len(jx))
                for i in ix:
                    for j in jx:
                        self.data[i * ncols + j] = next(values)

    def __iter__(self: Matrix[T]) -> Iterator[T]:
        """Return an iterator over the elements of the matrix"""
        yield from iter(self.data)

    def __reversed__(self: Matrix[T]) -> Iterator[T]:
        """Return a reverse iterator over the elements of the matrix"""
        yield from reversed(self.data)

    def __contains__(self: Matrix[T], value: Any) -> bool:
        """Return true if the matrix contains `value`, otherwise false"""
        return value in self.data

    def __add__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.add()`"""
        data = list(self.map(operator.add, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __sub__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.sub()`"""
        data = list(self.map(operator.sub, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __mul__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.mul()`"""
        data = list(self.map(operator.mul, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __truediv__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.truediv()`"""
        data = list(self.map(operator.truediv, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __floordiv__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.floordiv()`"""
        data = list(self.map(operator.floordiv, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __mod__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.mod()`"""
        data = list(self.map(operator.mod, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __pow__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.pow()`"""
        data = list(self.map(operator.pow, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __and__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[bool]:
        """Return the mapping of `logical_and()`"""
        data = list(self.map(logical_and, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __xor__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[bool]:
        """Return the mapping of `logical_xor()`"""
        data = list(self.map(logical_xor, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __or__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[bool]:
        """Return the mapping of `logical_or()`"""
        data = list(self.map(logical_or, other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __radd__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the reverse mapping of `operator.add()`"""
        data = list(self.map(binary_reverse(operator.add), other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __rsub__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the reverse mapping of `operator.sub()`"""
        data = list(self.map(binary_reverse(operator.sub), other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __rmul__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the reverse mapping of `operator.mul()`"""
        data = list(self.map(binary_reverse(operator.mul), other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __rtruediv__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the reverse mapping of `operator.truediv()`"""
        data = list(self.map(binary_reverse(operator.truediv), other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __rfloordiv__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the reverse mapping of `operator.floordiv()"""
        data = list(self.map(binary_reverse(operator.floordiv), other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __rmod__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the reverse mapping of `operator.mod()`"""
        data = list(self.map(binary_reverse(operator.mod), other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __rpow__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the reverse mapping of `operator.pow()`"""
        data = list(self.map(binary_reverse(operator.pow), other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __rand__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[bool]:
        """Return the reverse mapping of `logical_and()`"""
        data = list(self.map(binary_reverse(logical_and), other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __rxor__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[bool]:
        """Return the reverse mapping of `logical_xor()`"""
        data = list(self.map(binary_reverse(logical_xor), other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __ror__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[bool]:
        """Return the reverse mapping of `logical_or()`"""
        data = list(self.map(binary_reverse(logical_or), other))
        return Matrix.new(data, self.nrows, self.ncols)

    def __neg__(self: Matrix[T]) -> Matrix[Any]:
        """Return the mapping of `operator.neg()`"""
        data = list(self.map(operator.neg))
        return Matrix.new(data, self.nrows, self.ncols)

    def __pos__(self: Matrix[T]) -> Matrix[Any]:
        """Return the mapping of `operator.pos()`"""
        data = list(self.map(operator.pos))
        return Matrix.new(data, self.nrows, self.ncols)

    def __invert__(self: Matrix[T]) -> Matrix[bool]:
        """Return the mapping of `logical_not()`"""
        data = list(self.map(logical_not))
        return Matrix.new(data, self.nrows, self.ncols)

    def __matmul__(self: Matrix[T], other: Matrix[Any]) -> Matrix[Any]:
        """Return the matrix product

        In general, elements must behave "numerically" in their implementation
        of `__add__()` and `__mul__()` for a valid matrix product. This method
        attempts to generalize by using a left fold summation.

        If the operand matrices are of empty shapes `(M, 0)` and `(0, N)`,
        respectively, the product will be an `(M, N)` matrix filled with
        `None`.
        """
        if not isinstance(other, Matrix):
            return NotImplemented

        (m, n), (p, q) = self.shape, other.shape
        if n != p:
            raise ValueError("matrices must have equal inner dimensions")

        if not n:
            return Matrix.fill(None, m, q)

        return Matrix.new(
            [
                functools.reduce(
                    operator.add,
                    (self.data[i * n + k] * other.data[k * q + j] for k in range(n)),
                )
                for i in range(m)
                for j in range(q)
            ],
            nrows=m,
            ncols=q,
        )

    __rmatmul__ = __matmul__

    def __copy__(self: Matrix[T]) -> Matrix[T]:
        """Return a shallow copy of the matrix"""
        data = copy.copy(self.data)
        return Matrix.new(data, self.nrows, self.ncols)

    def __deepcopy__(self: Matrix[T], memo: dict[int, Any] | None = None) -> Matrix[T]:
        """Return a deep copy of the matrix"""
        data = copy.deepcopy(self.data, memo)
        return Matrix.new(data, self.nrows, self.ncols)

    def index(self: Matrix[T], value: T, start: int = 0, stop: int | None = None) -> int:
        """Return the index of the first element equal to `value`

        Raises `ValueError` if `value` could not be found in the matrix.
        """
        try:                                           # typeshed doesn't hint Optional[int] like it should
            index = super().index(value, start, stop)  # type: ignore[arg-type]
        except ValueError:
            raise ValueError("value not found") from None
        else:
            return index

    def count(self: Matrix[T], value: T) -> int:
        """Return the number of times `value` appears in the matrix"""
        return super().count(value)

    def reshape(self: Matrix[T], nrows: int, ncols: int) -> Matrix[T]:
        """Re-interpret the matrix's shape

        Raises `ValueError` if any of the given dimensions are negative, or if
        their product does not equal the matrix's current size.
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("dimensions must be non-negative")
        if (size := self.size) != nrows * ncols:
            raise ValueError(f"cannot re-shape size {size} matrix as shape {nrows} × {ncols}")
        self.nrows = nrows
        self.ncols = ncols
        return self

    @typing.overload
    def iter(self: Matrix[T]) -> Iterator[T]:
        pass

    @typing.overload
    def iter(self: Matrix[T], *, by: Rule) -> Iterator[Iterator[T]]:
        pass

    def iter(self: Matrix[T], *, by=None):
        """Return an iterator that yields the values, rows, or columns of the
        matrix

        If performed over the rows or columns, the returned iterator yields
        new iterators that then yield the values of a row or column.
        """
        if by is None:
            yield from iter(self.data)
            return

        nrows, ncols = self.shape

        if by is Rule.COL:
            for j in range(ncols):
                yield (self.data[i * ncols + j] for i in range(nrows))

        else:
            for i in range(nrows):
                yield (self.data[i * ncols + j] for j in range(ncols))

    def iter_collect(self: Matrix[T], *, by: Rule = Rule.ROW) -> Iterator[Matrix[T]]:
        """Return an iterator that yields the rows or columns as new matrices"""
        nrows, ncols = self.shape

        if by is Rule.COL:
            for j in range(ncols):
                data = self.data[j : nrows * ncols + j : ncols]
                yield Matrix.new(data, nrows=nrows, ncols=1)

        else:
            for i in range(nrows):
                data = self.data[i * ncols : i * ncols + ncols]
                yield Matrix.new(data, nrows=1, ncols=ncols)

    @typing.overload
    def map(self: Matrix[T1], func: Callable[[T1], R]) -> Iterator[R]:
        pass

    @typing.overload
    def map(self: Matrix[T1], func: Callable[[T1, T2], R], other1: Matrix[T2] | T2) -> Iterator[R]:
        pass

    @typing.overload
    def map(self: Matrix[T1], func: Callable[[T1, T2, T3], R], other1: Matrix[T2] | T2, other2: Matrix[T3] | T3) -> Iterator[R]:
        pass

    @typing.overload
    def map(self: Matrix[T1], func: Callable[..., R], *others: Matrix[Tx] | Tx) -> Iterator[R]:
        pass

    @typing.overload
    def map(self: Matrix[T1], func: Callable[[Iterator[T1]], R], *, by: Rule) -> Iterator[R]:
        pass

    @typing.overload
    def map(self: Matrix[T1], func: Callable[[Iterator[T1], Iterator[T2] | T2], R], other1: Matrix[T2] | T2, *, by: Rule) -> Iterator[R]:
        pass

    @typing.overload
    def map(self: Matrix[T1], func: Callable[[Iterator[T1], Iterator[T2] | T2, Iterator[T3] | T3], R], other1: Matrix[T2] | T2, other2: Matrix[T3] | T3, *, by: Rule) -> Iterator[R]:
        pass

    @typing.overload
    def map(self: Matrix[T1], func: Callable[..., R], *others: Matrix[Tx] | Tx, by: Rule) -> Iterator[R]:
        pass

    def map(self, func, *others, by=None):
        """Map `func` along the matrix's values, rows, or columns in parallel
        with other matrices and/or objects, yielding each result

        Raises `ValueError` if operand matrices differ in size or shape - size
        if mapping over the values, or shape if mapping over the rows/columns.
        """

        def other_iterators(limit):
            if not others:
                return
            m = getattr(self, limit)
            for i, other in enumerate(others, start=1):
                if isinstance(other, Matrix):
                    n = getattr(other, limit)
                    if m != n:
                        raise ValueError(f"incompatible {limit}s: operating matrix has {limit} {m} but operand {i} has {limit} {n}")
                    yield other.iter(by=by)
                else:
                    yield itertools.repeat(other)

        it1 = self.iter(by=by)
        itx = other_iterators(limit="size" if by is None else "shape")

        yield from map(func, it1, *itx)

    @typing.overload
    def apply(self: Matrix[T1], func: Callable[[T1], R]) -> Matrix[R]:
        pass

    @typing.overload
    def apply(self: Matrix[T1], func: Callable[[T1, T2], R], other1: Matrix[T2] | T2) -> Matrix[R]:
        pass

    @typing.overload
    def apply(self: Matrix[T1], func: Callable[[T1, T2, T3], R], other1: Matrix[T2] | T2, other2: Matrix[T3] | T3) -> Matrix[R]:
        pass

    @typing.overload
    def apply(self: Matrix[T1], func: Callable[..., R], *others: Matrix[Tx] | Tx) -> Matrix[R]:
        pass

    @typing.overload
    def apply(self: Matrix[T1], func: Callable[[Iterator[T1]], Iterable[R]], *, by: Rule) -> Matrix[R]:
        pass

    @typing.overload
    def apply(self: Matrix[T1], func: Callable[[Iterator[T1], Iterator[T2] | T2], Iterable[R]], other1: Matrix[T2] | T2, *, by: Rule) -> Matrix[R]:
        pass

    @typing.overload
    def apply(self: Matrix[T1], func: Callable[[Iterator[T1], Iterator[T2] | T2, Iterator[T3] | T3], Iterable[R]], other1: Matrix[T2] | T2, other2: Matrix[T3] | T3, *, by: Rule) -> Matrix[R]:
        pass

    @typing.overload
    def apply(self: Matrix[T1], func: Callable[..., Iterable[R]], *others: Matrix[Tx] | Tx, by: Rule) -> Matrix[R]:
        pass

    def apply(self, func, *others, by=None):
        """Map `func` along the matrix's values, rows, or columns in parallel
        with other matrices and/or objects, writing the results to the matrix

        Raises `ValueError` if operand matrices differ in size or shape - size
        if mapping over the values, or shape if mapping over the rows/columns.

        If applying by row or column, the input function must return a sequence
        type that has a size equal to the opposite dimension that was mapped
        over (e.g., application by row means the function must return a matrix
        that has a size equal to `self.ncols`, and vice versa).
        """
        items = self.map(func, *others, by=by)

        if by is not None:
            nrows, ncols = self.shape

            if by is Rule.COL:
                for j, other in enumerate(items):
                    ix = range(j, nrows * ncols + j, ncols)
                    for i, x in zip(ix, other):
                        self.data[i] = x

            else:
                for i, other in enumerate(items):
                    jx = range(i * ncols, i * ncols + ncols)
                    for j, x in zip(jx, other):
                        self.data[j] = x

        else:
            self.data[:] = items

        return self

    def collapse(self: Matrix[T], func: Callable[[Iterator[T]], R], *, by: Rule = Rule.ROW) -> Matrix[R]:
        self.data[:] = map(func, self.iter(by=by))
        if by is Rule.COL:
            self.nrows = 1
        else:
            self.ncols = 1
        return self

    def reduce(self: Matrix[T], func: Callable[[R, T], R], *, by: Rule = Rule.ROW) -> Matrix[R]:
        return self.collapse(functools.partial(functools.reduce, func), by=by)

    def replace(self: Matrix[T], old: T, new: T, *, times: int | None = None) -> Matrix[T]:
        """Replace elements equal to `old` with `new`

        If `times` is given, only the first `times` occurrences of `old` will
        be replaced.

        This method considers two objects equal if a comparison by identity or
        equality is satisfied, meaning that some objects that can only be
        checked by identity may be replaced through this method (e.g.,
        `math.nan` and similar objects).
        """
        ix = (i for i, x in enumerate(self) if x is old or x == old)
        for i in itertools.islice(ix, times):
            self.data[i] = new
        return self

    def reverse(self: Matrix[T]) -> Matrix[T]:
        """Reverse the matrix's elements in place"""
        self.data.reverse()
        return self

    def swap(self: Matrix[T], i: int, j: int, *, by: Rule = Rule.ROW) -> Matrix[T]:
        """Swap the two rows or columns beginning at indices `i` and `j`

        Note that, due to how `Matrix` stores its data, swapping is performed
        in linear time with respect to the specified dimension.
        """
        shape = self.shape
        m, n = shape[by.value], shape[not by.value]

        def index(key, n):
            k = key + (n if key < 0 else 0)
            if not (0 <= k < n):
                raise IndexError(f"{by.true_name} index out of range: there are {n} {by.true_name}s but index was {k!r}")
            return k

        i = index(i, m)
        j = index(j, m)

        if by is Rule.COL:

            def indices():
                for k in range(n):
                    yield (k * m + i, k * m + j)

        else:

            def indices():
                for k in range(n):
                    yield (i * n + k, j * n + k)

        for x, y in indices():
            self.data[x], self.data[y] = self.data[y], self.data[x]

        return self

    def flip(self: Matrix[T], *, by: Rule = Rule.ROW) -> Matrix[T]:
        """Reverse the matrix's rows or columns in place"""
        shape = self.shape
        m, n = shape[by.value], shape[not by.value]

        if n > 1:

            if by is Rule.COL:

                def indices(i, j):
                    for k in range(n):
                        yield (k * m + i, k * m + j)

            else:

                def indices(i, j):
                    for k in range(n):
                        yield (i * n + k, j * n + k)

            for i in range(m // 2):
                for x, y in indices(i, m - i - 1):
                    self.data[x], self.data[y] = self.data[y], self.data[x]

        else:
            self.data.reverse()  # Faster path for row and column vectors

        return self

    def flatten(self: Matrix[T], *, by: Rule = Rule.ROW) -> Matrix[T]:
        """Re-shape the matrix to a row or column vector

        If flattened to a column vector, the elements are put into column-major
        order.
        """
        if by is Rule.COL:
            self.transpose()
            self.nrows, self.ncols = self.size, 1
        else:
            self.nrows, self.ncols = 1, self.size
        return self

    def transpose(self: Matrix[T]) -> Matrix[T]:
        """Transpose the matrix in place"""
        nrows, ncols = self.shape
        if nrows > 1 and ncols > 1:
            self.data[:] = (self.data[i * ncols + j] for j in range(ncols) for i in range(nrows))
        self.nrows, self.ncols = ncols, nrows
        return self

    def stack(self: Matrix[T], other: Sequence[T], *, by: Rule = Rule.ROW) -> Matrix[T]:
        """Stack a sequence or other matrix along a rule

        Raises `ValueError` if the opposite dimension corresponding to the
        given rule differs between the operand matrices.

        If `other` is a sequence type, but not a matrix, it will be interpreted
        as a row or column vector if either row or column-rule is specified,
        respectively.
        """
        if self is other:
            other = other.copy()  # type: ignore[attr-defined]

        m, n = self.shape

        if by is Rule.COL:
            p, q = getattr(other, "shape", (len(other), 1))
            if m != p:
                raise ValueError(f"incompatible shapes: operating matrix has {m} rows but operand has {p}")
            if m > 1:
                left, right = iter(self), iter(other)
                self.data[:] = (
                    x
                    for _ in range(m)
                    for x in itertools.chain(itertools.islice(left, n), itertools.islice(right, q))
                )
            else:
                self.data.extend(other)
            self.ncols += q

        else:
            p, q = getattr(other, "shape", (1, len(other)))
            if n != q:
                raise ValueError(f"incompatible shapes: operating matrix has {n} columns but operand has {q}")
            self.data.extend(other)
            self.nrows += p

        return self

    def pull(self: Matrix[T], index: int = -1, *, by: Rule = Rule.ROW) -> Matrix[T]:
        """Remove and return a row or column from the matrix

        Raises `IndexError` if the matrix is empty, or if the index is out of
        range.
        """
        shape = self.shape
        m, n = shape[by.value], shape[not by.value]

        if (not m) or (not n):
            raise IndexError("cannot pull from empty matrix")

        i = index + (m if index < 0 else 0)
        if not (0 <= i < m):
            raise IndexError(f"index out of range: there are {m} {by.true_name}s but index was {index!r}")

        if by is Rule.COL:
            key   = slice(i, m * n + i, m)
            shape = (n, 1)
            self.ncols -= 1
        else:
            key   = slice(i * n, i * n + n)
            shape = (1, n)
            self.nrows -= 1

        data = self.data[key]
        del self.data[key]

        return Matrix.new(data, *shape)

    def copy(self: Matrix[T], *, deep: bool = False) -> Matrix[T]:
        """Return a shallow or deep copy of the matrix"""
        return copy.deepcopy(self) if deep else copy.copy(self)


def vector(values: Iterable[T], *, by: Rule = Rule.ROW) -> Matrix[T]:
    """Construct a row or column vector from an iterable, using its length to
    deduce the number of columns or rows, respectively
    """
    data = list(values)
    n = len(data)

    if by is Rule.COL:
        return Matrix.new(data, nrows=n, ncols=1)
    return Matrix.new(data, nrows=1, ncols=n)


def matrix(values: Iterable[Iterable[T]]) -> Matrix[T]:
    """Construct a matrix from a singly-nested iterable, using the shallowest
    iterable's length to deduce the number of rows, and the nested iterables'
    length to deduce the number of columns

    Raises `ValueError` if the length of the nested iterables is inconsistent
    (i.e., a representation of an irregular matrix).
    """
    data: list[T] = []

    rows = iter(values)
    try:
        row = next(rows)
    except StopIteration:
        return Matrix.new(data, nrows=0, ncols=0)
    else:
        data.extend(row)

    m = 1
    n = len(data)

    for m, row in enumerate(rows, start=2):
        k = 0
        for k, val in enumerate(row, start=1):
            data.append(val)
        if n != k:
            raise ValueError("values form an irregular matrix")

    return Matrix.new(data, nrows=m, ncols=n)
