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
from typing import Any, NamedTuple, ParamSpec, Type, TypeVar

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


class Rule(Enum):
    """The direction by which to operate within a matrix

    Analogous to NumPy's axes, but as an enum with just two members, `ROW` and
    `COL`. Parts of the documentation may refer to these members as "row-rule"
    and "column-rule".

    Members of this enum do not support equality with their integer values
    (meaning, a 0 or 1 cannot pass as a substitute for using `Rule.ROW` or
    `Rule.COL`).
    """

    ROW: int = 0
    COL: int = 1


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


def matmul_iterator(a: Matrix[Any], b: Matrix[Any]) -> tuple[Iterator[Any], Shape]:
    """Return an iterator that yields the matrix product's values in row-major
    order, and the shape to interpret it as

    Raises `ValueError` if the two matrices have unequal inner dimensions.
    """
    (m, n), (p, q) = a.shape, b.shape
    if n != p:
        raise ValueError("matrices must have equal inner dimensions")

    shape = Shape(m, q)

    if n:
        ls, rs = a.data, b.data

        def product() -> Iterator[Any]:
            ix, jx, kx = map(range, (m, q, n))
            for i in ix:
                for j in jx:
                    yield functools.reduce(
                        operator.add,
                        (ls[i * n + k] * rs[k * q + j] for k in kx),
                    )

        items = product()
    else:
        items = itertools.repeat(None, m * q)

    return (items, shape)


class Key(Enum):
    """Enum used to signal the multiplicity of a key, as returned by
    `indices()`

    Tuple keys are not considered key types in the traditional sense - they're
    treated simply as a container of keys since they're disallowed from
    containing other tuple keys.
    """

    ONE  = enum.auto()
    MANY = enum.auto()


def index(key: int | slice, n: int) -> tuple[Key, int | range]:
    if isinstance(key, int):
        if key < 0:
            key += n
        if not (0 <= key < n):
            raise IndexError
        return (Key.ONE, key)
    if isinstance(key, slice):
        return (Key.MANY, range(*key.indices(n)))
    raise TypeError


def indices(key: tuple[int | slice, int | slice], nrows: int, ncols: int) -> Iterator[tuple[Key, int | range]]:
    rowkey, colkey = key

    try:
        yield index(rowkey, nrows)
    except TypeError:
        raise TypeError("tuple index must contain integers and/or slices") from None
    except IndexError:
        raise IndexError(f"row index out of range: there are {nrows} rows but index was {rowkey}") from None

    try:
        yield index(colkey, ncols)
    except TypeError:
        raise TypeError("tuple index must contain integers and/or slices") from None
    except IndexError:
        raise IndexError(f"column index out of range: there are {ncols} columns but index was {colkey}") from None


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

    __slots__ = ("data", "shape")

    def __init__(self: Matrix[T], values: Iterable[T], nrows: int, ncols: int) -> None:
        """Construct a matrix from the elements of `values`, interpreting it as
        shape `nrows` × `ncols`
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("dimensions must be non-negative")

        data = list(values)
        if (size := len(data)) != nrows * ncols:
            raise ValueError(f"cannot interpret size {size} iterable as shape {nrows} × {ncols}")

        self.data: list[T] = data
        self.shape: Shape  = Shape(nrows, ncols)

    @classmethod
    def new(cls: Type[Matrix], data: list[T], shape: Shape) -> Matrix[T]:
        """Construct a matrix directly from a data list and shape

        This method exists primarily for the benefit of matrix-producing
        functions that have "pre-validated" the data and shape.

        Should be used with caution - this method is not marked as internal
        because its usage is not entirely discouraged if you're aware of the
        dangers.

        The following properties are required to construct a valid matrix:
        - `data` must be a flattened `list`. That is, the elements of the
          matrix must be on the shallowest depth. A nested list would imply a
          matrix that contains `list` instances.
        - `shape` must be a `Shape`, where the product of its dimensions must
          equal `len(data)`.
        """
        self = cls.__new__(cls)
        self.data, self.shape = data, shape
        return self

    @classmethod
    def fill(cls: Type[Matrix], value: T, nrows: int, ncols: int) -> Matrix[T]:
        """Construct a matrix of shape `nrows` × `ncols`, comprised solely of
        `value`
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("dimensions must be non-negative")
        data  = list(itertools.repeat(value, nrows * ncols))
        shape = Shape(nrows, ncols)
        return cls.new(data, shape)

    @classmethod
    def fill_like(cls: Type[Matrix], value: T, other: Matrix) -> Matrix[T]:
        """Construct a matrix of equal shape to `other`, comprised solely of
        `value`
        """
        nrows, ncols = other.shape
        return cls.fill(value, nrows=nrows, ncols=ncols)

    @property
    def size(self: Matrix[T]) -> int:
        """The product of the number of rows and columns"""
        nrows, ncols = self.shape
        return nrows * ncols

    @property
    def nrows(self: Matrix[T]) -> int:
        """The matrix's number of rows"""
        return self.shape.nrows

    @property
    def ncols(self: Matrix[T]) -> int:
        """The matrix's number of columns"""
        return self.shape.ncols

    @reprlib.recursive_repr(fillvalue="...")
    def __repr__(self: Matrix[T]) -> str:
        """Return a canonical representation of the matrix"""
        nrows, ncols = self.shape
        items = iter(self)

        result = StringIO()
        result.write(f"{self.__class__.__name__}([")

        if nrows and ncols:
            result.write("\n")

            for _ in range(nrows):
                result.write("    ")

                for _ in range(ncols):
                    chars = repr(next(items))
                    result.write(f"{chars}, ")

                result.write("\n")

        result.write(f"], nrows={nrows!r}, ncols={ncols!r})")

        return result.getvalue()

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
        return self.map(operator.lt, other)

    def __le__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.le()`"""
        return self.map(operator.le, other)

    def __eq__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:  # type: ignore[override]
        """Return the mapping of `operator.eq()`"""
        return self.map(operator.eq, other)

    def __ne__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:  # type: ignore[override]
        """Return the mapping of `operator.ne()`"""
        return self.map(operator.ne, other)

    def __gt__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.gt()`"""
        return self.map(operator.gt, other)

    def __ge__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.ge()`"""
        return self.map(operator.ge, other)

    def __call__(self: Matrix[Callable[P, R]], *args: P.args, **kwargs: P.kwargs) -> Matrix[R]:
        """Return a new matrix of the results from calling each element with
        the given arguments
        """

        def call(func: Callable[P, R], args: tuple[Any, ...], kwargs: dict[str, Any]) -> R:
            return func(*args, **kwargs)

        return self.map(call, args, kwargs)

    def __len__(self: Matrix[T]) -> int:
        """Return the matrix's size"""
        return self.size

    @typing.overload  # type: ignore[override]
    def __getitem__(self: Matrix[T], key: int) -> T:
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

        If `key` is an integer, it is treated as if it is indexing the
        flattened matrix, returning the corresponding value.

        If `key` is a tuple, the first index is applied against the rows, while
        the second is applied against the columns. A tuple of two integers,
        `(i, j)`, will return the element at row `i`, column `j`. All other
        tuple variations will return a new sub-matrix of shape `(M, N)`, where
        `M` is the length of the first slice's range, and `N` is the length of
        the second slice's range - either becomes 1 if the sub-key is an
        integer.
        """
        nrows, ncols = self.shape

        if isinstance(key, int):
            try:
                res = self.data[key]
            except IndexError:
                size = nrows * ncols
                raise IndexError(f"index out of range: size is {size} but index was {key}") from None
            else:
                return res
        try:
            (mi, ix), (mj, jx) = indices(key, nrows, ncols)
        except TypeError:
            raise TypeError("indices must be an integer or tuple of slices and/or integers") from None

        if mi is Key.ONE:
            if mj is Key.ONE:
                return self.data[ix * ncols + jx]
            else:
                data  = [self.data[ix * ncols + j] for j in jx]
                shape = Shape(1, len(jx))
        else:
            if mj is Key.ONE:
                data  = [self.data[i * ncols + jx] for i in ix]
                shape = Shape(len(ix), 1)
            else:
                data  = [self.data[i * ncols + j] for i in ix for j in jx]
                shape = Shape(len(ix), len(jx))

        return self.__class__.new(data, shape)

    @typing.overload  # type: ignore[override]
    def __setitem__(self: Matrix[T], key: int, value: T) -> None:
        pass

    @typing.overload
    def __setitem__(self: Matrix[T], key: tuple[int, int], value: T) -> None:
        pass

    @typing.overload
    def __setitem__(self: Matrix[T], key: tuple[int, slice], value: Matrix[T] | T) -> None:
        pass

    @typing.overload
    def __setitem__(self: Matrix[T], key: tuple[slice, int], value: Matrix[T] | T) -> None:
        pass

    @typing.overload
    def __setitem__(self: Matrix[T], key: tuple[slice, slice], value: Matrix[T] | T) -> None:
        pass

    def __setitem__(self, key, value):
        """Overwrite the element or sub-matrix corresponding to `key` with
        `value`

        If `key` is an integer, it is treated as if it is indexing
        the flattened matrix, overwriting the corresponding value.

        If `key` is a tuple, the first index is applied against the rows, while
        the second is applied against the columns. A tuple of two integers,
        `(i, j)`, will overwrite the element at row `i`, column `j`. All other
        tuple variations will overwrite a sub-matrix of shape `(M, N)`, where
        `M` is the length of the first slice's range, and `N` is the length of
        the second slice's range - either becomes 1 if the sub-key is an
        integer.
        """
        nrows, ncols = self.shape

        if isinstance(key, int):
            try:
                self.data[key] = value
            except IndexError:
                size = nrows * ncols
                raise IndexError(f"index out of range: size is {size} but index was {key}") from None
            else:
                return
        try:
            (mi, ix), (mj, jx) = indices(key, nrows, ncols)
        except TypeError:
            raise TypeError("indices must be an integer or tuple of slices and/or integers") from None

        def values(m: int) -> Iterator[T]:
            try:
                n = value.size
            except AttributeError:
                yield from itertools.repeat(value, m)
            else:
                if m != n:
                    raise ValueError(f"incompatible sizes: input matrix has size {n} but selection was size {m}")
                yield from iter(value)

        if mi is Key.ONE:
            if mj is Key.ONE:
                self.data[ix * ncols + jx] = value
            else:
                it = values(len(jx))
                for j, x in zip(jx, it):
                    self.data[ix * ncols + j] = x
        else:
            if mj is Key.ONE:
                it = values(len(ix))
                for i, x in zip(ix, it):
                    self.data[i * ncols + jx] = x
            else:
                it = values(len(ix) * len(jx))
                for i in ix:
                    for j in jx:
                        self.data[i * ncols + j] = next(it)

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
        return self.map(operator.add, other)

    def __sub__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.sub()`"""
        return self.map(operator.sub, other)

    def __mul__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.mul()`"""
        return self.map(operator.mul, other)

    def __truediv__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.truediv()`"""
        return self.map(operator.truediv, other)

    def __floordiv__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.floordiv()`"""
        return self.map(operator.floordiv, other)

    def __mod__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.mod()`"""
        return self.map(operator.mod, other)

    def __pow__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the mapping of `operator.pow()`"""
        return self.map(operator.pow, other)

    def __and__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[bool]:
        """Return the mapping of `logical_and()`"""
        return self.map(logical_and, other)

    def __xor__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[bool]:
        """Return the mapping of `logical_xor()`"""
        return self.map(logical_xor, other)

    def __or__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[bool]:
        """Return the mapping of `logical_or()`"""
        return self.map(logical_or, other)

    def __matmul__(self: Matrix[T], other: Matrix[Any]) -> Matrix[Any]:
        """Return the matrix product

        In general, elements must behave numerically in their implementation of
        `__add__()` and `__mul__()` for a valid matrix product. This method
        attempts to generalize by using a left fold for summation.

        If the operand matrices are of empty shapes `(M, 0)` and `(0, N)`,
        respectively, the product will be an `(M, N)` matrix filled with
        `None`.
        """
        if not isinstance(other, Matrix):
            return NotImplemented

        items, shape = matmul_iterator(self, other)

        data = list(items)
        return self.__class__.new(data, shape)

    def __radd__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the reverse mapping of `operator.add()`"""
        radd = binary_reverse(operator.add)
        return self.map(radd, other)

    def __rsub__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the reverse mapping of `operator.sub()`"""
        rsub = binary_reverse(operator.sub)
        return self.map(rsub, other)

    def __rmul__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the reverse mapping of `operator.mul()`"""
        rmul = binary_reverse(operator.mul)
        return self.map(rmul, other)

    def __rtruediv__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the reverse mapping of `operator.truediv()`"""
        rtruediv = binary_reverse(operator.truediv)
        return self.map(rtruediv, other)

    def __rfloordiv__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the reverse mapping of `operator.floordiv()"""
        rfloordiv = binary_reverse(operator.floordiv)
        return self.map(rfloordiv, other)

    def __rmod__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the reverse mapping of `operator.mod()`"""
        rmod = binary_reverse(operator.mod)
        return self.map(rmod, other)

    def __rpow__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the reverse mapping of `operator.pow()`"""
        rpow = binary_reverse(operator.pow)
        return self.map(rpow, other)

    def __rand__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[bool]:
        """Return the reverse mapping of `logical_and()`"""
        rand = binary_reverse(logical_and)
        return self.map(rand, other)

    def __rxor__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[bool]:
        """Return the reverse mapping of `logical_xor()`"""
        rxor = binary_reverse(logical_xor)
        return self.map(rxor, other)

    def __ror__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[bool]:
        """Return the reverse mapping of `logical_or()`"""
        ror = binary_reverse(logical_or)
        return self.map(ror, other)

    def __iadd__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the application of `operator.add()`"""
        return self.apply(operator.add, other)

    def __isub__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the application of `operator.sub()`"""
        return self.apply(operator.sub, other)

    def __imul__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the application of `operator.mul()`"""
        return self.apply(operator.mul, other)

    def __itruediv__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the application of `operator.truediv()`"""
        return self.apply(operator.truediv, other)

    def __ifloordiv__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the application of `operator.floordiv()`"""
        return self.apply(operator.floordiv, other)

    def __imod__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the application of `operator.mod()`"""
        return self.apply(operator.mod, other)

    def __ipow__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[Any]:
        """Return the application of `operator.pow()`"""
        return self.apply(operator.pow, other)

    def __iand__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[bool]:
        """Return the application of `logical_and()`"""
        return self.apply(logical_and, other)

    def __ixor__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[bool]:
        """Return the application of `logical_xor()`"""
        return self.apply(logical_xor, other)

    def __ior__(self: Matrix[T], other: Matrix[Any] | Any) -> Matrix[bool]:
        """Return the application of `logical_or()`"""
        return self.apply(logical_or, other)

    def __imatmul__(self: Matrix[T], other: Matrix[Any]) -> Matrix[Any]:
        """Return an application of the matrix product

        Note that the operating matrix's shape may be altered.

        In general, elements must behave numerically in their implementation of
        `__add__()` and `__mul__()` for a valid matrix product. This method
        attempts to generalize by using a left fold for summation.

        If the operand matrices are of empty shapes `(M, 0)` and `(0, N)`,
        respectively, the product will be an `(M, N)` matrix filled with
        `None`.
        """
        if not isinstance(other, Matrix):
            return NotImplemented

        items, shape = matmul_iterator(self, other)

        self.data[:] = items
        self.shape   = shape

        return self

    def __neg__(self: Matrix[T]) -> Matrix[Any]:
        """Return the mapping of `operator.neg()`"""
        return self.map(operator.neg)

    def __pos__(self: Matrix[T]) -> Matrix[Any]:
        """Return the mapping of `operator.pos()`"""
        return self.map(operator.pos)

    def __invert__(self: Matrix[T]) -> Matrix[bool]:
        """Return the mapping of `logical_not()`"""
        return self.map(logical_not)

    def __copy__(self: Matrix[T]) -> Matrix[T]:
        """Return a shallow copy of the matrix"""
        data  = copy.copy(self.data)
        shape = self.shape
        return self.__class__.new(data, shape)

    def __deepcopy__(self: Matrix[T], memo: dict[int, Any] | None = None) -> Matrix[T]:
        """Return a deep copy of the matrix"""
        data  = copy.deepcopy(self.data, memo)
        shape = self.shape
        return self.__class__.new(data, shape)

    def index(self: Matrix[T], value: T, start: int = 0, stop: int | None = None) -> int:
        """Return the index of the first element equal to `value`

        Raises `ValueError` if `value` could not be found in the matrix.
        """
        try:                                           # typeshed doesn't hint `stop: Optional[int]` like it should
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
        self.shape = Shape(nrows, ncols)
        return self

    @typing.overload
    def map(self: Matrix[T1], func: Callable[[T1], R]) -> Matrix[R]:
        pass

    @typing.overload
    def map(self: Matrix[T1], func: Callable[[T1, T2], R], other1: Matrix[T2] | T2) -> Matrix[R]:
        pass

    @typing.overload
    def map(self: Matrix[T1], func: Callable[[T1, T2, T3], R], other1: Matrix[T2] | T2, other2: Matrix[T3] | T3) -> Matrix[R]:
        pass

    @typing.overload
    def map(self: Matrix[T1], func: Callable[..., R], *others: Matrix[Tx] | Tx) -> Matrix[R]:
        pass

    def map(self, func, *others):
        """Map `func` onto the matrix to compose a new one, optionally in
        parallel with elements and/or other matrices

        Raises `ValueError` if not all matrix operands are identical in size.
        Non-matrix operands will never raise this exception.

        The operating matrix is always selected for the resultant matrix's
        shape.
        """
        operands = []
        if others:
            m = self.size
            for i, other in enumerate(others, start=1):
                if isinstance(other, Matrix):
                    n = other.size
                    if m != n:
                        raise ValueError(f"incompatible sizes: operating matrix has size {m} but operand {i} has size {n}")
                    operand = iter(other)
                else:
                    operand = itertools.repeat(other, m)
                operands.append(operand)
        data  = list(map(func, self, *operands))
        shape = self.shape
        return self.__class__.new(data, shape)

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

    def apply(self, func, *others):
        """Map `func` onto the matrix in place, optionally in parallel with
        elements and/or other matrices

        Raises `ValueError` if not all matrix operands are identical in size.
        Non-matrix operands will never raise this exception.
        """
        operands = []
        if others:
            m = self.size
            for i, other in enumerate(others, start=1):
                if isinstance(other, Matrix):
                    n = other.size
                    if m != n:
                        raise ValueError(f"incompatible sizes: operating matrix has size {m} but operand {i} has size {n}")
                    operand = iter(other)
                else:
                    operand = itertools.repeat(other, m)
                operands.append(operand)
        for i, result in enumerate(map(func, self, *operands)):
            self.data[i] = result
        return self

    def vals(self: Matrix[T], *, by: Rule = Rule.ROW) -> Iterator[T]:
        """Return an iterator over the values of the matrix in row or
        column-major order
        """
        if by is Rule.COL:
            nrows, ncols = self.shape
            jx, ix = range(ncols), range(nrows)
            for j in jx:
                for i in ix:
                    yield self.data[i * ncols + j]
            return
        yield from self.data

    def rows(self: Matrix[T]) -> Iterator[Matrix[T]]:
        """Return an iterator over the rows of the matrix"""
        nrows, ncols = self.shape
        shape = Shape(1, ncols)

        for i in range(nrows):
            data = self.data[i * ncols : i * ncols + ncols]
            yield self.__class__.new(data, shape)

    def cols(self: Matrix[T]) -> Iterator[Matrix[T]]:
        """Return an iterator over the columns of the matrix"""
        nrows, ncols = self.shape
        shape = Shape(nrows, 1)

        for j in range(ncols):
            data = self.data[j : nrows * ncols + j : ncols]
            yield self.__class__.new(data, shape)

    def replace(self: Matrix[T], old: T, new: T, *, times: int | None = None) -> Matrix[T]:
        """Replace elements equal to `old` with `new`

        If `times` is given, only the first `times` occurrences of `old` will
        be replaced.

        This method considers two objects equal if a comparison by identity or
        equality is satisfied, meaning that some objects that can only be
        checked by identity may be replaced through this method (e.g.,
        `math.nan` and similar objects).
        """

        def indices() -> Iterator[int]:
            for i, x in enumerate(self):
                if x is old or x == old:
                    yield i

        ix = indices()

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

        def index(key: int, n: int) -> int:
            if key < 0:
                key += n
            if not (0 <= key < n):
                rule = "row" if by is Rule.ROW else "column"
                raise IndexError(f"{rule} index out of range: there are {n} {rule}s but index was {key!r}")
            return key

        i = index(i, m)
        j = index(j, m)

        if by is Rule.COL:

            def indices() -> Iterator[tuple[int, int]]:
                for k in range(n):
                    yield (k * m + i, k * m + j)

        else:

            def indices() -> Iterator[tuple[int, int]]:
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

                def indices(i: int, j: int) -> Iterator[tuple[int, int]]:
                    for k in range(n):
                        yield (k * m + i, k * m + j)

            else:

                def indices(i: int, j: int) -> Iterator[tuple[int, int]]:
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
            self.data[:] = self.vals(by=by)
            self.shape = Shape(self.size, 1)
        else:
            self.shape = Shape(1, self.size)
        return self

    def transpose(self: Matrix[T]) -> Matrix[T]:
        """Transpose the matrix in place"""
        nrows, ncols = self.shape
        if nrows > 1 and ncols > 1:
            self.data[:] = self.vals(by=Rule.COL)
        self.shape = Shape(ncols, nrows)
        return self

    def stack(self: Matrix[T], other: Matrix[T], *, by: Rule = Rule.ROW) -> Matrix[T]:
        """Stack a matrix by row or column

        Raises `ValueError` if the opposite dimension corresponding to the
        given rule differs between the operand matrices.
        """
        if self is other:
            other = other.copy()

        (m, n), (p, q) = self.shape, other.shape

        if by is Rule.COL:
            if m != p:
                raise ValueError("matrices must have an equal number of rows")
            if m > 1:
                ls, rs = iter(self), iter(other)
                self.data[:] = (
                    x
                    for _ in range(m)
                    for x in itertools.chain(itertools.islice(ls, n), itertools.islice(rs, q))
                )
            else:
                self.data.extend(other)
            n += q
        else:
            if n != q:
                raise ValueError("matrices must have an equal number of columns")
            self.data.extend(other)
            m += p

        self.shape = Shape(m, n)

        return self

    def copy(self: Matrix[T], *, deep: bool = False) -> Matrix[T]:
        """Return a shallow or deep copy of the matrix"""
        return copy.deepcopy(self) if deep else copy.copy(self)


def vector(values: Iterable[T]) -> Matrix[T]:
    """Construct a row vector from an iterable, inferring an appropriate shape"""
    data: list[T] = []

    data.extend(values)
    n = len(data)

    shape = Shape(1, n)
    return Matrix.new(data, shape)


def matrix(values: Iterable[Iterable[T]]) -> Matrix[T]:
    """Construct a matrix from a singly-nested iterable, inferring an
    appropriate shape

    Raises `ValueError` if the iterable forms an irregular matrix.
    """
    data: list[T] = []

    rows = iter(values)
    try:
        row = next(rows)
    except StopIteration:
        shape = Shape(0, 0)
        return Matrix.new(data, shape)
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

    shape = Shape(m, n)
    return Matrix.new(data, shape)
