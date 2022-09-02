from __future__ import annotations

import copy
import functools
import itertools
import operator
import reprlib
import typing
from collections.abc import Callable, Iterable, Iterator, Sequence
from enum import Enum
from io import StringIO
from typing import Any, Literal, ParamSpec, SupportsIndex, Type, TypeVar

T = TypeVar("T")
R = TypeVar("R")

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
Tx = TypeVar("Tx")

P = ParamSpec("P")


class Rule(Enum):
    """The direction by which to operate within a matrix

    The value of a rule member is usable as an index that retrieves the rule's
    corresponding dimension from a matrix's shape (or any two-element sequence
    type).

    Additional methods are included to aid in managing index resolution and row
    or column-sequencing in a one-dimensional container.
    """

    ROW: int = 0
    COL: int = 1

    def __index__(self) -> int:
        """Return the rule's value

        This exists so that members can be used directly as an index for
        sequences that coerce integer keys via `operator.index()`.
        """
        return self.value

    @property
    def inverse(self) -> Rule:
        """The rule's inverse

        The column-rule if row-rule, or the row-rule if column-rule. Equivalent
        to `Rule(not self.value)`.
        """
        return Rule(not self.value)

    @property
    def true_name(self) -> str:
        """The rule's unformatted name"""
        return "column" if self is Rule.COL else "row"

    def subshape(self, shape: Shape) -> Shape:
        """Return the rule's shape given the matrix's shape"""
        shape = shape.copy()
        shape[self] = 1
        return shape

    def serialize(self, index: int, shape: Shape) -> tuple[int, int, int]:
        """Return the start, stop, and step values required to create a range
        or slice object of the rule's shape beginning at `index`

        The input `index` must be positive - negative indices may produce
        unexpected results. This requirement is not checked for.
        """
        major_index = self.value
        minor_index = not major_index

        major = shape[major_index]
        minor = shape[minor_index]

        major_step = major_index * major + minor_index
        minor_step = minor_index * minor + major_index

        start = minor_step * index
        stop  = major_step * minor + start

        return start, stop, major_step

    def range(self, index: int, shape: Shape) -> range:
        """Return a range of indices that can be used to construct a
        sub-sequence of the rule's shape beginning at `index`

        See `serialize()` for more details.
        """
        return range(*self.serialize(index, shape))

    def slice(self, index: int, shape: Shape) -> slice:
        """Return a slice that can be used to construct a sub-sequence of the
        rule's shape beginning at `index`

        See `serialize()` for more details.
        """
        return slice(*self.serialize(index, shape))


class Shape:
    """A mutable sequence-like type for storing the dimensions of a matrix

    Instances of `Shape` support integer-indexing (both reading and writing),
    but not slice-indexing. Negative components are never checked for during
    write operations.
    """

    __slots__ = ("data",)

    def __init__(self, nrows: int = 0, ncols: int = 0) -> None:
        """Construct a shape from its two dimensions"""
        self.data = [nrows, ncols]

    def __repr__(self) -> str:
        """Return a canonical representation of the shape"""
        return f"{self.__class__.__name__}(nrows={self.nrows!r}, ncols={self.ncols!r})"

    def __str__(self) -> str:
        """Return a string representation of the shape"""
        return f"{self.nrows} × {self.ncols}"

    def __eq__(self, other: Any) -> bool:
        """Return true if the two shapes have equal dimensions, otherwise false"""
        if not isinstance(other, Shape):
            return NotImplemented
        return self.data == other.data

    def __ne__(self, other: Any) -> bool:
        """Return true if the two shapes do not have equal dimensions,
        otherwise false
        """
        if not isinstance(other, Shape):
            return NotImplemented
        return self.data != other.data

    def __len__(self) -> Literal[2]:
        """Return literal 2"""
        return 2

    def __getitem__(self, key: SupportsIndex) -> int:
        """Return the dimension corresponding to `key`"""
        key = operator.index(key)
        return self.data[key]

    def __setitem__(self, key: SupportsIndex, value: int) -> None:
        """Set the dimension corresponding to `key` with `value`"""
        key = operator.index(key)
        self.data[key] = value

    def __iter__(self) -> Iterator[int]:
        """Return an iterator over the dimensions of the shape"""
        yield from iter(self.data)

    def __reversed__(self) -> Iterator[int]:
        """Return a reversed iterator over the dimensions of the shape"""
        yield from reversed(self.data)

    def __contains__(self, value: Any) -> bool:
        """Return true if the shape contains `value`, otherwise false"""
        return value in self.data

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Shape:
        """Return a copy of the shape"""
        return self.__class__(*self)  # Our components are immutable

    __copy__ = __deepcopy__

    @property
    def nrows(self) -> int:
        """The first dimension of the shape"""
        return self.data[0]

    @nrows.setter
    def nrows(self, value: int) -> None:
        self.data[0] = value

    @property
    def ncols(self) -> int:
        """The second dimension of the shape"""
        return self.data[1]

    @ncols.setter
    def ncols(self, value: int) -> None:
        self.data[1] = value

    @property
    def size(self) -> int:
        """The product of the shape's dimensions"""
        nrows, ncols = self.data
        return nrows * ncols

    def copy(self) -> Shape:
        """Return a copy of the shape"""
        return copy.deepcopy(self)

    def reverse(self) -> Shape:
        """Reverse the shape's dimensions in place"""
        self.data.reverse()
        return self

    def resolve_index(self, key: SupportsIndex, *, by: Rule = Rule.ROW) -> int:
        """Return an index `key` as an equivalent integer, respective to a rule

        Raises `IndexError` if the key is out of range.
        """
        n = self.data[by]

        res = operator.index(key)
        res = res + (n * (res < 0))

        if res < 0 or res >= n:
            name = by.true_name
            raise IndexError(f"there are {n} {name}s but index is {key}")

        return res

    def resolve_slice(self, key: slice, *, by: Rule = Rule.ROW) -> range:
        """Return a slice `key` as an equivalent sequence of indices,
        respective to a rule
        """
        n = self.data[by]
        return range(*key.indices(n))


def binary_reverse(func: Callable[[Any, Any], R]) -> Callable[[Any, Any], R]:
    """Return a wrapper of a given binary function that reverses its incoming
    arguments
    """
    def wrapper(x: Any, y: Any) -> R:
        return func(y, x)
    return wrapper


def logical_and(a: Any, b: Any, /) -> bool: return not not (a and b)
def logical_or(a: Any, b: Any, /) -> bool: return not not (a or b)
def logical_xor(a: Any, b: Any, /) -> bool: return (not not a) is not (not not b)
def logical_not(a: Any, /) -> bool: return not a


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

    __slots__ = ("data", "shape")

    def __init__(self: Matrix[T], values: Iterable[T], nrows: int, ncols: int) -> None:
        """Construct a matrix from the elements of `values`, interpreting it as
        shape `nrows` × `ncols`
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("dimensions must be non-negative")
        data, shape = list(values), Shape(nrows, ncols)
        if (n := len(data)) != shape.size:
            raise ValueError(f"cannot interpret size {n} iterable as shape {shape}")
        self.data, self.shape = data, shape

    @classmethod
    def new(cls: Type[Matrix], data: list[T], shape: Shape) -> Matrix[T]:
        """Construct a matrix directly from a data list and its dimensions

        This method exists primarily for the benefit of matrix-producing
        functions that have "pre-validated" the data and shape. Should be used
        with caution - this method is not marked as internal because its usage
        is not entirely discouraged if you're aware of the dangers.

        The following properties are required to construct a valid matrix:
        - `data` must be a flattened `list`. That is, the elements of the
          matrix must be on the shallowest depth. A nested list would imply a
          matrix that contains `list` instances.
        - `shape` must be a `Shape`, where the product of its values must equal
          `len(data)`.
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
        data = list(itertools.repeat(value, nrows * ncols))
        return cls.new(data, shape=Shape(nrows, ncols))

    @classmethod
    def fill_like(cls: Type[Matrix], value: T, other: Matrix[Any]) -> Matrix[T]:
        """Construct a matrix of equal shape to `other`, comprised solely of
        `value`
        """
        return cls.fill(value, *other.shape)

    @property
    def size(self: Matrix[T]) -> int:
        """The product of the number of rows and columns"""
        return self.shape.size

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
        return f"{self.__class__.__name__}({self.data!r}, nrows={self.nrows!r}, ncols={self.ncols!r})"

    def __str__(self: Matrix[T]) -> str:
        """Return a string representation of the matrix

        Elements of the matrix are organized into right-justified fields of
        width 10. If an element's string representation goes beyond this width,
        it will be truncated to 9 characters, using `…` as a suffix. A single
        space is used to delineate elements.

        Customizable string formatting (through `__format__()`) is a planned
        addition. For the time being, custom formatting must be done manually.
        The implementation for this method is subject to change.
        """
        shape = self.shape
        items = iter(self)

        max_width = 10
        result = StringIO()

        if shape.size:
            for _ in range(shape.nrows):
                result.write("| ")

                for _ in range(shape.ncols):
                    chars = str(next(items))
                    if len(chars) > max_width:
                        result.write(f"{chars[:max_width - 1]}…")
                    else:
                        result.write(chars.rjust(max_width))
                    result.write(" ")

                result.write("|\n")
        else:
            result.write("Empty matrix ")

        result.write(f"({shape})")

        return result.getvalue()

    def __lt__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the flattened mapping of `operator.lt()`"""
        return self.copy().flat_map(operator.lt, other)

    def __le__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the flattened mapping of `operator.le()`"""
        return self.copy().flat_map(operator.le, other)

    def __eq__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:  # type: ignore[override]
        """Return the flattened mapping of `operator.eq()`"""
        return self.copy().flat_map(operator.eq, other)

    def __ne__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:  # type: ignore[override]
        """Return the flattened mapping of `operator.ne()`"""
        return self.copy().flat_map(operator.ne, other)

    def __gt__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the flattened mapping of `operator.gt()`"""
        return self.copy().flat_map(operator.gt, other)

    def __ge__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the flattened mapping of `operator.ge()`"""
        return self.copy().flat_map(operator.ge, other)

    def __call__(self: Matrix[Callable[P, R]], *args: P.args, **kwargs: P.kwargs) -> Matrix[R]:
        """Return a new matrix of the results from calling each element with
        the given arguments
        """
        data = [func(*args, **kwargs) for func in self.data]
        return Matrix.new(data, shape=self.shape.copy())

    def __len__(self: Matrix[T]) -> int:
        """Return the matrix's size"""
        return len(self.data)

    @typing.overload
    def __getitem__(self: Matrix[T], key: SupportsIndex) -> T:
        pass

    @typing.overload
    def __getitem__(self: Matrix[T], key: slice) -> Matrix[T]:
        pass

    @typing.overload
    def __getitem__(self: Matrix[T], key: tuple[SupportsIndex, SupportsIndex]) -> T:
        pass

    @typing.overload
    def __getitem__(self: Matrix[T], key: tuple[SupportsIndex, slice]) -> Matrix[T]:
        pass

    @typing.overload
    def __getitem__(self: Matrix[T], key: tuple[slice, SupportsIndex]) -> Matrix[T]:
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
        the second slice's range - integers are treated as length 1 slices if
        mixed with at least one other slice.
        """
        shape = self.shape

        if isinstance(key, tuple):
            rowkey, colkey = key

            w = shape.ncols
            def getitems(indices, nrows, ncols):
                data = [self.data[i * w + j] for i, j in indices]
                return Matrix.new(data, shape=Shape(nrows, ncols))

            if isinstance(rowkey, slice):
                ix = shape.resolve_slice(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = shape.resolve_slice(colkey, by=Rule.COL)
                    return getitems(
                        indices=itertools.product(ix, jx),
                        nrows=len(ix),
                        ncols=len(jx),
                    )

                else:
                    j = shape.resolve_index(colkey, by=Rule.COL)
                    return getitems(
                        indices=zip(ix, itertools.repeat(j)),
                        nrows=len(ix),
                        ncols=1,
                    )

            else:
                i = shape.resolve_index(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = shape.resolve_slice(colkey, by=Rule.COL)
                    return getitems(
                        indices=zip(itertools.repeat(i), jx),
                        nrows=1,
                        ncols=len(jx),
                    )

                else:
                    j = shape.resolve_index(colkey, by=Rule.COL)
                    return self.data[i * w + j]

        if isinstance(key, slice):
            ix = range(*key.indices(shape.size))

            data = [self.data[i] for i in ix]
            return Matrix.new(data, shape=Shape(1, len(ix)))

        try:
            value = self.data[key]
        except IndexError:
            raise IndexError("index out of range") from None
        else:
            return value

    @typing.overload
    def __setitem__(self: Matrix[T], key: SupportsIndex, value: T) -> None:
        pass

    @typing.overload
    def __setitem__(self: Matrix[T], key: slice, value: Sequence[T]) -> None:
        pass

    @typing.overload
    def __setitem__(self: Matrix[T], key: tuple[SupportsIndex, SupportsIndex], value: T) -> None:
        pass

    @typing.overload
    def __setitem__(self: Matrix[T], key: tuple[SupportsIndex, slice], value: Sequence[T]) -> None:
        pass

    @typing.overload
    def __setitem__(self: Matrix[T], key: tuple[slice, SupportsIndex], value: Sequence[T]) -> None:
        pass

    @typing.overload
    def __setitem__(self: Matrix[T], key: tuple[slice, slice], value: Sequence[T]) -> None:
        pass

    def __setitem__(self, key, value):
        """Overwrite the element or sub-matrix corresponding to `key`

        If `key` is an integer or slice, it is treated as if it is indexing the
        flattened matrix, overwriting the corresponding value(s).

        If `key` is a tuple, the first index is applied against the rows, while
        the second is applied against the columns. A tuple of two integers,
        `(i, j)`, will overwrite the element at row `i`, column `j`. All other
        tuple variations will overwrite a sub-matrix of shape `(M, N)`, where
        `M` is the length of the first slice's range, and `N` is the length of
        the second slice's range - integers are treated as length 1 slices if
        mixed with at least one other slice.
        """
        shape = self.shape

        if isinstance(key, tuple):
            rowkey, colkey = key

            w = shape.ncols
            def setitems(indices, nrows, ncols):
                if (m := nrows * ncols) != (n := len(value)):
                    raise ValueError(f"slice selected {m} items but sequence has length {n}")
                for (i, j), x in zip(indices, value):
                    self.data[i * w + j] = x
                return

            if isinstance(rowkey, slice):
                ix = shape.resolve_slice(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = shape.resolve_slice(colkey, by=Rule.COL)
                    return setitems(
                        indices=itertools.product(ix, jx),
                        nrows=len(ix),
                        ncols=len(jx),
                    )

                else:
                    j = shape.resolve_index(colkey, by=Rule.COL)
                    return setitems(
                        indices=zip(ix, itertools.repeat(j)),
                        nrows=len(ix),
                        ncols=1,
                    )

            else:
                i = shape.resolve_index(rowkey, by=Rule.ROW)

                if isinstance(colkey, slice):
                    jx = shape.resolve_slice(colkey, by=Rule.COL)
                    return setitems(
                        indices=zip(itertools.repeat(i), jx),
                        nrows=1,
                        ncols=len(jx),
                    )

                else:
                    j = shape.resolve_index(colkey, by=Rule.COL)
                    self.data[i * w + j] = value
                    return

        if isinstance(key, slice):
            ix = range(*key.indices(shape.size))

            if (m := len(ix)) != (n := len(value)):
                raise ValueError(f"slice selected {m} items but sequence has length {n}")
            for i, x in zip(ix, value):
                self.data[i] = x
            return

        try:
            self.data[key] = value
        except IndexError:
            raise IndexError("index out of range") from None
        else:
            return

    def __iter__(self: Matrix[T]) -> Iterator[T]:
        """Return an iterator over the elements of the matrix"""
        yield from iter(self.data)

    def __reversed__(self: Matrix[T]) -> Iterator[T]:
        """Return a reverse iterator over the elements of the matrix"""
        yield from reversed(self.data)

    def __contains__(self: Matrix[T], value: Any) -> bool:
        """Return true if the matrix contains `value`, otherwise false"""
        return value in self.data

    def __add__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the flattened mapping of `operator.add()`"""
        return self.copy().flat_map(operator.add, other)

    def __sub__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the flattened mapping of `operator.sub()`"""
        return self.copy().flat_map(operator.sub, other)

    def __mul__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the flattened mapping of `operator.mul()`"""
        return self.copy().flat_map(operator.mul, other)

    def __truediv__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the flattened mapping of `operator.truediv()`"""
        return self.copy().flat_map(operator.truediv, other)

    def __floordiv__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the flattened mapping of `operator.floordiv()`"""
        return self.copy().flat_map(operator.floordiv, other)

    def __mod__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the flattened mapping of `operator.mod()`"""
        return self.copy().flat_map(operator.mod, other)

    def __pow__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the flattened mapping of `operator.pow()`"""
        return self.copy().flat_map(operator.pow, other)

    def __and__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[bool]:
        """Return the flattened mapping of `logical_and()`"""
        return self.copy().flat_map(logical_and, other)

    def __xor__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[bool]:
        """Return the flattened mapping of `logical_xor()`"""
        return self.copy().flat_map(logical_xor, other)

    def __or__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[bool]:
        """Return the flattened mapping of `logical_or()`"""
        return self.copy().flat_map(logical_or, other)

    def __radd__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the reverse flattened mapping of `operator.add()`"""
        radd = binary_reverse(operator.add)
        return self.copy().flat_map(radd, other)

    def __rsub__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the reverse flattened mapping of `operator.sub()`"""
        rsub = binary_reverse(operator.sub)
        return self.copy().flat_map(rsub, other)

    def __rmul__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the reverse flattened mapping of `operator.mul()`"""
        rmul = binary_reverse(operator.mul)
        return self.copy().flat_map(rmul, other)

    def __rtruediv__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the reverse flattened mapping of `operator.truediv()`"""
        rtruediv = binary_reverse(operator.truediv)
        return self.copy().flat_map(rtruediv, other)

    def __rfloordiv__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the reverse flattened mapping of `operator.floordiv()"""
        rfloordiv = binary_reverse(operator.floordiv)
        return self.copy().flat_map(rfloordiv, other)

    def __rmod__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the reverse flattened mapping of `operator.mod()`"""
        rmod = binary_reverse(operator.mod)
        return self.copy().flat_map(rmod, other)

    def __rpow__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[Any]:
        """Return the reverse flattened mapping of `operator.pow()`"""
        rpow = binary_reverse(operator.pow)
        return self.copy().flat_map(rpow, other)

    def __rand__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[bool]:
        """Return the reverse flattened mapping of `logical_and()`"""
        rand = binary_reverse(logical_and)
        return self.copy().flat_map(rand, other)

    def __rxor__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[bool]:
        """Return the reverse flattened mapping of `logical_xor()`"""
        rxor = binary_reverse(logical_xor)
        return self.copy().flat_map(rxor, other)

    def __ror__(self: Matrix[Any], other: Sequence[Any] | Any) -> Matrix[bool]:
        """Return the reverse flattened mapping of `logical_or()`"""
        ror = binary_reverse(logical_or)
        return self.copy().flat_map(ror, other)

    def __neg__(self: Matrix[Any]) -> Matrix[Any]:
        """Return the flattened mapping of `operator.neg()`"""
        return self.copy().flat_map(operator.neg)

    def __pos__(self: Matrix[Any]) -> Matrix[Any]:
        """Return the flattened mapping of `operator.pos()`"""
        return self.copy().flat_map(operator.pos)

    def __invert__(self: Matrix[Any]) -> Matrix[bool]:
        """Return the flattened mapping of `logical_not()`"""
        return self.copy().flat_map(logical_not)

    def __matmul__(self: Matrix[Any], other: Matrix[Any]) -> Matrix[Any]:
        """Return the matrix product

        Note that, during augmented assignment, the left-hand side matrix's
        shape may be altered.

        In general, elements must behave "numerically" in their implementation
        of `__add__()` and `__mul__()` for a valid matrix product. This method
        attempts to generalize by using a left fold summation.

        If the operand matrices are of empty shapes `(M, 0)` and `(0, N)`,
        respectively, the product will be an `(M, N)` matrix filled with
        `None`.
        """
        if not isinstance(other, Matrix):
            return NotImplemented

        (m, n), (p, q) = (self.shape, other.shape)
        if n != p:
            raise ValueError("matrices must have equal inner dimensions")
        if not n:
            return Matrix.fill(None, nrows=m, ncols=q)

        ix = range(m)
        jx = range(q)
        kx = range(n)

        return Matrix.new(
            [
                functools.reduce(
                    operator.add,
                    (self.data[i * n + k] * other.data[k * q + j] for k in kx),
                )
                for i in ix
                for j in jx
            ],
            shape=Shape(m, q),
        )

    __rmatmul__ = __matmul__

    def __copy__(self: Matrix[T]) -> Matrix[T]:
        """Return a shallow copy of the matrix"""
        return Matrix.new(copy.copy(self.data), shape=self.shape.copy())

    def __deepcopy__(self: Matrix[T], memo: dict[int, Any] | None = None) -> Matrix[T]:
        """Return a deep copy of the matrix"""
        return Matrix.new(copy.deepcopy(self.data, memo), shape=self.shape.copy())

    def index(self: Matrix[T], value: T, start: int = 0, stop: int | None = None) -> int:
        """Return the index of the first element equal to `value`

        Raises `ValueError` if the value could not be found in the matrix.
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
        shape = self.shape
        if (n := shape.size) != nrows * ncols:
            raise ValueError(f"cannot re-shape size {n} matrix as shape {nrows} × {ncols}")
        shape.nrows = nrows
        shape.ncols = ncols
        return self

    def slices(self: Matrix[T], *, by: Rule = Rule.ROW) -> Iterator[Matrix[T]]:
        """Return an iterator that yields shallow copies of each row or column"""
        shape = self.shape
        subshape = by.subshape(shape)
        for i in range(shape[by]):
            data = self.data[by.slice(i, shape)]
            yield Matrix.new(data, shape=subshape.copy())

    @typing.overload
    def flat_map(
        self: Matrix[T1],
        func: Callable[[T1], T1],
    ) -> Matrix[T1]:
        pass

    @typing.overload
    def flat_map(
        self: Matrix[T1],
        func: Callable[[T1, T2], T1],
        other1: Sequence[T2] | T2,
    ) -> Matrix[T1]:
        pass

    @typing.overload
    def flat_map(
        self: Matrix[T1],
        func: Callable[[T1, T2, T3], T1],
        other1: Sequence[T2] | T2,
        other2: Sequence[T3] | T3,
    ) -> Matrix[T1]:
        pass

    @typing.overload
    def flat_map(
        self: Matrix[T1],
        func: Callable[..., T1],
        *others: Sequence[Tx] | Tx,
    ) -> Matrix[T1]:
        pass

    def flat_map(self, func, *others):
        """Map `func` across the values in parallel with other sequences and/or
        objects, writing the results to the matrix

        Raises `ValueError` if operand sequences differ in size.
        """
        itx = []
        itx.append(iter(self))

        m = len(self)
        for i, other in enumerate(others, start=1):
            if isinstance(other, Sequence):
                if m != (n := len(other)):
                    raise ValueError(f"operating matrix has size {m} but operand {i} has size {n}")
                it = iter(other)
            else:
                it = itertools.repeat(other)
            itx.append(it)

        self.data[:] = map(func, *itx)

        return self

    @typing.overload
    def map(
        self: Matrix[T1],
        func: Callable[[Matrix[T1]], Sequence[T1]],
        *,
        by: Rule = Rule.ROW,
    ) -> Matrix[T1]:
        pass

    @typing.overload
    def map(
        self: Matrix[T1],
        func: Callable[[Matrix[T1], Matrix[T2] | T2], Sequence[T1]],
        other1: Matrix[T2] | T2,
        *,
        by: Rule = Rule.ROW,
    ) -> Matrix[T1]:
        pass

    @typing.overload
    def map(
        self: Matrix[T1],
        func: Callable[[Matrix[T1], Matrix[T2] | T2, Matrix[T3] | T3], Sequence[T1]],
        other1: Matrix[T2] | T2,
        other2: Matrix[T3] | T3,
        *,
        by: Rule = Rule.ROW,
    ) -> Matrix[T1]:
        pass

    @typing.overload
    def map(
        self: Matrix[T1],
        func: Callable[..., Sequence[T1]],
        *,
        by: Rule = Rule.ROW,
    ) -> Matrix[T1]:
        pass

    def map(self, func, *others, by=Rule.ROW):
        """Map `func` across the rows or columns in parallel with other
        matrices and/or objects, writing the results to the matrix

        Raises `ValueError` if operand matrices differ in the given dimension.
        """
        shape = self.shape

        itx = []
        itx.append(self.slices(by=by))

        m = shape[by]
        for i, other in enumerate(others, start=1):
            if isinstance(other, Matrix):
                if m != (n := other.shape[by]):
                    name = by.true_name
                    raise ValueError(f"operating matrix has {m} {name}s but operand {i} has {n}")
                it = other.slices(by=by)
            else:
                it = itertools.repeat(other)
            itx.append(it)

        dy = by.inverse

        m = shape[dy]
        for i, seq in enumerate(map(func, *itx)):
            if m != (n := len(seq)):
                name = dy.true_name
                raise ValueError(f"operating matrix has {m} {name}s but a mapping result has size {n}")
            self.data[by.slice(i, shape)] = seq

        return self

    def collapse(self: Matrix[T], func: Callable[[Matrix[T]], T], *, by: Rule = Rule.ROW) -> Matrix[T]:
        """Evaluate `func` over the rows or columns, collapsing it to the
        results

        Note that, for certain empty shapes, the matrix may expand if `func`
        returns a value for zero-length sequences.
        """
        self.data[:] = map(func, self.slices(by=by))
        self.shape[by.inverse] = 1
        return self

    def mask(self: Matrix[T], selector: Sequence[Any], null: T) -> Matrix[T]:
        """Replace the elements who have a true parallel value in `selector`
        with `null`

        Raises `ValueError` if the selector differs in size.
        """
        if (m := len(self)) != (n := len(selector)):
            raise ValueError(f"operating matrix has size {m} but selector has size {n}")
        for i, masked in enumerate(selector):
            if masked: self.data[i] = null
        return self

    def replace(self: Matrix[T], old: T, new: T, *, times: int | None = None) -> Matrix[T]:
        """Replace elements equal to `old` with `new`

        If `times` is given, only the first `times` occurrences of `old` will
        be replaced.

        This method considers two objects equal if a comparison by identity or
        equality is satisfied, which can sometimes be helpful for replacing
        objects such as `math.nan`.
        """
        ix = (i for i, x in enumerate(self) if x is old or x == old)
        for i in itertools.islice(ix, times):
            self.data[i] = new
        return self

    def reverse(self: Matrix[T]) -> Matrix[T]:
        """Reverse the matrix's elements in place"""
        self.data.reverse()
        return self

    def swap(self: Matrix[T], key1: SupportsIndex, key2: SupportsIndex, *, by: Rule = Rule.ROW) -> Matrix[T]:
        """Swap the two rows or columns beginning at the given indices

        Note that, due to how `Matrix` stores its data, swapping is performed
        in linear time with respect to the specified dimension.
        """
        data, shape = self.data, self.shape
        i = shape.resolve_index(key1, by=by)
        j = shape.resolve_index(key2, by=by)
        for h, k in zip(by.range(i, shape), by.range(j, shape)):
            data[h], data[k] = data[k], data[h]
        return self

    def flip(self: Matrix[T], *, by: Rule = Rule.ROW) -> Matrix[T]:
        """Reverse the matrix's rows or columns in place"""
        data, shape = self.data, self.shape
        n = shape[by]
        for i in range(n // 2):
            j = n - i - 1
            for h, k in zip(by.range(i, shape), by.range(j, shape)):
                data[h], data[k] = data[k], data[h]
        return self

    def flatten(self: Matrix[T], *, by: Rule = Rule.ROW) -> Matrix[T]:
        """Re-shape the matrix to a row or column vector

        If flattened to a column vector, the elements are arranged into
        column-major order.
        """
        if by is Rule.COL: self.transpose()  # For column-major order
        shape = self.shape
        shape[by.inverse] = shape.size
        shape[by] = 1
        return self

    def transpose(self: Matrix[T]) -> Matrix[T]:
        """Transpose the matrix in place"""
        data, shape = self.data, self.shape

        nrows, ncols = shape
        if nrows > 1 and ncols > 1:
            ix = range(nrows)
            jx = range(ncols)
            data[:] = (data[i * ncols + j] for j in jx for i in ix)
        shape.reverse()

        return self

    def stack(self: Matrix[T], other: Sequence[T], *, by: Rule = Rule.ROW) -> Matrix[T]:
        """Stack a sequence or other matrix along the rows or columns

        Raises `ValueError` if the opposite dimension corresponding to the
        given rule differs between the operand matrices.

        If `other` is a sequence type, but not a matrix, it will be interpreted
        as a row or column vector.
        """
        if self is other: other = other.copy()  # type: ignore[attr-defined]

        dy = by.inverse

        shape = self.shape
        if isinstance(other, Matrix):
            other_shape = other.shape
        else:
            other_shape = Shape()
            other_shape[by] = 1
            other_shape[dy] = len(other)

        if (m := shape[dy]) != (n := other_shape[dy]):
            name = dy.true_name
            raise ValueError(f"operating matrix has {m} {name}s but operand has {n}")

        (m, n), (_, q) = (shape, other_shape)

        if by is Rule.COL and m > 1:
            left, right = iter(self), iter(other)
            self.data[:] = (
                x
                for _ in range(m)
                for x in itertools.chain(itertools.islice(left, n), itertools.islice(right, q))
            )
        else:
            self.data.extend(other)

        shape[by] += other_shape[by]

        return self

    def pull(self: Matrix[T], key: SupportsIndex = -1, *, by: Rule = Rule.ROW) -> Matrix[T]:
        """Remove and return the row or column corresponding to `key`

        Raises `IndexError` if the matrix is empty, or if the index is out of
        range.
        """
        shape = self.shape
        slice = by.slice(shape.resolve_index(key, by=by), shape)

        data = self.data[slice]
        del self.data[slice]

        shape[by] -= 1

        return Matrix.new(data, shape=by.subshape(shape))

    def copy(self: Matrix[T], *, deep: bool = False) -> Matrix[T]:
        """Return a shallow or deep copy of the matrix"""
        return copy.deepcopy(self) if deep else copy.copy(self)


def vector(values: Iterable[T], *, by: Rule = Rule.ROW) -> Matrix[T]:
    """Construct a row or column vector from an iterable, using its length to
    deduce its dimensions
    """
    data = list(values)

    shape = Shape()
    shape[by] = 1
    shape[by.inverse] = len(data)

    return Matrix.new(data, shape=shape)


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
        return Matrix.new(data, shape=Shape())
    else:
        data.extend(row)

    m = 1
    n = len(data)

    for m, row in enumerate(rows, start=2):
        k = 0
        for k, val in enumerate(row, start=1):
            data.append(val)
        if n != k:
            raise ValueError(f"row {m} has length {k} but precedent row(s) have length {n}")

    return Matrix.new(data, shape=Shape(m, n))
