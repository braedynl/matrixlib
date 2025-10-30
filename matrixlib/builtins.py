from __future__ import annotations

__all__ = [
    "Matrix",
    "ComplexMatrix",
    "RealMatrix",
    "IntegerMatrix",
    "vec2",
    "vec3",
]

import cmath
import functools
import itertools
import math
import operator
import random
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Reversible, Sequence
from typing import (Any, Generic, Literal, Self, SupportsFloat, SupportsIndex,
                    TypeGuard, TypeVar, cast, overload, override)

from . import exceptions
from .accessors import (AbstractAccessor, ColFlipAccessor, ColSheerAccessor,
                        ColSliceAccessor, ColVectorAccessor, IdentityAccessor,
                        MatrixAccessor, MatrixSliceAccessor, ReverseAccessor,
                        Rotate090Accessor, Rotate180Accessor,
                        Rotate270Accessor, RowFlipAccessor, RowSheerAccessor,
                        RowSliceAccessor, RowVectorAccessor, SliceAccessor,
                        TransposeAccessor, ValueAccessor)
from .exceptions import (DemotionError, MismatchedDimensionError,
                         NegativeDimensionError, ReshapeError)
from .rule import COL, ROW, Rule
from .typeshed import Sortable

type Integer = int
type Real = float | Integer
type Complex = complex | Real
type EvenNumber = Literal[-16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16]
type OddNumber = Literal[-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
type Slice = slice[SupportsIndex | None, SupportsIndex | None, SupportsIndex | None]

M_co = TypeVar("M_co", covariant=True, bound=int, default=int)
N_co = TypeVar("N_co", covariant=True, bound=int, default=int)

T_co = TypeVar("T_co", covariant=True, bound=object, default=object)
ComplexT_co = TypeVar("ComplexT_co", covariant=True, bound=Complex, default=Complex)
RealT_co = TypeVar("RealT_co", covariant=True, bound=Real, default=Real)
IntegerT_co = TypeVar("IntegerT_co", covariant=True, bound=Integer, default=Integer)


class Matrix(Sequence[T_co], Generic[M_co, N_co, T_co]):
    """A basic one and two-dimensional immutable sequence.

    Implements a "hybrid" interface, behaving like a one-dimensional sequence
    by default, and extending with two-dimensional constructs and routines.
    One-dimensional iteration is always in row-major order unless specified
    otherwise.

    Key features:
    - **Shape typing** - the first two type arguments are bound to
      the matrix's row and column count, respectively, allowing dimension
      checks to be done by the type checker when using ``typing.Literal`` or
      ``typing.NewType``. For this reason, most shape checks are done in debug
      mode only (enabled by ``-O`` during compilation).
    - **Memory preservation** - slices, permutations, and certain construction
      routines are implemented using specialized storage types, called
      "accessors". See the ``AbstractAccessor`` class for more details.
    - **Easily extendable** - this class was made with extendability in mind -
      there are very few stipulations to implementing a sub-class of your own,
      and you can even make accessor types to use for yourself.
    """

    __slots__ = ("_accessor",)
    __match_args__ = ("array", "shape")
    _accessor: AbstractAccessor[M_co, N_co, T_co]

    def __new__(cls, array: Iterable[T_co] = (), shape: tuple[M_co, N_co] = (0, 0)) -> Self:
        self = super(Matrix, cls).__new__(cls)
        array = tuple(array)
        if __debug__:
            exceptions.check_positive_shape(shape)
            test_size = shape[0] * shape[1]
            true_size = len(array)
            if true_size != test_size:
                raise ReshapeError(
                    f"array contains {true_size} values but shape implies"
                    f" {test_size}",
                )
        match shape:
            case (1, _):
                self._accessor = cast(
                    AbstractAccessor[M_co, N_co, T_co],
                    RowVectorAccessor(array),
                )
            case (_, 1):
                self._accessor = cast(
                    AbstractAccessor[M_co, N_co, T_co],
                    ColVectorAccessor(array),
                )
            case (_, _):
                self._accessor = MatrixAccessor(array, shape)
        return self

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} shape={self.shape!r}>"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, Matrix):
            return self._accessor == other._accessor
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._accessor)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        return self

    __copy__ = __deepcopy__

    def __reduce__(self) -> tuple[object, ...]:
        return (self.from_accessor, (self._accessor,))

    @override
    def __len__(self) -> int:
        return len(self._accessor)

    @overload
    def __getitem__(self, index: SupportsIndex) -> T_co: ...
    @overload
    def __getitem__(self, index: Slice) -> Matrix[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, index: tuple[SupportsIndex, SupportsIndex]) -> T_co: ...
    @overload
    def __getitem__(self, index: tuple[SupportsIndex, Slice]) -> Matrix[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, index: tuple[Slice, SupportsIndex]) -> Matrix[Any, Literal[1], T_co]: ...
    @overload
    def __getitem__(self, index: tuple[Slice, Slice]) -> Matrix[Any, Any, T_co]: ...
    @override
    def __getitem__(
        self,
        index: SupportsIndex | Slice | tuple[SupportsIndex | Slice, SupportsIndex | Slice],
    ) -> T_co | Matrix[Any, Any, T_co]:
        target = self._accessor

        if isinstance(index, tuple):
            row_index, col_index = index

            if isinstance(row_index, slice):
                row_window = target.resolve_matrix_slice(row_index, by=ROW)

                if isinstance(col_index, slice):
                    col_window = target.resolve_matrix_slice(col_index, by=COL)

                    return Matrix[Any, Any, T_co].from_accessor(
                        accessor=MatrixSliceAccessor(
                            target,
                            row_window=row_window,
                            col_window=col_window,
                        ),
                    )

                col_index = target.resolve_matrix_index(col_index, by=COL)

                return Matrix[Any, Literal[1], T_co].from_accessor(
                    accessor=ColSliceAccessor(
                        target,
                        row_window=row_window,
                        col_index=col_index,
                    ),
                )

            row_index = target.resolve_matrix_index(row_index, by=ROW)

            if isinstance(col_index, slice):
                col_window = target.resolve_matrix_slice(col_index, by=COL)

                return Matrix[Literal[1], Any, T_co].from_accessor(
                    accessor=RowSliceAccessor(
                        target,
                        row_index=row_index,
                        col_window=col_window,
                    ),
                )

            col_index = target.resolve_matrix_index(col_index, by=COL)

            return target.matrix_access(row_index, col_index)

        if isinstance(index, slice):
            window = target.resolve_vector_slice(index)

            return Matrix[Literal[1], Any, T_co].from_accessor(
                accessor=SliceAccessor(
                    target,
                    window=window,
                ),
            )

        index = target.resolve_vector_index(index)

        return target.vector_access(index)

    @override
    def __iter__(self) -> Iterator[T_co]:
        return iter(self._accessor)

    @override
    def __reversed__(self) -> Iterator[T_co]:
        return reversed(self._accessor)

    @override
    def __contains__(self, value: object) -> bool:
        return value in self._accessor

    @classmethod
    def from_accessor(cls, accessor: AbstractAccessor[M_co, N_co, T_co]) -> Self:
        """Construct a matrix from an accessor.

        **Note**: This method bypasses the default constructor, as it is
        assumed that the accessor is fully validated. This method is used for
        internal optimisations but may be employed with proper care. This
        method will never raise an exception on its own, but may cause others
        to do so (often, very mysterious ones) if the accessor does not adhere
        to accessor implementation rules.
        """
        self = super(Matrix, cls).__new__(cls)
        self._accessor = accessor
        return self

    @classmethod
    def from_matrix(cls, matrix: Matrix[M_co, N_co, T_co]) -> Self:
        """Construct a matrix by referencing another's accessor."""
        return cls.from_accessor(matrix._accessor)

    @classmethod
    def from_function(cls, function: Callable[[int, int], T_co], shape: tuple[M_co, N_co]) -> Self:
        """Construct a matrix from an index-to-value function and shape.

        The mapping function should accept a row and column index pairing, and
        return a value of type ``T_co``.

        Raises ``NegativeDimensionError`` if a dimension of ``shape`` is
        negative (debug-only).
        """
        if __debug__:
            exceptions.check_positive_shape(shape)
        return cls.from_accessor(
            accessor=MatrixAccessor(
                array=tuple(
                    function(i, j)
                    for i in range(shape[0])
                    for j in range(shape[1])
                ),
                shape=shape,
            ),
        )

    @classmethod
    def row(cls, array: Iterable[T_co]) -> Self:
        """Construct a row vector from an iterable, efficiently.

        This method saves a small amount of memory by referencing the
        underlying array size when the first dimension is requested.

        **Note**: This method does not infer its dimension types. However,
        ``M_co`` will always be ``Literal[1]``, and ``N_co`` is the length of
        ``array``.
        """
        array = tuple(array)
        return cls.from_accessor(
            accessor=cast(
                AbstractAccessor[M_co, N_co, T_co],
                RowVectorAccessor(array),
            ),
        )

    @classmethod
    def col(cls, array: Iterable[T_co]) -> Self:
        """Construct a column vector from an iterable, efficiently.

        This method saves a small amount of memory by referencing the
        underlying array size when the second dimension is requested.

        **Note**: This method does not infer its dimenion types. However,
        ``M_co`` is the length of ``array``, and ``N_co`` will always be
        ``Literal[1]``.
        """
        array = tuple(array)
        return cls.from_accessor(
            accessor=cast(
                AbstractAccessor[M_co, N_co, T_co],
                ColVectorAccessor(array),
            ),
        )

    @classmethod
    def vector(cls, array: Iterable[T_co], *, by: Rule = Rule.ROW) -> Self:
        """Construct a row or column vector from an iterable, efficiently.

        **Note**: This method does not infer its dimension types. See the
        documentation of ``row()`` and ``col()`` for details.
        """
        if by is ROW:
            self = cls.row(array)
        else:
            self = cls.col(array)
        return self

    @classmethod
    def fill(cls, value: Callable[[], T_co], shape: tuple[M_co, N_co]) -> Self:
        """Construct a matrix comprised entirely of ``value()``,
        efficiently.

        Raises ``NegativeDimensionError`` if a dimension of ``shape`` is
        negative (debug-only).
        """
        if __debug__:
            exceptions.check_positive_shape(shape)
        return cls.from_accessor(
            accessor=ValueAccessor(
                value=value(),
                shape=shape,
            ),
        )

    @classmethod
    def from_nesting(cls, nesting: Iterable[Iterable[T_co]]) -> Self:
        """Construct a matrix from a singly-nested iterable, using the
        shallowest iterable's length to deduce the number of rows, and the
        nested iterables' length to deduce the number of columns.

        Raises ``ValueError`` if the length of the nested iterables is
        inconsistent (debug-only).

        **Note**: This method does not infer its dimension types.
        """
        array: list[T_co] = []

        row_count = 0
        true_col_count = 0

        rows = iter(nesting)
        try:
            row = next(rows)
        except StopIteration:
            return cls(
                array=array,
                shape=(cast(M_co, row_count), cast(N_co, true_col_count)),
            )
        else:
            array.extend(row)

        row_count += 1
        true_col_count = len(array)

        for row in rows:
            if __debug__:
                row = tuple(row)
                test_col_count = len(row)
                if true_col_count != test_col_count:
                    raise ValueError(
                        f"row at index {row_count} has length {test_col_count},"
                        f" but precedent rows have length {true_col_count}"
                    )
            array.extend(row)
            row_count += 1

        return cls(
            array=array,
            shape=(cast(M_co, row_count), cast(N_co, true_col_count)),
        )

    @property
    def array(self) -> tuple[T_co, ...]:
        """The matrix's values as a one-dimensional ``tuple``, aligned in
        row-major order.

        To preserve memory, some methods produce a ``Matrix`` instance that
        simply references the origin matrix upon indexing. These are called
        "views". Accessing this property can vary from being O(M * N) to O(1)
        depending on whether the ``Matrix`` does ("non-material") or does not
        ("material") use a view, respectively.

        A material ``Matrix`` will already have its values stored as a
        ``tuple``, and so the property simply returns a reference to it,
        whereas a non-material ``Matrix`` will have to iterate through all
        entries and collect them into a new ``tuple`` to reflect the view's
        ordering.
        """
        return self._accessor.to_tuple()

    @property
    def shape(self) -> tuple[M_co, N_co]:
        """The matrix shape."""
        return self._accessor.shape

    @property
    def row_count(self) -> M_co:
        """The number of rows."""
        return self._accessor.row_count

    @property
    def col_count(self) -> N_co:
        """The number of columns."""
        return self._accessor.col_count

    def to_nesting(self) -> list[list[T_co]]:
        """Return a singly-nested ``list`` representation of the matrix."""
        result = list[list[T_co]]()
        row_indices = range(self.row_count)
        col_indices = range(self.col_count)
        for row_index in row_indices:
            result.append([])
            for col_index in col_indices:
                matrix_index = (row_index, col_index)
                result[row_index].append(self[matrix_index])
        return result

    def to_mapping(self) -> dict[tuple[int, int], T_co]:
        """Return an index-to-value ``dict`` representation of the matrix."""
        result = dict[tuple[int, int], T_co]()
        row_indices = range(self.row_count)
        col_indices = range(self.col_count)
        for row_index in row_indices:
            for col_index in col_indices:
                matrix_index = (row_index, col_index)
                result[matrix_index] = self[matrix_index]
        return result

    def demote(self: Matrix[Literal[1], Literal[1], T_co]) -> T_co:
        """Return the only contained value of a ``(1, 1)`` matrix.

        Raises ``DemotionError`` if the matrix does not contain exactly one
        value.
        """
        if __debug__:
            size = len(self)
            if size > 1:
                raise DemotionError(
                    "cannot demote matrix of two or more values",
                )
            if size < 1:
                raise DemotionError("cannot demote empty matrix")
        return self[0]

    def materialize(self) -> Self:
        """Return a materialized copy of the matrix.

        To preserve memory, some methods produce a ``Matrix`` instance that
        simply references the origin matrix upon indexing. These are called
        "views" - these views are allowed to refer to other views, meaning
        that long "view chains" can be made under certain circumstances. A
        sufficiently long view chain can have an impact on indexing
        performance.

        Materialization is the process of "flattening" these view chains into
        a new container. Materializing an already-materialized matrix
        effectively does nothing.

        Materializing your matrix should only ever be necessary in rare
        circumstances. Consistent use of a matrix constructed from a series of
        multiple permutations (e.g., ``transpose()``, ``flip()``, ``rotate()``)
        may be a case where materialization is warranted, though note that this
        may vastly increase your program's memory usage, especially when the
        origin matrix is still in use.

        Materialization should only be done once, prior to calculations. Think
        of it as being akin to compiling a regular expression (from the ``re``
        module) into a ``Pattern`` object.
        """
        return self.__class__(self.array, self.shape)

    def transpose(self) -> Matrix[N_co, M_co, T_co]:
        """Return a transposed view of the matrix."""
        return Matrix[N_co, M_co, T_co].from_accessor(
            accessor=TransposeAccessor(self._accessor),
        )

    def flip(self, *, by: Rule = Rule.ROW) -> Matrix[M_co, N_co, T_co]:
        """Return a flipped view of the matrix."""
        target = self._accessor
        if by is ROW:
            accessor = RowFlipAccessor(target)
        else:
            accessor = ColFlipAccessor(target)
        return Matrix[M_co, N_co, T_co].from_accessor(accessor)

    @overload
    def rotate(self, n: EvenNumber) -> Matrix[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> Matrix[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: SupportsIndex) -> Matrix[Any, Any, T_co]: ...
    @overload
    def rotate(self) -> Matrix[N_co, M_co, T_co]: ...

    def rotate(self, n: SupportsIndex = 1) -> Matrix[Any, Any, T_co]:
        """Return a rotated view of the matrix."""
        target = self._accessor
        n = operator.index(n) % 4
        if n == 0:
            return self
        elif n == 1:
            accessor = Rotate090Accessor(target)
        elif n == 2:
            accessor = Rotate180Accessor(target)
        else:
            accessor = Rotate270Accessor(target)
        return Matrix[Any, Any, T_co].from_accessor(accessor)

    def reverse(self) -> Matrix[M_co, N_co, T_co]:
        """Return a reversed view of the matrix."""
        return Matrix[M_co, N_co, T_co].from_accessor(
            accessor=ReverseAccessor(self._accessor),
        )

    def values(self, *, by: Rule = Rule.ROW, reverse: bool = False) -> Iterator[T_co]:
        """Return an iterator over the values of the matrix.

        If by ``ROW``, values are yielded in row-major order. If by ``COL``,
        column-major order.
        """
        if by is ROW:
            values = self
        else:
            values = TransposeAccessor(self._accessor)
        return iter_or(values, reverse=reverse)

    def rows(self, *, reverse: bool = False) -> Iterator[Matrix[Literal[1], N_co, T_co]]:
        """Return an iterator that yields views over the rows of the matrix."""
        target = self._accessor
        for row_index in iter_or(range(self.row_count), reverse=reverse):
            yield Matrix[Literal[1], N_co, T_co].from_accessor(
                accessor=RowSheerAccessor(target, row_index=row_index),
            )

    def cols(self, *, reverse: bool = False) -> Iterator[Matrix[M_co, Literal[1], T_co]]:
        """Return an iterator that yields views over the columns of the matrix."""
        target = self._accessor
        for col_index in iter_or(range(self.col_count), reverse=reverse):
            yield Matrix[M_co, Literal[1], T_co].from_accessor(
                accessor=ColSheerAccessor(target, col_index=col_index),
            )

    @overload
    def vectors(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[Matrix[Literal[1], N_co, T_co]]: ...
    @overload
    def vectors(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[Matrix[M_co, Literal[1], T_co]]: ...
    @overload
    def vectors(self, *, by: Rule, reverse: bool = False) -> Iterator[Matrix[Any, Any, T_co]]: ...
    @overload
    def vectors(self, *, reverse: bool = False) -> Iterator[Matrix[Literal[1], N_co, T_co]]: ...

    def vectors(self, *, by: Rule = Rule.ROW, reverse: bool = False) -> Iterator[Matrix[Any, Any, T_co]]:
        """Return an iterator that yields views over the rows or columns of the
        matrix.

        Equivalent to calling ``rows()`` or ``cols()``. See their documentation
        for more details.
        """
        return self.rows(reverse=reverse) if by is ROW else self.cols(reverse=reverse)

    def _binary_matrix_map[S, R](self, mapper: Callable[[T_co, S], R], other: Matrix[M_co, N_co, S]) -> Iterator[R]:
        """Return an iterator that computes ``mapper`` with each value of two
        matrices, in parallel.

        Raises ``MismatchedDimensionError`` if the ``other`` matrix does not
        have an equal shape (debug-only).
        """
        if __debug__:
            exceptions.check_equal_shapes(self.shape, other.shape)
        return map(mapper, self, other)

    def _binary_scalar_map[S, R](self, mapper: Callable[[T_co, S], R], other: S) -> Iterator[R]:
        """Return an iterator that computes ``mapper`` with each value of the
        matrix and a constant scalar value.
        """
        return map(mapper, self, itertools.repeat(other))

    def _binary_scalar_map_r[S, R](self, mapper: Callable[[S, T_co], R], other: S) -> Iterator[R]:
        """Return an iterator that computes ``mapper`` with each value of the
        matrix and a constant scalar value, in reverse.
        """
        return map(mapper, itertools.repeat(other), self)

    def _unary_map[R](self, mapper: Callable[[T_co], R]) -> Iterator[R]:
        """Return an iterator that computes ``mapper`` with each value of the
        matrix.
        """
        return map(mapper, self)

    @overload
    def stack[S](self, *matrices: Matrix[Any, N_co, S], by: Literal[Rule.ROW]) -> Matrix[Any, N_co, T_co | S]: ...
    @overload
    def stack[S](self, *matrices: Matrix[M_co, Any, S], by: Literal[Rule.COL]) -> Matrix[M_co, Any, T_co | S]: ...
    @overload
    def stack[S](self, *matrices: Matrix[Any, Any, S], by: Rule) -> Matrix[Any, Any, T_co | S]: ...
    @overload
    def stack[S](self, *matrices: Matrix[Any, N_co, S]) -> Matrix[Any, N_co, T_co | S]: ...

    def stack[S](self, *matrices: Matrix[Any, Any, S], by: Rule = Rule.ROW) -> Matrix[Any, Any, T_co | S]:
        """Return the matrix stacked with one or more other matrices along a
        dimension.

        Raises ``MismatchedDimensionError`` if the opposite dimension to ``by``
        is inconsistent across the given matrices (debug-only).

        **Note**: This method does not fully infer its dimension types. The
        unknown dimension is the sum of the matrices' stacking dimensions (the
        ``by`` dimension).
        """
        dy = ~by

        shape: dict[Rule, Any] = {}
        shape[by] = self.shape[by] + sum(matrix.shape[by] for matrix in matrices)
        shape[dy] = self.shape[dy]

        if __debug__:
            true_count = shape[dy]
            for i, matrix in enumerate(matrices):
                test_count = matrix.shape[dy]
                if test_count != true_count:
                    raise MismatchedDimensionError(
                        f"matrix at index {i} has {test_count} {dy.handle}s,"
                        f" but precedent matrices have {true_count}"
                    )

        interleaving = (self,) + matrices

        return Matrix(
            array=interleave(
                interleaving,
                leave_counts=tuple(
                    matrix.col_count * (matrix.row_count ** dy.value)
                    for matrix in interleaving
                ),
            ),
            shape=(shape[ROW], shape[COL]),
        )

    def replace[S](self, old: object, new: S) -> Matrix[M_co, N_co, T_co | S]:
        """Return a new matrix with values equal to ``old`` replaced with
        ``new``.
        """

        def mapper(value: T_co, old: object = old, new: S = new) -> T_co | S:
            return new if value is old or value == old else value

        return Matrix(
            array=self._unary_map(mapper),
            shape=self.shape,
        )

    def equal(self, other: object) -> Matrix[M_co, N_co, bool]:
        """Return element-wise ``a == b``."""
        if isinstance(other, Matrix):
            return Matrix(
                array=self._binary_matrix_map(
                    operator.__eq__,
                    other,
                ),
                shape=self.shape,
            )
        return Matrix(
            array=self._binary_scalar_map(
                operator.__eq__,
                other,
            ),
            shape=self.shape,
        )

    def not_equal(self, other: object) -> Matrix[M_co, N_co, bool]:
        """Return element-wise ``a != b``."""
        if isinstance(other, Matrix):
            return Matrix(
                array=self._binary_matrix_map(
                    operator.__ne__,
                    other,
                ),
                shape=self.shape,
            )
        return Matrix(
            array=self._binary_scalar_map(
                operator.__ne__,
                other,
            ),
            shape=self.shape,
        )


class ComplexMatrix(Matrix[M_co, N_co, ComplexT_co]):

    __slots__ = ()

    @overload
    def __getitem__(self, index: SupportsIndex) -> ComplexT_co: ...
    @overload
    def __getitem__(self, index: Slice) -> ComplexMatrix[Literal[1], Any, ComplexT_co]: ...
    @overload
    def __getitem__(self, index: tuple[SupportsIndex, SupportsIndex]) -> ComplexT_co: ...
    @overload
    def __getitem__(self, index: tuple[SupportsIndex, Slice]) -> ComplexMatrix[Literal[1], Any, ComplexT_co]: ...
    @overload
    def __getitem__(self, index: tuple[Slice, SupportsIndex]) -> ComplexMatrix[Any, Literal[1], ComplexT_co]: ...
    @overload
    def __getitem__(self, index: tuple[Slice, Slice]) -> ComplexMatrix[Any, Any, ComplexT_co]: ...
    @override
    def __getitem__(
        self,
        index: SupportsIndex | Slice | tuple[SupportsIndex | Slice, SupportsIndex | Slice],
    ) -> ComplexT_co | ComplexMatrix[Any, Any, ComplexT_co]:
        result = super().__getitem__(index)
        if isinstance(result, Matrix):
            return ComplexMatrix[Any, Any, ComplexT_co].from_matrix(result)
        return result

    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, Integer], other: Matrix[M_co, N_co, Integer]) -> ComplexMatrix[M_co, N_co, Integer]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, Integer], other: Integer) -> ComplexMatrix[M_co, N_co, Integer]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, Real], other: Matrix[M_co, N_co, Real]) -> ComplexMatrix[M_co, N_co, Real]: ...
    @overload
    def __add__(self: ComplexMatrix[M_co, N_co, Real], other: Real) -> ComplexMatrix[M_co, N_co, Real]: ...
    @overload
    def __add__(self, other: Matrix[M_co, N_co, Complex]) -> ComplexMatrix[M_co, N_co]: ...
    @overload
    def __add__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...

    def __add__(self, other: Matrix[M_co, N_co, Complex] | Complex) -> ComplexMatrix[M_co, N_co]:
        if is_complex_matrix(other):
            return ComplexMatrix(
                array=self._binary_matrix_map(
                    operator.__add__,
                    other,
                ),
                shape=self.shape,
            )
        if is_complex_number(other):
            return ComplexMatrix(
                array=self._binary_scalar_map(
                    operator.__add__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __radd__(self: ComplexMatrix[M_co, N_co, Integer], other: Integer) -> ComplexMatrix[M_co, N_co, Integer]: ...
    @overload
    def __radd__(self: ComplexMatrix[M_co, N_co, Real], other: Real) -> ComplexMatrix[M_co, N_co, Real]: ...
    @overload
    def __radd__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...

    def __radd__(self, other: Complex) -> ComplexMatrix[M_co, N_co]:
        if is_complex_number(other):
            return ComplexMatrix(
                array=self._binary_scalar_map_r(
                    operator.__add__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, Integer], other: Matrix[M_co, N_co, Integer]) -> ComplexMatrix[M_co, N_co, Integer]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, Integer], other: Integer) -> ComplexMatrix[M_co, N_co, Integer]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, Real], other: Matrix[M_co, N_co, Real]) -> ComplexMatrix[M_co, N_co, Real]: ...
    @overload
    def __sub__(self: ComplexMatrix[M_co, N_co, Real], other: Real) -> ComplexMatrix[M_co, N_co, Real]: ...
    @overload
    def __sub__(self, other: Matrix[M_co, N_co, Complex]) -> ComplexMatrix[M_co, N_co]: ...
    @overload
    def __sub__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...

    def __sub__(self, other: Matrix[M_co, N_co, Complex] | Complex) -> ComplexMatrix[M_co, N_co]:
        if is_complex_matrix(other):
            return ComplexMatrix(
                array=self._binary_matrix_map(
                    operator.__sub__,
                    other,
                ),
                shape=self.shape,
            )
        if is_complex_number(other):
            return ComplexMatrix(
                array=self._binary_scalar_map(
                    operator.__sub__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __rsub__(self: ComplexMatrix[M_co, N_co, Integer], other: Integer) -> ComplexMatrix[M_co, N_co, Integer]: ...
    @overload
    def __rsub__(self: ComplexMatrix[M_co, N_co, Real], other: Real) -> ComplexMatrix[M_co, N_co, Real]: ...
    @overload
    def __rsub__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...

    def __rsub__(self, other: Complex) -> ComplexMatrix[M_co, N_co]:
        if is_complex_number(other):
            return ComplexMatrix(
                array=self._binary_scalar_map_r(
                    operator.__sub__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, Integer], other: Matrix[M_co, N_co, Integer]) -> ComplexMatrix[M_co, N_co, Integer]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, Integer], other: Integer) -> ComplexMatrix[M_co, N_co, Integer]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, Real], other: Matrix[M_co, N_co, Real]) -> ComplexMatrix[M_co, N_co, Real]: ...
    @overload
    def __mul__(self: ComplexMatrix[M_co, N_co, Real], other: Real) -> ComplexMatrix[M_co, N_co, Real]: ...
    @overload
    def __mul__(self, other: Matrix[M_co, N_co, Complex]) -> ComplexMatrix[M_co, N_co]: ...
    @overload
    def __mul__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...

    def __mul__(self, other: Matrix[M_co, N_co, Complex] | Complex) -> ComplexMatrix[M_co, N_co]:
        if is_complex_matrix(other):
            return ComplexMatrix(
                array=self._binary_matrix_map(
                    operator.__mul__,
                    other,
                ),
                shape=self.shape,
            )
        if is_complex_number(other):
            return ComplexMatrix(
                array=self._binary_scalar_map(
                    operator.__mul__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __rmul__(self: ComplexMatrix[M_co, N_co, Integer], other: Integer) -> ComplexMatrix[M_co, N_co, Integer]: ...
    @overload
    def __rmul__(self: ComplexMatrix[M_co, N_co, Real], other: Real) -> ComplexMatrix[M_co, N_co, Real]: ...
    @overload
    def __rmul__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...

    def __rmul__(self, other: Complex) -> ComplexMatrix[M_co, N_co]:
        if is_complex_number(other):
            return ComplexMatrix(
                array=self._binary_scalar_map_r(
                    operator.__mul__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __truediv__(self: ComplexMatrix[M_co, N_co, Real], other: Matrix[M_co, N_co, Real]) -> ComplexMatrix[M_co, N_co, Real]: ...
    @overload
    def __truediv__(self: ComplexMatrix[M_co, N_co, Real], other: Real) -> ComplexMatrix[M_co, N_co, Real]: ...
    @overload
    def __truediv__(self, other: Matrix[M_co, N_co, Complex]) -> ComplexMatrix[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...

    def __truediv__(self, other: Matrix[M_co, N_co, Complex] | Complex) -> ComplexMatrix[M_co, N_co]:
        if is_complex_matrix(other):
            return ComplexMatrix(
                array=self._binary_matrix_map(
                    operator.__truediv__,
                    other,
                ),
                shape=self.shape,
            )
        if is_complex_number(other):
            return ComplexMatrix(
                array=self._binary_scalar_map(
                    operator.__truediv__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __rtruediv__(self: ComplexMatrix[M_co, N_co, Real], other: Real) -> ComplexMatrix[M_co, N_co, Real]: ...
    @overload
    def __rtruediv__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...

    def __rtruediv__(self, other: Complex) -> ComplexMatrix[M_co, N_co]:
        if is_complex_number(other):
            return ComplexMatrix(
                array=self._binary_scalar_map_r(
                    operator.__truediv__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __neg__(self: ComplexMatrix[M_co, N_co, Integer]) -> ComplexMatrix[M_co, N_co, Integer]: ...
    @overload
    def __neg__(self: ComplexMatrix[M_co, N_co, Real]) -> ComplexMatrix[M_co, N_co, Real]: ...
    @overload
    def __neg__(self) -> ComplexMatrix[M_co, N_co]: ...

    def __neg__(self) -> ComplexMatrix[M_co, N_co]:
        return ComplexMatrix(
            array=self._unary_map(operator.__neg__),
            shape=self.shape,
        )

    @overload
    def __pos__(self: ComplexMatrix[M_co, N_co, Integer]) -> ComplexMatrix[M_co, N_co, Integer]: ...
    @overload
    def __pos__(self: ComplexMatrix[M_co, N_co, Real]) -> ComplexMatrix[M_co, N_co, Real]: ...
    @overload
    def __pos__(self) -> ComplexMatrix[M_co, N_co]: ...

    def __pos__(self) -> ComplexMatrix[M_co, N_co]:
        return ComplexMatrix(
            array=self._unary_map(operator.__pos__),
            shape=self.shape,
        )

    # Absolute value of a complex number is its distance from the origin,
    # so return a RealMatrix.

    @overload
    def __abs__(self: ComplexMatrix[M_co, N_co, Integer]) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __abs__(self) -> RealMatrix[M_co, N_co]: ...

    def __abs__(self) -> RealMatrix[M_co, N_co]:
        return RealMatrix(
            array=self._unary_map(abs),
            shape=self.shape,
        )

    def __complex__(self: ComplexMatrix[Literal[1], Literal[1], ComplexT_co]) -> complex:
        return complex(self.demote())

    @property
    def real(self) -> RealMatrix[M_co, N_co]:
        """The matrix's real components."""
        return RealMatrix(
            array=self._unary_map(lambda x: x.real),
            shape=self.shape,
        )

    @property
    def imag(self) -> RealMatrix[M_co, N_co]:
        """The matrix's imaginary components."""
        return RealMatrix(
            array=self._unary_map(lambda x: x.imag),
            shape=self.shape,
        )

    @override
    def transpose(self) -> ComplexMatrix[N_co, M_co, ComplexT_co]:
        return ComplexMatrix[N_co, M_co, ComplexT_co].from_matrix(super().transpose())

    @override
    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrix[M_co, N_co, ComplexT_co]:
        return ComplexMatrix[M_co, N_co, ComplexT_co].from_matrix(super().flip(by=by))

    @overload
    def rotate(self, n: EvenNumber) -> ComplexMatrix[M_co, N_co, ComplexT_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> ComplexMatrix[N_co, M_co, ComplexT_co]: ...
    @overload
    def rotate(self, n: SupportsIndex) -> ComplexMatrix[Any, Any, ComplexT_co]: ...
    @overload
    def rotate(self) -> ComplexMatrix[N_co, M_co, ComplexT_co]: ...
    @override
    def rotate(self, n: SupportsIndex = 1) -> ComplexMatrix[Any, Any, ComplexT_co]:
        return ComplexMatrix[Any, Any, ComplexT_co].from_matrix(super().rotate(n=n))

    @override
    def reverse(self) -> ComplexMatrix[M_co, N_co, ComplexT_co]:
        return ComplexMatrix[M_co, N_co, ComplexT_co].from_matrix(super().reverse())

    @override
    def rows(self, *, reverse: bool = False) -> Iterator[ComplexMatrix[Literal[1], N_co, ComplexT_co]]:
        return map(ComplexMatrix[Literal[1], N_co, ComplexT_co].from_matrix, super().rows(reverse=reverse))

    @override
    def cols(self, *, reverse: bool = False) -> Iterator[ComplexMatrix[M_co, Literal[1], ComplexT_co]]:
        return map(ComplexMatrix[M_co, Literal[1], ComplexT_co].from_matrix, super().cols(reverse=reverse))

    @overload
    def vectors(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[ComplexMatrix[Literal[1], N_co, ComplexT_co]]: ...
    @overload
    def vectors(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[ComplexMatrix[M_co, Literal[1], ComplexT_co]]: ...
    @overload
    def vectors(self, *, by: Rule, reverse: bool = False) -> Iterator[ComplexMatrix[Any, Any, ComplexT_co]]: ...
    @overload
    def vectors(self, *, reverse: bool = False) -> Iterator[ComplexMatrix[Literal[1], N_co, ComplexT_co]]: ...
    @override
    def vectors(self, *, by: Rule = Rule.ROW, reverse: bool = False) -> Iterator[ComplexMatrix[Any, Any, ComplexT_co]]:
        return map(ComplexMatrix[Any, Any, ComplexT_co].from_matrix, super().vectors(by=by, reverse=reverse))

    @overload
    def conjugate(self: ComplexMatrix[M_co, N_co, Integer]) -> ComplexMatrix[M_co, N_co, Integer]: ...
    @overload
    def conjugate(self: ComplexMatrix[M_co, N_co, Real]) -> ComplexMatrix[M_co, N_co, Real]: ...
    @overload
    def conjugate(self) -> ComplexMatrix[M_co, N_co]: ...

    def conjugate(self) -> ComplexMatrix[M_co, N_co]:
        """Return the complex conjugate of the matrix."""
        return ComplexMatrix(
            array=self._unary_map(lambda x: x.conjugate()),
            shape=self.shape,
        )

    @overload
    def transjugate(self: ComplexMatrix[M_co, N_co, Integer]) -> ComplexMatrix[N_co, M_co, Integer]: ...
    @overload
    def transjugate(self: ComplexMatrix[M_co, N_co, Real]) -> ComplexMatrix[N_co, M_co, Real]: ...
    @overload
    def transjugate(self) -> ComplexMatrix[N_co, M_co]: ...

    def transjugate(self) -> ComplexMatrix[N_co, M_co]:
        """Return the transposed complex conjugate of the matrix."""
        return self.transpose().conjugate()

    def is_close(
        self,
        other: Matrix[M_co, N_co, Complex] | Complex,
        *,
        rel_tol: SupportsFloat = 1e-09,
        abs_tol: SupportsFloat = 0.0,
    ) -> Matrix[M_co, N_co, bool]:
        """Return an element-wise "close" comparison with another matrix or
        scalar.

        Internally delegates to ``cmath.isclose()``. See its documentation for
        more details.

        Raises ``MismatchedDimensionError`` if ``other`` is a ``ComplexMatrix``
        of unequal shape (debug-only).
        """
        is_close = functools.partial(cmath.isclose, rel_tol=rel_tol, abs_tol=abs_tol)
        if is_complex_matrix(other):
            return Matrix(
                array=self._binary_matrix_map(
                    is_close,
                    other,
                ),
                shape=self.shape,
            )
        if is_complex_number(other):
            return Matrix(
                array=self._binary_scalar_map(
                    is_close,
                    other,
                ),
                shape=self.shape,
            )
        raise TypeError


class RealMatrix(ComplexMatrix[M_co, N_co, RealT_co]):

    __slots__ = ()

    @classmethod
    def random(cls, shape: tuple[M_co, N_co]) -> RealMatrix[M_co, N_co]:
        """Construct a matrix of random numbers within the range [0, 1).

        Internally uses built-in ``random.random()``, thus inheriting the
        state of the global random number generator.

        **Note**: Unlike most other class methods, this one always returns a
        ``RealMatrix`` unless overriden by a child class.
        """
        if __debug__:
            exceptions.check_positive_shape(shape)
        return RealMatrix[M_co, N_co].from_accessor(
            accessor=MatrixAccessor(
                array=tuple(
                    random.random()
                    for _ in range(shape[0] * shape[1])
                ),
                shape=shape,
            ),
        )

    def __lt__(self, other: RealMatrix) -> bool:
        """Return true if lexicographic ``a < b``, otherwise false"""
        if is_real_matrix(other):
            return self.compare(other) < 0
        return NotImplemented

    def __le__(self, other: RealMatrix) -> bool:
        """Return true if lexicographic ``a <= b``, otherwise false"""
        if is_real_matrix(other):
            return self.compare(other) <= 0
        return NotImplemented

    def __gt__(self, other: RealMatrix) -> bool:
        """Return true if lexicographic ``a > b``, otherwise false"""
        if is_real_matrix(other):
            return self.compare(other) > 0
        return NotImplemented

    def __ge__(self, other: RealMatrix) -> bool:
        """Return true if lexicographic ``a >= b``, otherwise false"""
        if is_real_matrix(other):
            return self.compare(other) >= 0
        return NotImplemented

    @overload
    def __getitem__(self, index: SupportsIndex) -> RealT_co: ...
    @overload
    def __getitem__(self, index: Slice) -> RealMatrix[Literal[1], Any, RealT_co]: ...
    @overload
    def __getitem__(self, index: tuple[SupportsIndex, SupportsIndex]) -> RealT_co: ...
    @overload
    def __getitem__(self, index: tuple[SupportsIndex, Slice]) -> RealMatrix[Literal[1], Any, RealT_co]: ...
    @overload
    def __getitem__(self, index: tuple[Slice, SupportsIndex]) -> RealMatrix[Any, Literal[1], RealT_co]: ...
    @overload
    def __getitem__(self, index: tuple[Slice, Slice]) -> RealMatrix[Any, Any, RealT_co]: ...
    @override
    def __getitem__(
        self,
        index: SupportsIndex | Slice | tuple[SupportsIndex | Slice, SupportsIndex | Slice],
    ) -> RealT_co | RealMatrix[Any, Any, RealT_co]:
        result = super().__getitem__(index)
        if isinstance(result, Matrix):
            return RealMatrix[Any, Any, RealT_co].from_matrix(result)
        return result

    @overload
    def __add__(self: RealMatrix[M_co, N_co, Integer], other: Matrix[M_co, N_co, Integer]) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __add__(self: RealMatrix[M_co, N_co, Integer], other: Integer) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __add__(self, other: Matrix[M_co, N_co, Real]) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __add__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __add__(self, other: Matrix[M_co, N_co, Complex]) -> ComplexMatrix[M_co, N_co]: ...
    @overload
    def __add__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...
    @override
    def __add__(self, other: Matrix[M_co, N_co, Complex] | Complex) -> ComplexMatrix[M_co, N_co]:
        if is_real_object(other):
            return RealMatrix[M_co, N_co].from_matrix(super().__add__(other))
        return super().__add__(other)

    @overload
    def __radd__(self: RealMatrix[M_co, N_co, Integer], other: Integer) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __radd__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __radd__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...
    @override
    def __radd__(self, other: Complex) -> ComplexMatrix[M_co, N_co]:
        if is_real_number(other):
            return RealMatrix[M_co, N_co].from_matrix(super().__radd__(other))
        return super().__radd__(other)

    @overload
    def __sub__(self: RealMatrix[M_co, N_co, Integer], other: Matrix[M_co, N_co, Integer]) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __sub__(self: RealMatrix[M_co, N_co, Integer], other: Integer) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __sub__(self, other: Matrix[M_co, N_co, Real]) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __sub__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __sub__(self, other: Matrix[M_co, N_co, Complex]) -> ComplexMatrix[M_co, N_co]: ...
    @overload
    def __sub__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...
    @override
    def __sub__(self, other: Matrix[M_co, N_co, Complex] | Complex) -> ComplexMatrix[M_co, N_co]:
        if is_real_object(other):
            return RealMatrix[M_co, N_co].from_matrix(super().__sub__(other))
        return super().__sub__(other)

    @overload
    def __rsub__(self: RealMatrix[M_co, N_co, Integer], other: Integer) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __rsub__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __rsub__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...
    @override
    def __rsub__(self, other: Complex) -> ComplexMatrix[M_co, N_co]:
        if is_real_number(other):
            return RealMatrix[M_co, N_co].from_matrix(super().__rsub__(other))
        return super().__rsub__(other)

    @overload
    def __mul__(self: RealMatrix[M_co, N_co, Integer], other: Matrix[M_co, N_co, Integer]) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __mul__(self: RealMatrix[M_co, N_co, Integer], other: Integer) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __mul__(self, other: Matrix[M_co, N_co, Real]) -> RealMatrix[M_co, N_co, Real]: ...
    @overload
    def __mul__(self, other: Real) -> RealMatrix[M_co, N_co, Real]: ...
    @overload
    def __mul__(self, other: Matrix[M_co, N_co, Complex]) -> ComplexMatrix[M_co, N_co]: ...
    @overload
    def __mul__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...
    @override
    def __mul__(self, other: Matrix[M_co, N_co, Complex] | Complex) -> ComplexMatrix[M_co, N_co]:
        if is_real_object(other):
            return RealMatrix[M_co, N_co].from_matrix(super().__mul__(other))
        return super().__mul__(other)

    @overload
    def __rmul__(self: RealMatrix[M_co, N_co, Integer], other: Integer) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __rmul__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __rmul__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...
    @override
    def __rmul__(self, other: Complex) -> ComplexMatrix[M_co, N_co]:
        if is_real_number(other):
            return RealMatrix[M_co, N_co].from_matrix(super().__rmul__(other))
        return super().__rmul__(other)

    @overload
    def __matmul__[P: int](self: RealMatrix[M_co, N_co, Integer], other: Matrix[N_co, P, Integer]) -> RealMatrix[M_co, P, Integer]: ...
    @overload
    def __matmul__[P: int](self, other: Matrix[N_co, P, Real]) -> RealMatrix[M_co, P]: ...

    def __matmul__[P: int](self, other: Matrix[N_co, P, Real]) -> RealMatrix[M_co, P]:
        if not is_real_matrix(other):
            return NotImplemented

        a = self
        mn = self.shape

        b = other
        np = other.shape

        (m, n1), (n2, p) = mn, np

        if __debug__:
            if n1 != n2:
                raise MismatchedDimensionError(
                    f"cannot multiply matrices with mismatched inner"
                    f" dimensions, left operand has {n1} columns but right"
                    f" operand has {n2} rows"
                )

        if not n1:
            return IntegerMatrix[M_co, P].zeroes((m, p))

        return RealMatrix[M_co, P](
            array=(
                math.sumprod(row, col)
                for row in a.rows()
                for col in b.cols()
            ),
            shape=(m, p),
        )

    @overload
    def __truediv__(self, other: Matrix[M_co, N_co, Real]) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: Matrix[M_co, N_co, Complex]) -> ComplexMatrix[M_co, N_co]: ...
    @overload
    def __truediv__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...
    @override
    def __truediv__(self, other: Matrix[M_co, N_co, Complex] | Complex) -> ComplexMatrix[M_co, N_co]:
        if is_real_object(other):
            return RealMatrix[M_co, N_co].from_matrix(super().__truediv__(other))
        return super().__truediv__(other)

    @overload
    def __rtruediv__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __rtruediv__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...
    @override
    def __rtruediv__(self, other: Complex) -> ComplexMatrix[M_co, N_co]:
        if is_real_number(other):
            return RealMatrix[M_co, N_co].from_matrix(super().__rtruediv__(other))
        return super().__rtruediv__(other)

    @overload
    def __floordiv__(self: RealMatrix[M_co, N_co, Integer], other: Matrix[M_co, N_co, Integer]) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __floordiv__(self: RealMatrix[M_co, N_co, Integer], other: Integer) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __floordiv__(self, other: Matrix[M_co, N_co, Real]) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __floordiv__(self, other: Real) -> RealMatrix[M_co, N_co]: ...

    def __floordiv__(self, other: Matrix[M_co, N_co, Real] | Real) -> RealMatrix[M_co, N_co]:
        if is_real_matrix(other):
            return RealMatrix(
                array=self._binary_matrix_map(
                    operator.__floordiv__,
                    other,
                ),
                shape=self.shape,
            )
        if is_real_number(other):
            return RealMatrix(
                array=self._binary_scalar_map(
                    operator.__floordiv__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __rfloordiv__(self: RealMatrix[M_co, N_co, Integer], other: Integer) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __rfloordiv__(self, other: Real) -> RealMatrix[M_co, N_co]: ...

    def __rfloordiv__(self, other: Real) -> RealMatrix[M_co, N_co]:
        if is_real_number(other):
            return RealMatrix(
                array=self._binary_scalar_map_r(
                    operator.__floordiv__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __mod__(self: RealMatrix[M_co, N_co, Integer], other: Matrix[M_co, N_co, Integer]) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __mod__(self: RealMatrix[M_co, N_co, Integer], other: Integer) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __mod__(self, other: Matrix[M_co, N_co, Real]) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __mod__(self, other: Real) -> RealMatrix[M_co, N_co]: ...

    def __mod__(self, other: Matrix[M_co, N_co, Real] | Real) -> RealMatrix[M_co, N_co]:
        if is_real_matrix(other):
            return RealMatrix(
                array=self._binary_matrix_map(
                    operator.__mod__,
                    other,
                ),
                shape=self.shape,
            )
        if is_real_number(other):
            return RealMatrix(
                array=self._binary_scalar_map(
                    operator.__mod__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __rmod__(self: RealMatrix[M_co, N_co, Integer], other: Integer) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __rmod__(self, other: Real) -> RealMatrix[M_co, N_co]: ...

    def __rmod__(self, other: Real) -> RealMatrix[M_co, N_co]:
        if is_real_number(other):
            return RealMatrix(
                array=self._binary_scalar_map_r(
                    operator.__mod__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __divmod__(self: RealMatrix[M_co, N_co, Integer], other: Matrix[M_co, N_co, Integer]) -> tuple[RealMatrix[M_co, N_co, Integer], RealMatrix[M_co, N_co, Integer]]: ...
    @overload
    def __divmod__(self: RealMatrix[M_co, N_co, Integer], other: Integer) -> tuple[RealMatrix[M_co, N_co, Integer], RealMatrix[M_co, N_co, Integer]]: ...
    @overload
    def __divmod__(self, other: Matrix[M_co, N_co, Real]) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...
    @overload
    def __divmod__(self, other: Real) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...

    def __divmod__(self, other: Matrix[M_co, N_co, Real] | Real) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]:
        if is_real_object(other):
            return (self // other, self % other)
        return NotImplemented

    @overload
    def __rdivmod__(self: RealMatrix[M_co, N_co, Integer], other: Integer) -> tuple[RealMatrix[M_co, N_co, Integer], RealMatrix[M_co, N_co, Integer]]: ...
    @overload
    def __rdivmod__(self, other: Real) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...

    def __rdivmod__(self, other: Real) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]:
        if is_real_number(other):
            return (other // self, other % self)
        return NotImplemented

    # NOTE: In Pyright, keeping the ComplexMatrix -> ComplexMatrix overload
    # warns that it overlaps with the RealMatrix -> RealMatrix overload, which
    # is true. Removing it, however, warns that we aren't covering all of the
    # overloads of the parent class, which is also true. Thus, we shall ignore.

    # Oddly, Pyright does not get upset about IntegerMatrix's version of this
    # overload set (which is just IntegerMatrix -> IntegerMatrix). It's only an
    # issue for RealMatrix. Likely a bug on Pyright's end.

    @overload
    def __neg__(self: RealMatrix[M_co, N_co, Integer]) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __neg__(self) -> RealMatrix[M_co, N_co]: ...
    @override
    def __neg__(self) -> RealMatrix[M_co, N_co]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return RealMatrix[M_co, N_co].from_matrix(super().__neg__())

    @overload
    def __pos__(self: RealMatrix[M_co, N_co, Integer]) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __pos__(self) -> RealMatrix[M_co, N_co]: ...
    @override
    def __pos__(self) -> RealMatrix[M_co, N_co]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return RealMatrix[M_co, N_co].from_matrix(super().__pos__())

    def __float__(self: RealMatrix[Literal[1], Literal[1], RealT_co]) -> float:
        return float(self.demote())

    @overload
    def __round__(self: RealMatrix[Literal[1], Literal[1], RealT_co], ndigits: None = None) -> int: ...
    @overload
    def __round__(self: RealMatrix[Literal[1], Literal[1], RealT_co], ndigits: SupportsIndex) -> float: ...

    def __round__(self: RealMatrix[Literal[1], Literal[1], RealT_co], ndigits: SupportsIndex | None = None) -> float:
        return round(self.demote(), ndigits)

    def __floor__(self: RealMatrix[Literal[1], Literal[1], RealT_co]) -> int:
        return math.floor(self.demote())

    def __ceil__(self: RealMatrix[Literal[1], Literal[1], RealT_co]) -> int:
        return math.ceil(self.demote())

    def __trunc__(self: RealMatrix[Literal[1], Literal[1], RealT_co]) -> int:
        return math.trunc(self.demote())

    @property
    @override
    def real(self) -> RealMatrix[M_co, N_co]:
        return self

    @property
    @override
    def imag(self) -> IntegerMatrix[M_co, N_co]:
        return IntegerMatrix[M_co, N_co].zeroes(self.shape)

    @override
    def transpose(self) -> RealMatrix[N_co, M_co, RealT_co]:
        return RealMatrix[N_co, M_co, RealT_co].from_matrix(super().transpose())

    @override
    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrix[M_co, N_co, RealT_co]:
        return RealMatrix[M_co, N_co, RealT_co].from_matrix(super().flip(by=by))

    @overload
    def rotate(self, n: EvenNumber) -> RealMatrix[M_co, N_co, RealT_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> RealMatrix[N_co, M_co, RealT_co]: ...
    @overload
    def rotate(self, n: SupportsIndex) -> RealMatrix[Any, Any, RealT_co]: ...
    @overload
    def rotate(self) -> RealMatrix[N_co, M_co, RealT_co]: ...
    @override
    def rotate(self, n: SupportsIndex = 1) -> RealMatrix[Any, Any, RealT_co]:
        return RealMatrix[Any, Any, RealT_co].from_matrix(super().rotate(n=n))

    @override
    def reverse(self) -> RealMatrix[M_co, N_co, RealT_co]:
        return RealMatrix[M_co, N_co, RealT_co].from_matrix(super().reverse())

    @override
    def rows(self, *, reverse: bool = False) -> Iterator[RealMatrix[Literal[1], N_co, RealT_co]]:
        return map(RealMatrix[Literal[1], N_co, RealT_co].from_matrix, super().rows(reverse=reverse))

    @override
    def cols(self, *, reverse: bool = False) -> Iterator[RealMatrix[M_co, Literal[1], RealT_co]]:
        return map(RealMatrix[M_co, Literal[1], RealT_co].from_matrix, super().cols(reverse=reverse))

    @overload
    def vectors(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[RealMatrix[Literal[1], N_co, RealT_co]]: ...
    @overload
    def vectors(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[RealMatrix[M_co, Literal[1], RealT_co]]: ...
    @overload
    def vectors(self, *, by: Rule, reverse: bool = False) -> Iterator[RealMatrix[Any, Any, RealT_co]]: ...
    @overload
    def vectors(self, *, reverse: bool = False) -> Iterator[RealMatrix[Literal[1], N_co, RealT_co]]: ...
    @override
    def vectors(self, *, by: Rule = Rule.ROW, reverse: bool = False) -> Iterator[RealMatrix[Any, Any, RealT_co]]:
        return map(RealMatrix[Any, Any, RealT_co].from_matrix, super().vectors(by=by, reverse=reverse))

    # NOTE: Same typing problem as with __neg__() and __pos__() - see note
    # above them for more details.

    @overload
    def conjugate(self: RealMatrix[M_co, N_co, Integer]) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def conjugate(self) -> RealMatrix[M_co, N_co]: ...
    @override
    def conjugate(self) -> RealMatrix[M_co, N_co]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self

    @overload
    def transjugate(self: RealMatrix[M_co, N_co, Integer]) -> RealMatrix[N_co, M_co, Integer]: ...
    @overload
    def transjugate(self) -> RealMatrix[N_co, M_co]: ...
    @override
    def transjugate(self) -> RealMatrix[N_co, M_co]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return RealMatrix[N_co, M_co].from_matrix(super().transjugate())

    def compare(self, other: Matrix[int, int, Real]) -> Literal[-1, 0, 1]:
        """Return literal ``-1``, ``0``, or ``+1`` if the matrix
        lexicographically compares less than, equal, or greater than ``other``,
        respectively

        Matrices are lexicographically compared values first, shapes second -
        similar to how built-in sequences compare values first, lengths second.
        """
        def compare(a: Iterable[Real], b: Iterable[Real]) -> Literal[-1, 0, 1]:
            if a is b:
                return 0
            for x, y in zip(a, b):
                if x is y or x == y:
                    continue
                return -1 if x < y else 1
            return 0
        return (compare(self, other) or compare(self.shape, other.shape))

    def lesser(self, other: Matrix[M_co, N_co, Real] | Real) -> Matrix[M_co, N_co, bool]:
        """Return element-wise ``a < b``."""
        if is_real_matrix(other):
            return Matrix(
                array=self._binary_matrix_map(
                    operator.__lt__,
                    other,
                ),
                shape=self.shape,
            )
        if is_real_number(other):
            return Matrix(
                array=self._binary_scalar_map(
                    operator.__lt__,
                    other,
                ),
                shape=self.shape,
            )
        raise TypeError

    def lesser_equal(self, other: Matrix[M_co, N_co, Real] | Real) -> Matrix[M_co, N_co, bool]:
        """Return element-wise ``a <= b``."""
        if is_real_matrix(other):
            return Matrix(
                array=self._binary_matrix_map(
                    operator.__le__,
                    other,
                ),
                shape=self.shape,
            )
        if is_real_number(other):
            return Matrix(
                array=self._binary_scalar_map(
                    operator.__le__,
                    other,
                ),
                shape=self.shape,
            )
        raise TypeError

    def greater(self, other: Matrix[M_co, N_co, Real] | Real) -> Matrix[M_co, N_co, bool]:
        """Return element-wise ``a > b``."""
        if is_real_matrix(other):
            return Matrix(
                array=self._binary_matrix_map(
                    operator.__gt__,
                    other,
                ),
                shape=self.shape,
            )
        if is_real_number(other):
            return Matrix(
                array=self._binary_scalar_map(
                    operator.__gt__,
                    other,
                ),
                shape=self.shape,
            )
        raise TypeError

    def greater_equal(self, other: Matrix[M_co, N_co, Real] | Real) -> Matrix[M_co, N_co, bool]:
        """Return element-wise ``a >= b``."""
        if is_real_matrix(other):
            return Matrix(
                array=self._binary_matrix_map(
                    operator.__ge__,
                    other,
                ),
                shape=self.shape,
            )
        if is_real_number(other):
            return Matrix(
                array=self._binary_scalar_map(
                    operator.__ge__,
                    other,
                ),
                shape=self.shape,
            )
        raise TypeError

    @overload
    def round(self, ndigits: None = None) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def round(self, ndigits: SupportsIndex) -> RealMatrix[M_co, N_co]: ...

    def round(self, ndigits: SupportsIndex | None = None) -> RealMatrix[M_co, N_co]:
        """Return the matrix with its values rounded to ``ndigits`` precision
        after the decimal point.

        Internally delegates to built-in ``round()``. See its documentation for
        more details.
        """
        if ndigits is None:
            return IntegerMatrix(
                array=self._unary_map(round),
                shape=self.shape,
            )
        return RealMatrix(
            array=self._unary_map(
                functools.partial(round, ndigits=ndigits),
            ),
            shape=self.shape,
        )

    def floor(self) -> IntegerMatrix[M_co, N_co]:
        """Return the matrix with its values floored."""
        return IntegerMatrix(
            array=self._unary_map(math.floor),
            shape=self.shape,
        )

    def ceil(self) -> IntegerMatrix[M_co, N_co]:
        """Return the matrix with its values ceiled."""
        return IntegerMatrix(
            array=self._unary_map(math.ceil),
            shape=self.shape,
        )

    def trunc(self) -> IntegerMatrix[M_co, N_co]:
        """Return the matrix with its values truncated."""
        return IntegerMatrix(
            array=self._unary_map(math.trunc),
            shape=self.shape,
        )

    def sort(
        self,
        *,
        key: Callable[[RealT_co], Sortable] | None = None,
        reverse: bool = False,
    ) -> RealMatrix[M_co, N_co, RealT_co]:
        """Return the matrix sorted.

        Internally delegates to built-in ``sorted()``. See its documentation
        for more details.
        """
        return RealMatrix(
            array=sorted(self, key=key, reverse=reverse),
            shape=self.shape,
        )


class IntegerMatrix(RealMatrix[M_co, N_co, IntegerT_co]):

    __slots__ = ()

    @classmethod
    def identity[N: int](
        cls: type[IntegerMatrix[N, N, IntegerT_co]],
        count: N,
    ) -> IntegerMatrix[N, N]:
        """Construct an identity matrix, efficiently.

        Raises ``NegativeDimensionError`` if ``count`` is negative
        (debug-only).

        **Note**: Unlike most other class methods, this one always returns an
        ``IntegerMatrix`` unless overriden by a child class.
        """
        if __debug__:
            if count < 0:
                raise NegativeDimensionError("shape dimensions must be positive")
        return IntegerMatrix[N, N].from_accessor(
            accessor=IdentityAccessor(
                non_zero_value=1,
                shape=(count, count),
                zero_value=0,
            ),
        )

    @classmethod
    def zeroes(cls, shape: tuple[M_co, N_co]) -> IntegerMatrix[M_co, N_co]:
        """Construct a matrix comprised entirely of zeroes, efficiently.

        **Note**: Unlike most other class methods, this one always returns an
        ``IntegerMatrix`` unless overriden by a child class.
        """
        return IntegerMatrix[M_co, N_co].fill(lambda: 0, shape)

    @classmethod
    def ones(cls, shape: tuple[M_co, N_co]) -> IntegerMatrix[M_co, N_co]:
        """Construct a matrix comprised entirely of ones, efficiently.

        **Note**: Unlike most other class methods, this one always returns an
        ``IntegerMatrix`` unless overriden by a child class.
        """
        return IntegerMatrix[M_co, N_co].fill(lambda: 1, shape)

    @overload
    def __getitem__(self, index: SupportsIndex) -> IntegerT_co: ...
    @overload
    def __getitem__(self, index: Slice) -> IntegerMatrix[Literal[1], Any, IntegerT_co]: ...
    @overload
    def __getitem__(self, index: tuple[SupportsIndex, SupportsIndex]) -> IntegerT_co: ...
    @overload
    def __getitem__(self, index: tuple[SupportsIndex, Slice]) -> IntegerMatrix[Literal[1], Any, IntegerT_co]: ...
    @overload
    def __getitem__(self, index: tuple[Slice, SupportsIndex]) -> IntegerMatrix[Any, Literal[1], IntegerT_co]: ...
    @overload
    def __getitem__(self, index: tuple[Slice, Slice]) -> IntegerMatrix[Any, Any, IntegerT_co]: ...
    @override
    def __getitem__(
        self,
        index: SupportsIndex | Slice | tuple[SupportsIndex | Slice, SupportsIndex | Slice],
    ) -> IntegerT_co | IntegerMatrix[Any, Any, IntegerT_co]:
        result = super().__getitem__(index)
        if isinstance(result, Matrix):
            return IntegerMatrix[Any, Any, IntegerT_co].from_matrix(result)
        return result

    @overload
    def __add__(self, other: Matrix[M_co, N_co, Integer]) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __add__(self, other: Integer) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __add__(self, other: Matrix[M_co, N_co, Real]) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __add__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __add__(self, other: Matrix[M_co, N_co, Complex]) -> ComplexMatrix[M_co, N_co]: ...
    @overload
    def __add__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...
    @override
    def __add__(self, other: Matrix[M_co, N_co, Complex] | Complex) -> ComplexMatrix[M_co, N_co]:
        if is_integer_object(other):
            return IntegerMatrix[M_co, N_co].from_matrix(super().__add__(other))
        return super().__add__(other)

    @overload
    def __radd__(self, other: Integer) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __radd__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __radd__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...
    @override
    def __radd__(self, other: Complex) -> ComplexMatrix[M_co, N_co]:
        if is_integer_number(other):
            return IntegerMatrix[M_co, N_co].from_matrix(super().__radd__(other))
        return super().__radd__(other)

    @overload
    def __sub__(self, other: Matrix[M_co, N_co, Integer]) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __sub__(self, other: Integer) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __sub__(self, other: Matrix[M_co, N_co, Real]) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __sub__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __sub__(self, other: Matrix[M_co, N_co, Complex]) -> ComplexMatrix[M_co, N_co]: ...
    @overload
    def __sub__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...
    @override
    def __sub__(self, other: Matrix[M_co, N_co, Complex] | Complex) -> ComplexMatrix[M_co, N_co]:
        if is_integer_object(other):
            return IntegerMatrix[M_co, N_co].from_matrix(super().__sub__(other))
        return super().__sub__(other)

    @overload
    def __rsub__(self, other: Integer) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __rsub__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __rsub__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...
    @override
    def __rsub__(self, other: Complex) -> ComplexMatrix[M_co, N_co]:
        if is_integer_number(other):
            return IntegerMatrix[M_co, N_co].from_matrix(super().__rsub__(other))
        return super().__rsub__(other)

    @overload
    def __mul__(self, other: Matrix[M_co, N_co, Integer]) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __mul__(self, other: Integer) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __mul__(self, other: Matrix[M_co, N_co, Real]) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __mul__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __mul__(self, other: Matrix[M_co, N_co, Complex]) -> ComplexMatrix[M_co, N_co]: ...
    @overload
    def __mul__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...
    @override
    def __mul__(self, other: Matrix[M_co, N_co, Complex] | Complex) -> ComplexMatrix[M_co, N_co]:
        if is_integer_object(other):
            return IntegerMatrix[M_co, N_co].from_matrix(super().__mul__(other))
        return super().__mul__(other)

    @overload
    def __rmul__(self, other: Integer) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __rmul__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __rmul__(self, other: Complex) -> ComplexMatrix[M_co, N_co]: ...
    @override
    def __rmul__(self, other: Complex) -> ComplexMatrix[M_co, N_co]:
        if is_integer_number(other):
            return IntegerMatrix[M_co, N_co].from_matrix(super().__rmul__(other))
        return super().__rmul__(other)

    # NOTE: No override for __truediv__()/__rtruediv__() - division (often)
    # leaves the realm of integer numbers.

    @overload
    def __matmul__[P: int](self, other: Matrix[N_co, P, Integer]) -> IntegerMatrix[M_co, P]: ...
    @overload
    def __matmul__[P: int](self, other: Matrix[N_co, P, Real]) -> RealMatrix[M_co, P]: ...
    @override
    def __matmul__[P: int](self, other: Matrix[N_co, P, Real]) -> RealMatrix[M_co, P]:
        if is_integer_matrix(other):
            return IntegerMatrix[M_co, P].from_matrix(super().__matmul__(other))
        return super().__matmul__(other)

    @overload
    def __floordiv__(self, other: Matrix[M_co, N_co, Integer]) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __floordiv__(self, other: Integer) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __floordiv__(self, other: Matrix[M_co, N_co, Real]) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __floordiv__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @override
    def __floordiv__(self, other: Matrix[M_co, N_co, Real] | Real) -> RealMatrix[M_co, N_co]:
        if is_integer_object(other):
            return IntegerMatrix[M_co, N_co].from_matrix(super().__floordiv__(other))
        return super().__floordiv__(other)

    @overload
    def __rfloordiv__(self, other: Integer) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __rfloordiv__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @override
    def __rfloordiv__(self, other: Real) -> RealMatrix[M_co, N_co]:
        if is_integer_number(other):
            return IntegerMatrix[M_co, N_co].from_matrix(super().__rfloordiv__(other))
        return super().__rfloordiv__(other)

    @overload
    def __mod__(self, other: Matrix[M_co, N_co, Integer]) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __mod__(self, other: Integer) -> RealMatrix[M_co, N_co, Integer]: ...
    @overload
    def __mod__(self, other: Matrix[M_co, N_co, Real]) -> RealMatrix[M_co, N_co]: ...
    @overload
    def __mod__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @override
    def __mod__(self, other: Matrix[M_co, N_co, Real] | Real) -> RealMatrix[M_co, N_co]:
        if is_integer_object(other):
            return IntegerMatrix[M_co, N_co].from_matrix(super().__mod__(other))
        return super().__mod__(other)

    @overload
    def __rmod__(self, other: Integer) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __rmod__(self, other: Real) -> RealMatrix[M_co, N_co]: ...
    @override
    def __rmod__(self, other: Real) -> RealMatrix[M_co, N_co]:
        if is_integer_number(other):
            return IntegerMatrix[M_co, N_co].from_matrix(super().__rmod__(other))
        return super().__rmod__(other)

    @overload
    def __divmod__(self, other: Matrix[M_co, N_co, Integer]) -> tuple[IntegerMatrix[M_co, N_co], IntegerMatrix[M_co, N_co]]: ...
    @overload
    def __divmod__(self, other: Integer) -> tuple[IntegerMatrix[M_co, N_co], IntegerMatrix[M_co, N_co]]: ...
    @overload
    def __divmod__(self, other: Matrix[M_co, N_co, Real]) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...
    @overload
    def __divmod__(self, other: Real) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...
    @override
    def __divmod__(self, other: Matrix[M_co, N_co, Real] | Real) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]:
        if is_integer_object(other):
            a, b = super().__divmod__(other)
            return (
                IntegerMatrix[M_co, N_co].from_matrix(a),
                IntegerMatrix[M_co, N_co].from_matrix(b),
            )
        return super().__divmod__(other)

    @overload
    def __rdivmod__(self, other: Integer) -> tuple[IntegerMatrix[M_co, N_co], IntegerMatrix[M_co, N_co]]: ...
    @overload
    def __rdivmod__(self, other: Real) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]: ...
    @override
    def __rdivmod__(self, other: Real) -> tuple[RealMatrix[M_co, N_co], RealMatrix[M_co, N_co]]:
        if is_integer_number(other):
            a, b = super().__rdivmod__(other)
            return (
                IntegerMatrix[M_co, N_co].from_matrix(a),
                IntegerMatrix[M_co, N_co].from_matrix(b),
            )
        return super().__rdivmod__(other)

    @overload
    def __lshift__(self, other: Matrix[M_co, N_co, Integer]) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __lshift__(self, other: Integer) -> IntegerMatrix[M_co, N_co]: ...

    def __lshift__(self, other: Matrix[M_co, N_co, Integer] | Integer) -> IntegerMatrix[M_co, N_co]:
        if is_integer_matrix(other):
            return IntegerMatrix(
                array=self._binary_matrix_map(
                    operator.__lshift__,
                    other,
                ),
                shape=self.shape,
            )
        if is_integer_number(other):
            return IntegerMatrix(
                array=self._binary_scalar_map(
                    operator.__lshift__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    def __rlshift__(self, other: Integer) -> IntegerMatrix[M_co, N_co]:
        if is_integer_number(other):
            return IntegerMatrix(
                array=self._binary_scalar_map_r(
                    operator.__lshift__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __rshift__(self, other: Matrix[M_co, N_co, Integer]) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __rshift__(self, other: Integer) -> IntegerMatrix[M_co, N_co]: ...

    def __rshift__(self, other: Matrix[M_co, N_co, Integer] | Integer) -> IntegerMatrix[M_co, N_co]:
        if is_integer_matrix(other):
            return IntegerMatrix(
                array=self._binary_matrix_map(
                    operator.__rshift__,
                    other,
                ),
                shape=self.shape,
            )
        if is_integer_number(other):
            return IntegerMatrix(
                array=self._binary_scalar_map(
                    operator.__rshift__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    def __rrshift__(self, other: Integer) -> IntegerMatrix[M_co, N_co]:
        if is_integer_number(other):
            return IntegerMatrix(
                array=self._binary_scalar_map_r(
                    operator.__rshift__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __and__(self, other: Matrix[M_co, N_co, Integer]) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __and__(self, other: Integer) -> IntegerMatrix[M_co, N_co]: ...

    def __and__(self, other: Matrix[M_co, N_co, Integer] | Integer) -> IntegerMatrix[M_co, N_co]:
        if is_integer_matrix(other):
            return IntegerMatrix(
                array=self._binary_matrix_map(
                    operator.__and__,
                    other,
                ),
                shape=self.shape,
            )
        if is_integer_number(other):
            return IntegerMatrix(
                array=self._binary_scalar_map(
                    operator.__and__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    def __rand__(self, other: Integer) -> IntegerMatrix[M_co, N_co]:
        if is_integer_number(other):
            return IntegerMatrix(
                array=self._binary_scalar_map_r(
                    operator.__and__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __xor__(self, other: Matrix[M_co, N_co, Integer]) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __xor__(self, other: Integer) -> IntegerMatrix[M_co, N_co]: ...

    def __xor__(self, other: Matrix[M_co, N_co, Integer] | Integer) -> IntegerMatrix[M_co, N_co]:
        if is_integer_matrix(other):
            return IntegerMatrix(
                array=self._binary_matrix_map(
                    operator.__xor__,
                    other,
                ),
                shape=self.shape,
            )
        if is_integer_number(other):
            return IntegerMatrix(
                array=self._binary_scalar_map(
                    operator.__xor__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    def __rxor__(self, other: Integer) -> IntegerMatrix[M_co, N_co]:
        if is_integer_number(other):
            return IntegerMatrix(
                array=self._binary_scalar_map_r(
                    operator.__xor__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @overload
    def __or__(self, other: Matrix[M_co, N_co, Integer]) -> IntegerMatrix[M_co, N_co]: ...
    @overload
    def __or__(self, other: Integer) -> IntegerMatrix[M_co, N_co]: ...

    def __or__(self, other: Matrix[M_co, N_co, Integer] | Integer) -> IntegerMatrix[M_co, N_co]:
        if is_integer_matrix(other):
            return IntegerMatrix(
                array=self._binary_matrix_map(
                    operator.__or__,
                    other,
                ),
                shape=self.shape,
            )
        if is_integer_number(other):
            return IntegerMatrix(
                array=self._binary_scalar_map(
                    operator.__or__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    def __ror__(self, other: Integer) -> IntegerMatrix[M_co, N_co]:
        if is_integer_number(other):
            return IntegerMatrix(
                array=self._binary_scalar_map_r(
                    operator.__or__,
                    other,
                ),
                shape=self.shape,
            )
        return NotImplemented

    @override
    def __neg__(self) -> IntegerMatrix[M_co, N_co]:
        return IntegerMatrix[M_co, N_co].from_matrix(super().__neg__())

    @override
    def __pos__(self) -> IntegerMatrix[M_co, N_co]:
        return IntegerMatrix[M_co, N_co].from_matrix(super().__pos__())

    @override
    def __abs__(self) -> IntegerMatrix[M_co, N_co]:
        return IntegerMatrix[M_co, N_co].from_matrix(super().__abs__())

    def __invert__(self) -> IntegerMatrix[M_co, N_co]:
        return IntegerMatrix(
            array=self._unary_map(operator.__invert__),
            shape=self.shape,
        )

    def __int__(self: IntegerMatrix[Literal[1], Literal[1], IntegerT_co]) -> int:
        return int(self.demote())

    @property
    @override
    def real(self) -> IntegerMatrix[M_co, N_co]:
        return self

    @override
    def transpose(self) -> IntegerMatrix[N_co, M_co, IntegerT_co]:
        return IntegerMatrix[N_co, M_co, IntegerT_co].from_matrix(super().transpose())

    @override
    def flip(self, *, by: Rule = Rule.ROW) -> IntegerMatrix[M_co, N_co, IntegerT_co]:
        return IntegerMatrix[M_co, N_co, IntegerT_co].from_matrix(super().flip(by=by))

    @overload
    def rotate(self, n: EvenNumber) -> IntegerMatrix[M_co, N_co, IntegerT_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> IntegerMatrix[N_co, M_co, IntegerT_co]: ...
    @overload
    def rotate(self, n: SupportsIndex) -> IntegerMatrix[Any, Any, IntegerT_co]: ...
    @overload
    def rotate(self) -> IntegerMatrix[N_co, M_co, IntegerT_co]: ...
    @override
    def rotate(self, n: SupportsIndex = 1) -> IntegerMatrix[Any, Any, IntegerT_co]:
        return IntegerMatrix[Any, Any, IntegerT_co].from_matrix(super().rotate(n=n))

    @override
    def reverse(self) -> IntegerMatrix[M_co, N_co, IntegerT_co]:
        return IntegerMatrix[M_co, N_co, IntegerT_co].from_matrix(super().reverse())

    @override
    def rows(self, *, reverse: bool = False) -> Iterator[IntegerMatrix[Literal[1], N_co, IntegerT_co]]:
        return map(IntegerMatrix[Literal[1], N_co, IntegerT_co].from_matrix, super().rows(reverse=reverse))

    @override
    def cols(self, *, reverse: bool = False) -> Iterator[IntegerMatrix[M_co, Literal[1], IntegerT_co]]:
        return map(IntegerMatrix[M_co, Literal[1], IntegerT_co].from_matrix, super().cols(reverse=reverse))

    @overload
    def vectors(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[IntegerMatrix[Literal[1], N_co, IntegerT_co]]: ...
    @overload
    def vectors(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[IntegerMatrix[M_co, Literal[1], IntegerT_co]]: ...
    @overload
    def vectors(self, *, by: Rule, reverse: bool = False) -> Iterator[IntegerMatrix[Any, Any, IntegerT_co]]: ...
    @overload
    def vectors(self, *, reverse: bool = False) -> Iterator[IntegerMatrix[Literal[1], N_co, IntegerT_co]]: ...
    @override
    def vectors(self, *, by: Rule = Rule.ROW, reverse: bool = False) -> Iterator[IntegerMatrix[Any, Any, IntegerT_co]]:
        return map(IntegerMatrix[Any, Any, IntegerT_co].from_matrix, super().vectors(by=by, reverse=reverse))

    @override
    def conjugate(self) -> IntegerMatrix[M_co, N_co]:
        return self

    @override
    def transjugate(self) -> IntegerMatrix[N_co, M_co]:
        return IntegerMatrix[N_co, M_co].from_matrix(super().transjugate())

    @override
    def sort(
        self,
        *,
        key: Callable[[IntegerT_co], Sortable] | None = None,
        reverse: bool = False,
    ) -> IntegerMatrix[M_co, N_co, IntegerT_co]:
        return IntegerMatrix[M_co, N_co, IntegerT_co].from_matrix(super().sort(key=key, reverse=reverse))


def iter_or[T](reversible: Reversible[T], *, reverse: bool = False) -> Iterator[T]:
    """Return the iterator of an object, optionally its reverse iterator."""
    return reversed(reversible) if reverse else iter(reversible)


def interleave[T](iterables: tuple[Iterable[T], ...], leave_counts: tuple[int, ...]) -> Iterator[T]:
    """Return an iterator that, for each integer N in ``leave_counts``, yields
    N elements from the parallel iterable of ``iterables`` repeatedly until all
    have been exhausted.
    """
    assert len(iterables) == len(leave_counts)

    sentinel = object()

    iterators = tuple(map(iter, iterables))
    index_queue = deque(range(len(iterators)))

    while index_queue:
        index = index_queue.popleft()
        iterator = iterators[index]

        exhausted = False

        leave_count = leave_counts[index]
        if leave_count:
            for _ in range(leave_count):
                value = next(iterator, sentinel)
                if value is sentinel:
                    exhausted = True
                    break
                else:
                    yield value  # type: ignore
        else:
            exhausted = True

        if not exhausted:
            index_queue.append(index)


def vec2[RealT: Real = Real](x: RealT, y: RealT) -> RealMatrix[Literal[2], Literal[1], RealT]:
    """Convenience function for constructing a ``RealMatrix`` of shape
    ``(2, 1)``, commonly used in 2D space models.
    """
    return RealMatrix[Literal[2], Literal[1], RealT].col((x, y))


def vec3[RealT: Real = Real](x: RealT, y: RealT, z: RealT) -> RealMatrix[Literal[3], Literal[1], RealT]:
    """Convenience function for constructing a ``RealMatrix`` of shape
    ``(3, 1)``, commonly used in 3D space models.
    """
    return RealMatrix[Literal[3], Literal[1], RealT].col((x, y, z))


def is_complex_number(obj: object) -> TypeGuard[Complex]:
    """Return true if ``obj`` is an instance of ``Complex``, otherwise false."""
    return isinstance(obj, (complex, float, int))


def is_complex_matrix(obj: object) -> TypeGuard[Matrix[Any, Any, Complex]]:
    """Return true if ``obj`` is an instance of ``Matrix``, and contains only
    instances of ``Complex``.
    """
    if isinstance(obj, Matrix):
        if isinstance(obj, ComplexMatrix):
            return True
        return all(map(is_complex_number, obj))
    return False


def is_complex_object(obj: object) -> TypeGuard[Matrix[Any, Any, Complex] | Complex]:
    """Return true if ``obj`` is an instance of either a ``Matrix`` of
    ``Complex`` values, or a single ``Complex`` value.
    """
    return is_complex_number(obj) or is_complex_matrix(obj)


def is_real_number(obj: object) -> TypeGuard[Real]:
    """Return true if ``obj`` is an instance of ``Real``, otherwise false."""
    return isinstance(obj, (float, int))


def is_real_matrix(obj: object) -> TypeGuard[Matrix[Any, Any, Real]]:
    """Return true if ``obj`` is an instance of ``Matrix``, and contains only
    instances of ``Real``.
    """
    if isinstance(obj, Matrix):
        if isinstance(obj, RealMatrix):
            return True
        return all(map(is_real_number, obj))
    return False


def is_real_object(obj: object) -> TypeGuard[Matrix[Any, Any, Real] | Real]:
    """Return true if ``obj`` is an instance of either a ``Matrix`` of
    ``Real`` values, or a single ``Real`` value.
    """
    return is_real_number(obj) or is_real_matrix(obj)


def is_integer_number(obj: object) -> TypeGuard[Integer]:
    """Return true if ``obj`` is an instance of ``Integer``, otherwise false."""
    return isinstance(obj, int)


def is_integer_matrix(obj: object) -> TypeGuard[Matrix[Any, Any, Integer]]:
    """Return true if ``obj`` is an instance of ``Matrix``, and contains only
    instances of ``Integer``.
    """
    if isinstance(obj, Matrix):
        if isinstance(obj, IntegerMatrix):
            return True
        return all(map(is_integer_number, obj))
    return False


def is_integer_object(obj: object) -> TypeGuard[Matrix[Any, Any, Integer] | Integer]:
    """Return true if ``obj`` is an instance of either a ``Matrix`` of
    ``Integer`` values, or a single ``Integer`` value.
    """
    return is_integer_number(obj) or is_integer_matrix(obj)
