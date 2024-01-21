from __future__ import annotations

import itertools
import operator
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import (Any, Generic, Literal, Optional, SupportsIndex, TypeVar,
                    Union, overload)

from typing_extensions import Self, TypeAlias, override

from . import accessors
from .accessors import COL, ROW, Rule

EvenNumber: TypeAlias = Literal[-16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16]
OddNumber: TypeAlias = Literal[-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]

T_co = TypeVar("T_co", covariant=True)
S_co = TypeVar("S_co", covariant=True)
R_co = TypeVar("R_co", covariant=True)

T_contra = TypeVar("T_contra", contravariant=True)
S_contra = TypeVar("S_contra", contravariant=True)
R_contra = TypeVar("R_contra", contravariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class Matrix(Sequence[T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("_accessor")
    _accessor: accessors.AbstractAccessor[T_co]

    @overload
    def __init__(self: Matrix[Literal[1], Literal[0], T_co]) -> None: ...
    @overload
    def __init__(self, array: accessors.AbstractAccessor[T_co]) -> None: ...
    @overload
    def __init__(self, array: Matrix[M_co, N_co, T_co]) -> None: ...
    @overload
    def __init__(self: Matrix[Literal[1], Literal[0], T_co], array: tuple[()]) -> None: ...
    @overload
    def __init__(self: Matrix[Literal[1], Literal[1], T_co], array: tuple[T_co]) -> None: ...
    @overload
    def __init__(self: Matrix[Literal[1], N_co, T_co], array: Iterable[T_co]) -> None: ...
    @overload
    def __init__(self: Matrix[Literal[1], Literal[0], T_co], *, shape: Literal[Rule.ROW]) -> None: ...
    @overload
    def __init__(self: Matrix[Literal[0], Literal[1], T_co], *, shape: Literal[Rule.COL]) -> None: ...
    @overload
    def __init__(self, *, shape: Rule) -> None: ...
    @overload
    def __init__(self: Matrix[M_co, N_co, T_co], *, shape: tuple[M_co, N_co]) -> None: ...
    @overload
    def __init__(self: Matrix[Literal[1], N_co, T_co], array: Iterable[T_co], shape: Literal[Rule.ROW]) -> None: ...
    @overload
    def __init__(self: Matrix[M_co, Literal[1], T_co], array: Iterable[T_co], shape: Literal[Rule.COL]) -> None: ...
    @overload
    def __init__(self, array: Iterable[T_co], shape: Rule) -> None: ...
    @overload
    def __init__(self, array: Iterable[T_co], shape: tuple[M_co, N_co]) -> None: ...

    def __init__(
        self,
        array: Union[accessors.AbstractAccessor[T_co], Matrix[M_co, N_co, T_co], Iterable[T_co]] = (),
        shape: Union[Rule, tuple[M_co, N_co], None] = None,
    ) -> None:
        if shape is None:
            if isinstance(array, accessors.AbstractAccessor):
                self._accessor = array
            elif isinstance(array, Matrix):
                self._accessor = array._accessor
            else:
                array = tuple(array)
                count = len(array)
                if count > 1:
                    self._accessor = accessors.RowVectorAccessor(array)
                elif count:
                    self._accessor = accessors.ValueAccessor(array[0])
                else:
                    self._accessor = accessors.NULLARY_ACCESSOR_1x0
            return

        if isinstance(array, accessors.AbstractAccessor):
            array = array.materialize()
        elif isinstance(array, Matrix):
            array = array._accessor.materialize()
        else:
            array = tuple(array)

        if isinstance(shape, Rule):
            count = len(array)
            if shape is ROW:
                if count > 1:
                    self._accessor = accessors.RowVectorAccessor(array)
                elif count:
                    self._accessor = accessors.ValueAccessor(array[0])
                else:
                    self._accessor = accessors.NULLARY_ACCESSOR_1x0
            else:
                if count > 1:
                    self._accessor = accessors.ColVectorAccessor(array)
                elif count:
                    self._accessor = accessors.ValueAccessor(array[0])
                else:
                    self._accessor = accessors.NULLARY_ACCESSOR_0x1
            return

        row_count = shape[0]
        col_count = shape[1]

        if __debug__:
            test_size = row_count * col_count
            true_size = len(array)
            if row_count < 0 or col_count < 0:
                raise ValueError("dimensions must be non-negative")
            if test_size != true_size:
                raise ValueError(f"cannot interpret size {true_size} iterable as shape ({row_count}, {col_count})")

        if col_count > 1:
            if row_count > 1:
                self._accessor = accessors.MatrixAccessor(array, shape)
            elif row_count:
                self._accessor = accessors.RowVectorAccessor(array)
            else:
                self._accessor = accessors.ColCountAccessor(col_count)
        elif col_count:
            if row_count > 1:
                self._accessor = accessors.ColVectorAccessor(array)
            elif row_count:
                self._accessor = accessors.ValueAccessor(array[0])
            else:
                self._accessor = accessors.NULLARY_ACCESSOR_0x1
        else:
            if row_count > 1:
                self._accessor = accessors.RowCountAccessor(row_count)
            elif row_count:
                self._accessor = accessors.NULLARY_ACCESSOR_1x0
            else:
                self._accessor = accessors.NULLARY_ACCESSOR_0x0

    def __repr__(self) -> str:
        return f"Matrix([{', '.join(map(repr, self))}], shape={self.shape!r})"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, Matrix):
            return self._accessor == other._accessor  # pyright: ignore[reportUnknownMemberType]
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._accessor)

    @override
    def __len__(self) -> int:
        return len(self._accessor)

    @overload
    def __getitem__(self, index: SupportsIndex) -> T_co: ...
    @overload
    def __getitem__(self, index: slice) -> Matrix[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, index: tuple[SupportsIndex, SupportsIndex]) -> T_co: ...
    @overload
    def __getitem__(self, index: tuple[SupportsIndex, slice]) -> Matrix[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, index: tuple[slice, SupportsIndex]) -> Matrix[Any, Literal[1], T_co]: ...
    @overload
    def __getitem__(self, index: tuple[slice, slice]) -> Matrix[Any, Any, T_co]: ...
    @override
    def __getitem__(
        self,
        index: Union[
            SupportsIndex,
            slice,
            tuple[Union[SupportsIndex, slice], Union[SupportsIndex, slice]],
        ],
    ) -> Union[T_co, Matrix[Any, Any, T_co]]:
        accessor = self._accessor

        if isinstance(index, tuple):
            row_index, col_index = index

            if isinstance(row_index, slice):
                row_window = accessor.resolve_matrix_slice(row_index, by=ROW)

                if isinstance(col_index, slice):
                    col_window = accessor.resolve_matrix_slice(col_index, by=COL)

                    return Matrix(
                        accessors.MatrixSliceAccessor(
                            accessor,
                            row_window=row_window,
                            col_window=col_window,
                        ),
                    )
                else:
                    col_index = accessor.resolve_matrix_index(col_index, by=COL)

                    return Matrix(
                        accessors.ColSliceAccessor(
                            accessor,
                            row_window=row_window,
                            col_index=col_index,
                        ),
                    )

            else:
                row_index = accessor.resolve_matrix_index(row_index, by=ROW)

                if isinstance(col_index, slice):
                    col_window = accessor.resolve_matrix_slice(col_index, by=COL)

                    return Matrix(
                        accessors.RowSliceAccessor(
                            accessor,
                            row_index=row_index,
                            col_window=col_window,
                        ),
                    )
                else:
                    col_index = accessor.resolve_matrix_index(col_index, by=COL)

                    return accessor.matrix_access(
                        row_index=row_index,
                        col_index=col_index,
                    )

        elif isinstance(index, slice):
            window = accessor.resolve_vector_slice(index)

            return Matrix(
                accessors.SliceAccessor(
                    accessor,
                    window=window,
                ),
            )

        else:
            index = accessor.resolve_vector_index(index)

            return accessor.vector_access(index=index)

    @override
    def __iter__(self) -> Iterator[T_co]:
        return iter(self._accessor)

    @override
    def __reversed__(self) -> Iterator[T_co]:
        return reversed(self._accessor)

    @override
    def __contains__(self, value: object) -> bool:
        return value in self._accessor

    def __deepcopy__(self, memo: Optional[dict[int, Any]] = None) -> Self:
        return self

    def __copy__(self) -> Self:
        return self

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self._accessor.shape  # type: ignore

    @property
    def row_count(self) -> M_co:
        return self._accessor.row_count  # type: ignore

    @property
    def col_count(self) -> N_co:
        return self._accessor.col_count  # type: ignore

    def materialize(self) -> Matrix[M_co, N_co, T_co]:
        return Matrix(self._accessor.materialize(), self.shape)

    def transpose(self) -> Matrix[N_co, M_co, T_co]:
        return Matrix(accessors.TransposeAccessor(self._accessor))

    def flip(self, *, by: Rule = Rule.ROW) -> Matrix[M_co, N_co, T_co]:
        target = self._accessor
        if by is ROW:
            return Matrix(accessors.RowFlipAccessor(target))
        else:
            return Matrix(accessors.ColFlipAccessor(target))

    @overload
    def rotate(self, n: EvenNumber) -> Matrix[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> Matrix[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: SupportsIndex) -> Matrix[Any, Any, T_co]: ...
    @overload
    def rotate(self) -> Matrix[N_co, M_co, T_co]: ...

    def rotate(self, n: SupportsIndex = 1) -> Matrix[Any, Any, T_co]:
        n = operator.index(n) % 4
        if not n:
            return self
        target = self._accessor
        if n == 1:
            return Matrix(accessors.Rotate090Accessor(target))
        elif n == 2:
            return Matrix(accessors.Rotate180Accessor(target))
        else:
            return Matrix(accessors.Rotate270Accessor(target))

    def reverse(self) -> Matrix[M_co, N_co, T_co]:
        return self.rotate(2)

    def values(self, *, by: Rule = Rule.ROW) -> Iterator[T_co]:
        target = self._accessor
        if by is ROW:
            return iter(target)
        else:
            return iter(accessors.TransposeAccessor(target))

    @overload
    def slices(self, *, by: Literal[Rule.ROW]) -> Iterator[Matrix[Literal[1], N_co, T_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL]) -> Iterator[Matrix[M_co, Literal[1], T_co]]: ...
    @overload
    def slices(self, *, by: Rule) -> Iterator[Matrix[Any, Any, T_co]]: ...
    @overload
    def slices(self) -> Iterator[Matrix[Literal[1], N_co, T_co]]: ...

    def slices(self, *, by: Rule = Rule.ROW) -> Iterator[Matrix[Any, Any, T_co]]:
        target = self._accessor
        if by is ROW:
            for row_index in range(self.row_count):
                yield Matrix(accessors.RowSheerAccessor(target, row_index=row_index))
        else:
            for col_index in range(self.col_count):
                yield Matrix(accessors.ColSheerAccessor(target, col_index=col_index))

    @overload
    def stack(self, other: Matrix[Any, N_co, S_co], *, by: Literal[Rule.ROW]) -> Matrix[Any, N_co, Union[T_co, S_co]]: ...
    @overload
    def stack(self, other: Matrix[M_co, Any, S_co], *, by: Literal[Rule.COL]) -> Matrix[M_co, Any, Union[T_co, S_co]]: ...
    @overload
    def stack(self, other: Matrix[Any, Any, S_co], *, by: Rule) -> Matrix[Any, Any, Union[T_co, S_co]]: ...
    @overload
    def stack(self, other: Matrix[Any, N_co, S_co]) -> Matrix[Any, N_co, Union[T_co, S_co]]: ...

    def stack(self, other: Matrix[Any, Any, S_co], *, by: Rule = Rule.ROW) -> Matrix[Any, Any, Union[T_co, S_co]]:
        target_head = self._accessor
        target_tail = other._accessor
        if __debug__:
            dy = ~by
            if target_head.shape[dy] != target_tail.shape[dy]:
                raise ValueError(f"cannot {by.handle}-stack matrices with differing number of {dy.handle}s")
        if by is ROW:
            return Matrix(accessors.RowStackAccessor(target_head, target_tail))
        else:
            return Matrix(accessors.ColStackAccessor(target_head, target_tail))

    def _unary_map(
        self,
        mapper: Callable[[T_co], R_co],
    ) -> Matrix[M_co, N_co, R_co]:
        return Matrix(
            array=map(mapper, self),
            shape=self.shape,
        )

    def _binary_matrix_map(
        self,
        mapper: Callable[[T_co, S_co], R_co],
        other: Matrix[M_co, N_co, S_co],
    ) -> Matrix[M_co, N_co, R_co]:
        if __debug__:
            if self.shape != other.shape:
                raise ValueError
        return Matrix(
            array=map(mapper, self, other),
            shape=self.shape,
        )

    def _binary_scalar_map(
        self,
        mapper: Callable[[T_co, S_contra], R_co],
        other: S_contra,
    ) -> Matrix[M_co, N_co, R_co]:
        return Matrix(
            array=map(mapper, self, itertools.repeat(other)),
            shape=self.shape,
        )

    def equal(self, other: Union[Matrix[M_co, N_co, object], object]) -> Matrix[M_co, N_co, bool]:
        if isinstance(other, Matrix):
            return self._binary_matrix_map(
                operator.__eq__,
                other,  # type: ignore[arg-type]
            )
        else:
            return self._binary_scalar_map(
                operator.__eq__,
                other,
            )

    def not_equal(self, other: Union[Matrix[M_co, N_co, object], object]) -> Matrix[M_co, N_co, bool]:
        if isinstance(other, Matrix):
            return self._binary_matrix_map(
                operator.__ne__,
                other,  # type: ignore[arg-type]
            )
        else:
            return self._binary_scalar_map(
                operator.__ne__,
                other,
            )
