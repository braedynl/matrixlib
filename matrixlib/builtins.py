from __future__ import annotations

__all__ = ["Matrix"]

import operator
from collections.abc import Iterable, Iterator, Sequence
from typing import (Any, Generic, Literal, Self, SupportsIndex, TypeAlias,
                    TypeVar, overload)

from typing_extensions import override

from .accessors import (AbstractAccessor, ColFlipAccessor, ColSheerAccessor,
                        ColSliceAccessor, ColStackAccessor, ColVectorAccessor,
                        MatrixAccessor, MatrixSliceAccessor,
                        NULLARY_ACCESSOR_0x0, NULLARY_ACCESSOR_0x1,
                        NULLARY_ACCESSOR_1x0, Rotate090Accessor,
                        Rotate180Accessor, Rotate270Accessor, RowFlipAccessor,
                        RowSheerAccessor, RowSliceAccessor, RowStackAccessor,
                        RowVectorAccessor, SliceAccessor, TransposeAccessor,
                        ValueAccessor, ZeroColAccessor, ZeroRowAccessor)
from .rule import COL, ROW, Rule

EvenNumber: TypeAlias = Literal[-16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16]
OddNumber: TypeAlias = Literal[-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)
P_co = TypeVar("P_co", covariant=True, bound=int)
Q_co = TypeVar("Q_co", covariant=True, bound=int)

T_co = TypeVar("T_co", covariant=True)
S_co = TypeVar("S_co", covariant=True)

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
P = TypeVar("P", bound=int)
Q = TypeVar("Q", bound=int)

T = TypeVar("T")
S = TypeVar("S")


class Matrix(Sequence[T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("_accessor",)
    __match_args__ = ("array", "shape")
    _accessor: AbstractAccessor[M_co, N_co, T_co]

    @overload
    def __init__(self: Matrix[Literal[1], Literal[0], Any]) -> None: ...
    @overload
    def __init__(self: Matrix[Literal[1], Any, T], array: Iterable[T]) -> None: ...
    @overload
    def __init__(self: Matrix[M, N, Any], *, shape: tuple[M, N]) -> None: ...
    @overload
    def __init__(self: Matrix[Literal[1], Any, Any], *, shape: Literal[Rule.ROW]) -> None: ...
    @overload
    def __init__(self: Matrix[Any, Literal[1], Any], *, shape: Literal[Rule.COL]) -> None: ...
    @overload
    def __init__(self: Matrix[Any, Any, Any], *, shape: Rule) -> None: ...
    @overload
    def __init__(self, array: Iterable[T_co], shape: tuple[M_co, N_co]) -> None: ...
    @overload
    def __init__(self: Matrix[Literal[1], Any, T], array: Iterable[T], shape: Literal[Rule.ROW]) -> None: ...
    @overload
    def __init__(self: Matrix[Any, Literal[1], T], array: Iterable[T], shape: Literal[Rule.COL]) -> None: ...
    @overload
    def __init__(self: Matrix[Any, Any, T], array: Iterable[T], shape: Rule) -> None: ...

    def __init__(
        self,
        array: Iterable[T_co] = (),
        shape: tuple[M_co, N_co] | Rule = Rule.ROW,
    ) -> None:
        array = tuple(array)
        if isinstance(shape, tuple):
            row_count = shape[0]
            col_count = shape[1]
            if __debug__:
                if row_count < 0 or col_count < 0:
                    raise ValueError
                true_size = len(array)
                test_size = row_count * col_count
                if true_size != test_size:
                    raise ValueError
            if row_count > 1:
                if col_count > 1:
                    self._accessor = MatrixAccessor(array, shape)
                elif col_count:
                    self._accessor = ColVectorAccessor(array)  # pyright: ignore
                else:
                    self._accessor = ZeroColAccessor(row_count)  # pyright: ignore
            elif row_count:
                if col_count > 1:
                    self._accessor = RowVectorAccessor(array)  # pyright: ignore
                elif col_count:
                    self._accessor = ValueAccessor(array[0])  # pyright: ignore
                else:
                    self._accessor = NULLARY_ACCESSOR_1x0  # pyright: ignore
            else:
                if col_count > 1:
                    self._accessor = ZeroRowAccessor(col_count)  # pyright: ignore
                elif col_count:
                    self._accessor = NULLARY_ACCESSOR_0x1  # pyright: ignore
                else:
                    self._accessor = NULLARY_ACCESSOR_0x0  # pyright: ignore
        else:
            size = len(array)
            if shape is ROW:
                if size > 1:
                    self._accessor = RowVectorAccessor(array)  # pyright: ignore
                elif size:
                    self._accessor = ValueAccessor(array[0])  # pyright: ignore
                else:
                    self._accessor = NULLARY_ACCESSOR_1x0  # pyright: ignore
            else:
                if size > 1:
                    self._accessor = ColVectorAccessor(array)  # pyright: ignore
                elif size:
                    self._accessor = ValueAccessor(array[0])  # pyright: ignore
                else:
                    self._accessor = NULLARY_ACCESSOR_0x1  # pyright: ignore

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} shape={self.shape!r}>"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, Matrix):
            return self._accessor == other._accessor  # pyright: ignore[reportUnknownMemberType]
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._accessor)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        return self

    __copy__ = __deepcopy__

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
        index: SupportsIndex | slice | tuple[SupportsIndex | slice, SupportsIndex | slice],
    ) -> T_co | Matrix[Any, Any, T_co]:
        accessor = self._accessor

        if isinstance(index, tuple):
            row_index, col_index = index

            if isinstance(row_index, slice):
                row_window = accessor.resolve_matrix_slice(row_index, by=ROW)

                if isinstance(col_index, slice):
                    col_window = accessor.resolve_matrix_slice(col_index, by=COL)

                    return Matrix[Any, Any, T_co].from_accessor(
                        accessor=MatrixSliceAccessor(
                            accessor,
                            row_window=row_window,
                            col_window=col_window,
                        ),
                    )
                else:
                    col_index = accessor.resolve_matrix_index(col_index, by=COL)

                    return Matrix[Any, Literal[1], T_co].from_accessor(
                        accessor=ColSliceAccessor(
                            accessor,
                            row_window=row_window,
                            col_index=col_index,
                        ),
                    )

            else:
                row_index = accessor.resolve_matrix_index(row_index, by=ROW)

                if isinstance(col_index, slice):
                    col_window = accessor.resolve_matrix_slice(col_index, by=COL)

                    return Matrix[Literal[1], Any, T_co].from_accessor(
                        accessor=RowSliceAccessor(
                            accessor,
                            row_index=row_index,
                            col_window=col_window,
                        ),
                    )
                else:
                    col_index = accessor.resolve_matrix_index(col_index, by=COL)

                    return accessor.matrix_access(row_index, col_index)

        elif isinstance(index, slice):
            window = accessor.resolve_vector_slice(index)

            return Matrix[Literal[1], Any, T_co].from_accessor(
                accessor=SliceAccessor(
                    accessor,
                    window=window,
                ),
            )

        else:
            index = accessor.resolve_vector_index(index)

            return accessor.vector_access(index)

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
        self = cls.__new__(cls)
        self._accessor = accessor
        return self

    @classmethod
    def from_matrix(cls, matrix: Matrix[M_co, N_co, T_co]) -> Self:
        return cls.from_accessor(matrix._accessor)

    @property
    def array(self) -> tuple[T_co, ...]:
        return self._accessor.to_tuple()

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self._accessor.shape

    @property
    def row_count(self) -> M_co:
        return self._accessor.row_count

    @property
    def col_count(self) -> N_co:
        return self._accessor.col_count

    def materialize(self) -> Matrix[M_co, N_co, T_co]:
        return Matrix(self.array, self.shape)

    def transpose(self) -> Matrix[N_co, M_co, T_co]:
        accessor = TransposeAccessor(self._accessor)
        return Matrix[N_co, M_co, T_co].from_accessor(accessor)

    def flip(self, *, by: Rule = Rule.ROW) -> Matrix[M_co, N_co, T_co]:
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
        n = operator.index(n) % 4
        if not n:
            return self
        target = self._accessor
        if n == 1:
            accessor = Rotate090Accessor(target)
        elif n == 2:
            accessor = Rotate180Accessor(target)
        else:
            accessor = Rotate270Accessor(target)
        return Matrix[Any, Any, T_co].from_accessor(accessor)

    def reverse(self) -> Matrix[M_co, N_co, T_co]:
        return self.rotate(2)

    def values(self, *, by: Rule = Rule.ROW) -> Iterator[T_co]:
        if by is ROW:
            iterable = self
        else:
            iterable = TransposeAccessor(self._accessor)
        return iter(iterable)

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
                yield Matrix[Literal[1], N_co, T_co].from_accessor(
                    accessor=RowSheerAccessor(target, row_index=row_index),
                )
        else:
            for col_index in range(self.col_count):
                yield Matrix[M_co, Literal[1], T_co].from_accessor(
                    accessor=ColSheerAccessor(target, col_index=col_index),
                )

    @overload
    def stack(self, other: Matrix[Any, N_co, S_co], *, by: Literal[Rule.ROW]) -> Matrix[Any, N_co, T_co | S_co]: ...
    @overload
    def stack(self, other: Matrix[M_co, Any, S_co], *, by: Literal[Rule.COL]) -> Matrix[M_co, Any, T_co | S_co]: ...
    @overload
    def stack(self, other: Matrix[Any, Any, S_co], *, by: Rule) -> Matrix[Any, Any, T_co | S_co]: ...
    @overload
    def stack(self, other: Matrix[Any, N_co, S_co]) -> Matrix[Any, N_co, T_co | S_co]: ...

    def stack(self, other: Matrix[Any, Any, S_co], *, by: Rule = Rule.ROW) -> Matrix[Any, Any, T_co | S_co]:
        target_head = self._accessor
        target_tail = other._accessor
        if __debug__:
            dy = ~by
            if target_head.shape[dy] != target_tail.shape[dy]:
                raise ValueError(f"cannot {by.handle}-stack matrices with differing number of {dy.handle}s")
        if by is ROW:
            accessor = RowStackAccessor(target_head, target_tail)
        else:
            accessor = ColStackAccessor(target_head, target_tail)
        return Matrix[Any, Any, T_co | S_co].from_accessor(accessor)
