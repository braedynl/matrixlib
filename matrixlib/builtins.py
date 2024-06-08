from __future__ import annotations

__all__ = ["Matrix"]

from collections.abc import Iterable, Sequence
from typing import (Any, Generic, Literal, Self, SupportsIndex, TypeVar,
                    overload)

from typing_extensions import override

from .accessors import (AbstractAccessor, ColSliceAccessor, ColVectorAccessor,
                        MatrixSliceAccessor, NULLARY_ACCESSOR_0x1,
                        NULLARY_ACCESSOR_1x0, RowSliceAccessor,
                        RowVectorAccessor, SliceAccessor, ValueAccessor)
from .rule import COL, ROW, Rule

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)

T_co = TypeVar("T_co", covariant=True)


class Matrix(Sequence[T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("_accessor",)
    _accessor: AbstractAccessor[M_co, N_co, T_co]

    @overload
    def __init__(self: Matrix[Literal[1], Literal[0], Any]) -> None: ...
    @overload
    def __init__(self: Matrix[Literal[1], Any, Any], array: Iterable[T_co]) -> None: ...
    @overload
    def __init__(self, array: Iterable[T_co], shape: tuple[M_co, N_co]) -> None: ...
    @overload
    def __init__(self: Matrix[Literal[1], Any, Any], array: Iterable[T_co], shape: Literal[Rule.ROW]) -> None: ...
    @overload
    def __init__(self: Matrix[Any, Literal[1], Any], array: Iterable[T_co], shape: Literal[Rule.COL]) -> None: ...
    @overload
    def __init__(self: Matrix[Any, Any, Any], array: Iterable[T_co], shape: Rule) -> None: ...

    def __init__(
        self,
        array: Iterable[T_co] = (),
        shape: Rule | tuple[M_co, N_co] = Rule.ROW,
    ) -> None:
        array = tuple(array)
        if isinstance(shape, Rule):
            size = len(array)
            if size == 1:
                self._accessor = ValueAccessor(array[0])  # pyright: ignore[reportAttributeAccessIssue]
                return
            if shape is ROW:
                if size > 1:
                    self._accessor = RowVectorAccessor(array)  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    self._accessor = NULLARY_ACCESSOR_1x0  # pyright: ignore[reportAttributeAccessIssue]
            else:
                if size > 1:
                    self._accessor = ColVectorAccessor(array)  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    self._accessor = NULLARY_ACCESSOR_0x1  # pyright: ignore[reportAttributeAccessIssue]
            return

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
                        MatrixSliceAccessor(
                            accessor,
                            row_window=row_window,
                            col_window=col_window,
                        ),
                    )
                else:
                    col_index = accessor.resolve_matrix_index(col_index, by=COL)

                    return Matrix[Any, Literal[1], T_co].from_accessor(
                        ColSliceAccessor(
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
                        RowSliceAccessor(
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
                SliceAccessor(
                    accessor,
                    window=window,
                ),
            )

        else:
            index = accessor.resolve_vector_index(index)

            return accessor.vector_access(index)

    @classmethod
    def from_accessor(cls, accessor: AbstractAccessor[M_co, N_co, T_co]) -> Self:
        self = cls.__new__(cls)
        self._accessor = accessor
        return self

    @classmethod
    def from_matrix(cls, matrix: Matrix[M_co, N_co, T_co]) -> Self:
        return cls.from_accessor(matrix._accessor)
