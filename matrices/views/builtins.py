from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import (Any, Generic, Literal, Protocol, SupportsIndex, TypeVar,
                    Union, overload)

from ..abc import (ComplexMatrixLike, IntegerMatrixLike, MatrixLike,
                   RealMatrixLike)
from ..builtins import (ComplexMatrix, ComplexMatrixOperatorsMixin,
                        IntegerMatrix, IntegerMatrixOperatorsMixin, Matrix,
                        MatrixOperatorsMixin, RealMatrix,
                        RealMatrixOperatorsMixin)
from ..rule import COL, ROW, Rule
from .abc import MatrixViewLike

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class SupportsMatrixPermutation(Protocol[T_co, M_co, N_co]):

    @property
    def target(self) -> MatrixLike[T_co, M_co, N_co]:
        raise NotImplementedError

    def _resolve_vector_index(self, key: SupportsIndex) -> int:
        raise NotImplementedError

    def _resolve_matrix_index(self, key: SupportsIndex, *, by: Rule = Rule.ROW) -> int:
        raise NotImplementedError

    def _resolve_vector_slice(self, key: slice) -> range:
        raise NotImplementedError

    def _resolve_matrix_slice(self, key: slice, *, by: Rule = Rule.ROW) -> range:
        raise NotImplementedError


class MatrixPermutationMixin(Generic[T_co, M_co, N_co], metaclass=ABCMeta):

    __slots__ = ()

    @overload
    def __getitem__(
        self: SupportsMatrixPermutation[T_co, M_co, N_co],
        key: int,
    ) -> T_co: ...
    @overload
    def __getitem__(
        self: SupportsMatrixPermutation[T_co, M_co, N_co],
        key: slice,
    ) -> MatrixLike[T_co, Literal[1], Any]: ...
    @overload
    def __getitem__(
        self: SupportsMatrixPermutation[T_co, M_co, N_co],
        key: tuple[int, int],
    ) -> T_co: ...
    @overload
    def __getitem__(
        self: SupportsMatrixPermutation[T_co, M_co, N_co],
        key: tuple[int, slice],
    ) -> MatrixLike[T_co, Literal[1], Any]: ...
    @overload
    def __getitem__(
        self: SupportsMatrixPermutation[T_co, M_co, N_co],
        key: tuple[slice, int],
    ) -> MatrixLike[T_co, Any, Literal[1]]: ...
    @overload
    def __getitem__(
        self: SupportsMatrixPermutation[T_co, M_co, N_co],
        key: tuple[slice, slice],
    ) -> MatrixLike[T_co, Any, Any]: ...

    def __getitem__(self, key):
        array = self.target.array

        permute_vector_index = self._permute_vector_index
        permute_matrix_index = self._permute_matrix_index

        if isinstance(key, (tuple, list)):
            row_key, col_key = key

            if isinstance(row_key, slice):
                row_indices = self._resolve_matrix_slice(row_key, by=ROW)

                if isinstance(col_key, slice):
                    col_indices = self._resolve_matrix_slice(col_key, by=COL)
                    return Matrix(
                        array=(
                            array[permute_matrix_index(row_index, col_index)]
                            for row_index in row_indices
                            for col_index in col_indices
                        ),
                        shape=(len(row_indices), len(col_indices)),
                    )

                col_index = self._resolve_matrix_index(col_key, by=COL)
                return Matrix(
                    array=(
                        array[permute_matrix_index(row_index, col_index)]
                        for row_index in row_indices
                    ),
                    shape=(len(row_indices), 1),
                )

            row_index = self._resolve_matrix_index(row_key, by=ROW)

            if isinstance(col_key, slice):
                col_indices = self._resolve_matrix_slice(col_key, by=COL)
                return Matrix(
                    array=(
                        array[permute_matrix_index(row_index, col_index)]
                        for col_index in col_indices
                    ),
                    shape=(1, len(col_indices)),
                )

            col_index = self._resolve_matrix_index(col_key, by=COL)
            return array[permute_matrix_index(row_index, col_index)]

        if isinstance(key, slice):
            val_indices = self._resolve_vector_slice(key)
            return Matrix(
                array=(
                    array[permute_vector_index(val_index)]
                    for val_index in val_indices
                ),
                shape=(1, len(val_indices)),
            )

        val_index = self._resolve_vector_index(key)
        return array[permute_vector_index(val_index)]

    @abstractmethod
    def _permute_vector_index(
        self: SupportsMatrixPermutation[T_co, M_co, N_co],
        val_index: int,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def _permute_matrix_index(
        self: SupportsMatrixPermutation[T_co, M_co, N_co],
        row_index: int,
        col_index: int,
    ) -> int:
        raise NotImplementedError


class MatrixTranspose(
    MatrixPermutationMixin[T_co, M_co, N_co],
    MatrixOperatorsMixin[T_co, M_co, N_co],
    MatrixViewLike[T_co, M_co, N_co],
):

    __slots__ = ("_target",)

    def __init__(self, target: MatrixLike[T_co, N_co, M_co]) -> None:
        self._target: MatrixLike[T_co, N_co, M_co] = target

    @property
    def array(self) -> Sequence[T_co]:
        return self

    @property
    def shape(self) -> tuple[M_co, N_co]:
        shape = self.target.shape
        return (shape[1], shape[0])

    @property
    def nrows(self) -> M_co:
        return self.target.ncols

    @property
    def ncols(self) -> N_co:
        return self.target.nrows

    @property
    def target(self) -> MatrixLike[T_co, N_co, M_co]:
        return self._target

    def transpose(self) -> MatrixLike[T_co, N_co, M_co]:
        return self.target

    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T_co, M_co, N_co]:
        MatrixPermutation = (MatrixRowFlip, MatrixColFlip)[by.value]
        return MatrixPermutation(self)

    def reverse(self) -> MatrixLike[T_co, M_co, N_co]:
        return MatrixReverse(self)

    @overload
    def n(self, by: Literal[Rule.ROW]) -> M_co: ...
    @overload
    def n(self, by: Literal[Rule.COL]) -> N_co: ...
    @overload
    def n(self, by: Rule) -> Union[M_co, N_co]: ...

    def n(self, by):
        return self.target.n(~by)

    def _permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        return self._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )

    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        return col_index * self.nrows + row_index


class ComplexMatrixTranspose(
    ComplexMatrixOperatorsMixin[M_co, N_co],
    ComplexMatrixLike[M_co, N_co],
    MatrixTranspose[complex, M_co, N_co],
):

    __slots__ = ()

    def __init__(self, target: ComplexMatrixLike[N_co, M_co]) -> None:
        self._target: ComplexMatrixLike[N_co, M_co] = target

    @overload
    def __getitem__(self, key: int) -> complex: ...
    @overload
    def __getitem__(self, key: slice) -> ComplexMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> complex: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> ComplexMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> ComplexMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> ComplexMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        value = MatrixTranspose.__getitem__(self, key)
        if isinstance(value, MatrixLike):
            return ComplexMatrix(value)
        return value

    @property
    def target(self) -> ComplexMatrixLike[N_co, M_co]:
        return self._target

    def transpose(self) -> ComplexMatrixLike[N_co, M_co]:
        return self.target

    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[M_co, N_co]:
        MatrixPermutation = (ComplexMatrixRowFlip, ComplexMatrixColFlip)[by.value]
        return MatrixPermutation(self)

    def reverse(self) -> ComplexMatrixLike[M_co, N_co]:
        return ComplexMatrixReverse(self)


class RealMatrixTranspose(
    RealMatrixOperatorsMixin[M_co, N_co],
    RealMatrixLike[M_co, N_co],
    MatrixTranspose[float, M_co, N_co],
):

    __slots__ = ()

    def __init__(self, target: RealMatrixLike[N_co, M_co]) -> None:
        self._target: RealMatrixLike[N_co, M_co] = target

    @overload
    def __getitem__(self, key: int) -> float: ...
    @overload
    def __getitem__(self, key: slice) -> RealMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> float: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> RealMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> RealMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> RealMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        value = MatrixTranspose.__getitem__(self, key)
        if isinstance(value, MatrixLike):
            return RealMatrix(value)
        return value

    @property
    def target(self) -> RealMatrixLike[N_co, M_co]:
        return self._target

    def transpose(self) -> RealMatrixLike[N_co, M_co]:
        return self.target

    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[M_co, N_co]:
        MatrixPermutation = (RealMatrixRowFlip, RealMatrixColFlip)[by.value]
        return MatrixPermutation(self)

    def reverse(self) -> RealMatrixLike[M_co, N_co]:
        return RealMatrixReverse(self)


class IntegerMatrixTranspose(
    IntegerMatrixOperatorsMixin[M_co, N_co],
    IntegerMatrixLike[M_co, N_co],
    MatrixTranspose[int, M_co, N_co],
):

    __slots__ = ()

    def __init__(self, target: IntegerMatrixLike[N_co, M_co]) -> None:
        self._target: IntegerMatrixLike[N_co, M_co] = target

    @overload
    def __getitem__(self, key: int) -> int: ...
    @overload
    def __getitem__(self, key: slice) -> IntegerMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> int: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> IntegerMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> IntegerMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> IntegerMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        value = MatrixTranspose.__getitem__(self, key)
        if isinstance(value, MatrixLike):
            return IntegerMatrix(value)
        return value

    @property
    def target(self) -> IntegerMatrixLike[N_co, M_co]:
        return self._target

    def transpose(self) -> IntegerMatrixLike[N_co, M_co]:
        return self.target

    def flip(self, *, by: Rule = Rule.ROW) -> IntegerMatrixLike[M_co, N_co]:
        MatrixPermutation = (IntegerMatrixRowFlip, IntegerMatrixColFlip)[by.value]
        return MatrixPermutation(self)

    def reverse(self) -> IntegerMatrixLike[M_co, N_co]:
        return IntegerMatrixReverse(self)


class MatrixRowFlip(
    MatrixPermutationMixin[T_co, M_co, N_co],
    MatrixOperatorsMixin[T_co, M_co, N_co],
    MatrixViewLike[T_co, M_co, N_co],
):

    __slots__ = ("_target",)

    def __init__(self, target: MatrixLike[T_co, M_co, N_co]) -> None:
        self._target: MatrixLike[T_co, M_co, N_co] = target

    @property
    def array(self) -> Sequence[T_co]:
        return self

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self.target.shape

    @property
    def target(self) -> MatrixLike[T_co, M_co, N_co]:
        return self._target

    def transpose(self) -> MatrixLike[T_co, N_co, M_co]:
        return MatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T_co, M_co, N_co]:
        if by is Rule.ROW:
            return self.target
        return MatrixColFlip(self)

    def reverse(self) -> MatrixLike[T_co, M_co, N_co]:
        return MatrixReverse(self)

    def _permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        return self._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )

    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        return (self.nrows - row_index - 1) * self.ncols + col_index


class ComplexMatrixRowFlip(
    ComplexMatrixOperatorsMixin[M_co, N_co],
    ComplexMatrixLike[M_co, N_co],
    MatrixRowFlip[complex, M_co, N_co],
):

    __slots__ = ()

    def __init__(self, target: ComplexMatrixLike[M_co, N_co]) -> None:
        self._target: ComplexMatrixLike[M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> complex: ...
    @overload
    def __getitem__(self, key: slice) -> ComplexMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> complex: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> ComplexMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> ComplexMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> ComplexMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        value = MatrixRowFlip.__getitem__(self, key)
        if isinstance(value, MatrixLike):
            return ComplexMatrix(value)
        return value

    @property
    def target(self) -> ComplexMatrixLike[M_co, N_co]:
        return self._target

    def transpose(self) -> ComplexMatrixLike[N_co, M_co]:
        return ComplexMatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[M_co, N_co]:
        if by is Rule.ROW:
            return self.target
        return ComplexMatrixColFlip(self)

    def reverse(self) -> ComplexMatrixLike[M_co, N_co]:
        return ComplexMatrixReverse(self)


class RealMatrixRowFlip(
    RealMatrixOperatorsMixin[M_co, N_co],
    RealMatrixLike[M_co, N_co],
    MatrixRowFlip[float, M_co, N_co],
):

    __slots__ = ()

    def __init__(self, target: RealMatrixLike[M_co, N_co]) -> None:
        self._target: RealMatrixLike[M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> float: ...
    @overload
    def __getitem__(self, key: slice) -> RealMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> float: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> RealMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> RealMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> RealMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        value = MatrixRowFlip.__getitem__(self, key)
        if isinstance(value, MatrixLike):
            return RealMatrix(value)
        return value

    @property
    def target(self) -> RealMatrixLike[M_co, N_co]:
        return self._target

    def transpose(self) -> RealMatrixLike[N_co, M_co]:
        return RealMatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[M_co, N_co]:
        if by is Rule.ROW:
            return self.target
        return RealMatrixColFlip(self)

    def reverse(self) -> RealMatrixLike[M_co, N_co]:
        return RealMatrixReverse(self)


class IntegerMatrixRowFlip(
    IntegerMatrixOperatorsMixin[M_co, N_co],
    IntegerMatrixLike[M_co, N_co],
    MatrixRowFlip[int, M_co, N_co],
):

    __slots__ = ()

    def __init__(self, target: IntegerMatrixLike[M_co, N_co]) -> None:
        self._target: IntegerMatrixLike[M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> int: ...
    @overload
    def __getitem__(self, key: slice) -> IntegerMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> int: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> IntegerMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> IntegerMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> IntegerMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        value = MatrixRowFlip.__getitem__(self, key)
        if isinstance(value, MatrixLike):
            return IntegerMatrix(value)
        return value

    @property
    def target(self) -> IntegerMatrixLike[M_co, N_co]:
        return self._target

    def transpose(self) -> IntegerMatrixLike[N_co, M_co]:
        return IntegerMatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> IntegerMatrixLike[M_co, N_co]:
        if by is Rule.ROW:
            return self.target
        return IntegerMatrixColFlip(self)

    def reverse(self) -> IntegerMatrixLike[M_co, N_co]:
        return IntegerMatrixReverse(self)


class MatrixColFlip(
    MatrixPermutationMixin[T_co, M_co, N_co],
    MatrixOperatorsMixin[T_co, M_co, N_co],
    MatrixViewLike[T_co, M_co, N_co],
):

    __slots__ = ("_target",)

    def __init__(self, target: MatrixLike[T_co, M_co, N_co]) -> None:
        self._target: MatrixLike[T_co, M_co, N_co] = target

    @property
    def array(self) -> Sequence[T_co]:
        return self

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self.target.shape

    @property
    def target(self) -> MatrixLike[T_co, M_co, N_co]:
        return self._target

    def transpose(self) -> MatrixLike[T_co, N_co, M_co]:
        return MatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T_co, M_co, N_co]:
        if by is Rule.COL:
            return self.target
        return MatrixRowFlip(self)

    def reverse(self) -> MatrixLike[T_co, M_co, N_co]:
        return MatrixReverse(self)

    def _permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        return self._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )

    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        return row_index * self.ncols + (self.ncols - col_index - 1)


class ComplexMatrixColFlip(
    ComplexMatrixOperatorsMixin[M_co, N_co],
    ComplexMatrixLike[M_co, N_co],
    MatrixColFlip[complex, M_co, N_co],
):

    __slots__ = ()

    def __init__(self, target: ComplexMatrixLike[M_co, N_co]) -> None:
        self._target: ComplexMatrixLike[M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> complex: ...
    @overload
    def __getitem__(self, key: slice) -> ComplexMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> complex: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> ComplexMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> ComplexMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> ComplexMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        value = MatrixColFlip.__getitem__(self, key)
        if isinstance(value, MatrixLike):
            return ComplexMatrix(value)
        return value

    @property
    def target(self) -> ComplexMatrixLike[M_co, N_co]:
        return self._target

    def transpose(self) -> ComplexMatrixLike[N_co, M_co]:
        return ComplexMatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[M_co, N_co]:
        if by is Rule.COL:
            return self.target
        return ComplexMatrixRowFlip(self)

    def reverse(self) -> ComplexMatrixLike[M_co, N_co]:
        return ComplexMatrixReverse(self)


class RealMatrixColFlip(
    RealMatrixOperatorsMixin[M_co, N_co],
    RealMatrixLike[M_co, N_co],
    MatrixColFlip[float, M_co, N_co],
):

    __slots__ = ()

    def __init__(self, target: RealMatrixLike[M_co, N_co]) -> None:
        self._target: RealMatrixLike[M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> float: ...
    @overload
    def __getitem__(self, key: slice) -> RealMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> float: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> RealMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> RealMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> RealMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        value = MatrixColFlip.__getitem__(self, key)
        if isinstance(value, MatrixLike):
            return RealMatrix(value)
        return value

    @property
    def target(self) -> RealMatrixLike[M_co, N_co]:
        return self._target

    def transpose(self) -> RealMatrixLike[N_co, M_co]:
        return RealMatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[M_co, N_co]:
        if by is Rule.COL:
            return self.target
        return RealMatrixRowFlip(self)

    def reverse(self) -> RealMatrixLike[M_co, N_co]:
        return RealMatrixReverse(self)


class IntegerMatrixColFlip(
    IntegerMatrixOperatorsMixin[M_co, N_co],
    IntegerMatrixLike[M_co, N_co],
    MatrixColFlip[int, M_co, N_co],
):

    __slots__ = ()

    def __init__(self, target: IntegerMatrixLike[M_co, N_co]) -> None:
        self._target: IntegerMatrixLike[M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> int: ...
    @overload
    def __getitem__(self, key: slice) -> IntegerMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> int: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> IntegerMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> IntegerMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> IntegerMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        value = MatrixColFlip.__getitem__(self, key)
        if isinstance(value, MatrixLike):
            return IntegerMatrix(value)
        return value

    @property
    def target(self) -> IntegerMatrixLike[M_co, N_co]:
        return self._target

    def transpose(self) -> IntegerMatrixLike[N_co, M_co]:
        return IntegerMatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> IntegerMatrixLike[M_co, N_co]:
        if by is Rule.COL:
            return self.target
        return IntegerMatrixRowFlip(self)

    def reverse(self) -> IntegerMatrixLike[M_co, N_co]:
        return IntegerMatrixReverse(self)


class MatrixReverse(
    MatrixPermutationMixin[T_co, M_co, N_co],
    MatrixOperatorsMixin[T_co, M_co, N_co],
    MatrixViewLike[T_co, M_co, N_co],
):

    __slots__ = ("_target",)

    def __init__(self, target: MatrixLike[T_co, M_co, N_co]) -> None:
        self._target: MatrixLike[T_co, M_co, N_co] = target

    @property
    def array(self) -> Sequence[T_co]:
        return self

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self.target.shape

    @property
    def target(self) -> MatrixLike[T_co, M_co, N_co]:
        return self._target

    def transpose(self) -> MatrixLike[T_co, N_co, M_co]:
        return MatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T_co, M_co, N_co]:
        MatrixPermutation = (MatrixRowFlip, MatrixColFlip)[by.value]
        return MatrixPermutation(self)

    def reverse(self) -> MatrixLike[T_co, M_co, N_co]:
        return self.target

    def _permute_vector_index(self, val_index: int) -> int:
        return len(self) - val_index - 1

    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        val_index = row_index * self.ncols + col_index
        return self._permute_vector_index(
            val_index=val_index,
        )


class ComplexMatrixReverse(
    ComplexMatrixOperatorsMixin[M_co, N_co],
    ComplexMatrixLike[M_co, N_co],
    MatrixReverse[complex, M_co, N_co],
):

    __slots__ = ()

    def __init__(self, target: ComplexMatrixLike[M_co, N_co]) -> None:
        self._target: ComplexMatrixLike[M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> complex: ...
    @overload
    def __getitem__(self, key: slice) -> ComplexMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> complex: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> ComplexMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> ComplexMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> ComplexMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        value = MatrixReverse.__getitem__(self, key)
        if isinstance(value, MatrixLike):
            return ComplexMatrix(value)
        return value

    @property
    def target(self) -> ComplexMatrixLike[M_co, N_co]:
        return self._target

    def transpose(self) -> ComplexMatrixLike[N_co, M_co]:
        return ComplexMatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[M_co, N_co]:
        MatrixPermutation = (ComplexMatrixRowFlip, ComplexMatrixColFlip)[by.value]
        return MatrixPermutation(self)

    def reverse(self) -> ComplexMatrixLike[M_co, N_co]:
        return self.target


class RealMatrixReverse(
    RealMatrixOperatorsMixin[M_co, N_co],
    RealMatrixLike[M_co, N_co],
    MatrixReverse[float, M_co, N_co],
):

    __slots__ = ()

    def __init__(self, target: RealMatrixLike[M_co, N_co]) -> None:
        self._target: RealMatrixLike[M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> float: ...
    @overload
    def __getitem__(self, key: slice) -> RealMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> float: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> RealMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> RealMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> RealMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        value = MatrixReverse.__getitem__(self, key)
        if isinstance(value, MatrixLike):
            return RealMatrix(value)
        return value

    @property
    def target(self) -> RealMatrixLike[M_co, N_co]:
        return self._target

    def transpose(self) -> RealMatrixLike[N_co, M_co]:
        return RealMatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[M_co, N_co]:
        MatrixPermutation = (RealMatrixRowFlip, RealMatrixColFlip)[by.value]
        return MatrixPermutation(self)

    def reverse(self) -> RealMatrixLike[M_co, N_co]:
        return self.target


class IntegerMatrixReverse(
    IntegerMatrixOperatorsMixin[M_co, N_co],
    IntegerMatrixLike[M_co, N_co],
    MatrixReverse[int, M_co, N_co],
):

    __slots__ = ()

    def __init__(self, target: IntegerMatrixLike[M_co, N_co]) -> None:
        self._target: IntegerMatrixLike[M_co, N_co] = target

    @overload
    def __getitem__(self, key: int) -> int: ...
    @overload
    def __getitem__(self, key: slice) -> IntegerMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> int: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> IntegerMatrixLike[Literal[1], Any]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> IntegerMatrixLike[Any, Literal[1]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> IntegerMatrixLike[Any, Any]: ...

    def __getitem__(self, key):
        value = MatrixReverse.__getitem__(self, key)
        if isinstance(value, MatrixLike):
            return IntegerMatrix(value)
        return value

    @property
    def target(self) -> IntegerMatrixLike[M_co, N_co]:
        return self._target

    def transpose(self) -> IntegerMatrixLike[N_co, M_co]:
        return IntegerMatrixTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> IntegerMatrixLike[M_co, N_co]:
        MatrixPermutation = (IntegerMatrixRowFlip, IntegerMatrixColFlip)[by.value]
        return MatrixPermutation(self)

    def reverse(self) -> IntegerMatrixLike[M_co, N_co]:
        return self.target
