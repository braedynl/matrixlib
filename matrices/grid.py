from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Iterator, Reversible, Sequence
from typing import (Any, Generic, Literal, Optional, TypeVar, Union, final,
                    overload)

from typing_extensions import Self, TypeAlias

from .abc import Shaped
from .key import Key
from .rule import COL, ROW, Rule

__all__ = ["AbstractGrid", "AbstractGridPermutation", "Grid"]

T = TypeVar("T")

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)

EvenNumber: TypeAlias = Literal[-4, -2, 0, 2, 4]
OddNumber: TypeAlias  = Literal[-3, -1, 1, 3]


@overload
def values(iterable: Reversible[T], /, *, reverse: Literal[True]) -> Iterator[T]: ...
@overload
def values(iterable: Iterable[T], /, *, reverse: bool = False) -> Iterator[T]: ...

def values(iterable, /, *, reverse=False):
    return (reversed if reverse else iter)(iterable)


class AbstractGrid(Shaped[M_co, N_co], Sequence[T_co], Generic[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, AbstractGrid):
            return (
                self.shape == other.shape
                and
                all(x is y or x == y for x, y in zip(self, other))
            )
        return NotImplemented

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> AbstractGrid[Literal[1], Any, T_co]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> AbstractGrid[Literal[1], Any, T_co]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> AbstractGrid[Any, Literal[1], T_co]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> AbstractGrid[Any, Any, T_co]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: Key[int, int]) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: Key[int, slice]) -> AbstractGrid[Literal[1], Any, T_co]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: Key[slice, int]) -> AbstractGrid[Any, Literal[1], T_co]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: Key[slice, slice]) -> AbstractGrid[Any, Any, T_co]: ...

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    def __deepcopy__(self, memo=None) -> Self:
        return self

    __copy__ = __deepcopy__

    @property
    @abstractmethod
    def array(self) -> Sequence[T_co]:
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> tuple[M_co, N_co]:
        raise NotImplementedError

    def transpose(self) -> AbstractGrid[N_co, M_co, T_co]:
        GridPermutation = GridTranspose
        return GridPermutation(self)

    def flip(self, *, by: Rule = Rule.ROW) -> AbstractGrid[M_co, N_co, T_co]:
        GridPermutation = (GridRowFlip, GridColFlip)[by.value]
        return GridPermutation(self)

    @overload
    def rotate(self, n: EvenNumber) -> AbstractGrid[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> AbstractGrid[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Union[AbstractGrid[M_co, N_co, T_co], AbstractGrid[N_co, M_co, T_co]]: ...
    @overload
    def rotate(self) -> AbstractGrid[N_co, M_co, T_co]: ...

    def rotate(self, n=1):
        if (n := n % 4):
            GridPermutation = (GridRotation90, GridRotation180, GridRotation270)[n - 1]
            return GridPermutation(self)
        return self

    def reverse(self) -> AbstractGrid[M_co, N_co, T_co]:
        return self.rotate(2)

    @overload
    def n(self, by: Literal[Rule.ROW]) -> M_co: ...
    @overload
    def n(self, by: Literal[Rule.COL]) -> N_co: ...
    @overload
    def n(self, by: Rule) -> Union[M_co, N_co]: ...

    def n(self, by):
        return self.shape[by.value]

    def values(self, *, by: Rule = Rule.ROW, reverse: bool = False) -> Iterator[T_co]:
        row_indices = range(self.nrows)
        col_indices = range(self.ncols)
        if by is Rule.ROW:
            for row_index in values(
                row_indices,
                reverse=reverse,
            ):
                for col_index in values(
                    col_indices,
                    reverse=reverse,
                ):
                    yield self[row_index, col_index]
        else:
            for col_index in values(
                col_indices,
                reverse=reverse,
            ):
                for row_index in values(
                    row_indices,
                    reverse=reverse,
                ):
                    yield self[row_index, col_index]

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[AbstractGrid[Literal[1], N_co, T_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[AbstractGrid[M_co, Literal[1], T_co]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[AbstractGrid[Any, Any, T_co]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[AbstractGrid[Literal[1], N_co, T_co]]: ...

    def slices(self, *, by=Rule.ROW, reverse=False):
        indices = range(self.n(by))
        key = Key()
        key[~by] = slice(None)
        for index in values(indices, reverse=reverse):
            key[by] = index
            yield self[key]

    def materialize(self) -> AbstractGrid[M_co, N_co, T_co]:
        return self

    def _resolve_vector_index(self, key: int) -> int:
        bound = len(self)
        if key < 0:
            key += bound
        if key < 0 or key >= bound:
            raise IndexError(f"there are {bound} items but index is {key}")
        return key

    def _resolve_matrix_index(self, key: int, *, by: Rule = Rule.ROW) -> int:
        bound = self.n(by)
        if key < 0:
            key += bound
        if key < 0 or key >= bound:
            handle = by.handle
            raise IndexError(f"there are {bound} {handle}s but index is {key}")
        return key

    def _resolve_vector_slice(self, key: slice) -> range:
        bound = len(self)
        return range(*key.indices(bound))

    def _resolve_matrix_slice(self, key: slice, *, by: Rule = Rule.ROW) -> range:
        bound = self.n(by)
        return range(*key.indices(bound))


@final
class Grid(AbstractGrid[M_co, N_co, T_co]):

    __slots__ = ("_array", "_shape")

    @overload
    def __init__(self, array: AbstractGrid[M_co, N_co, T_co]) -> None: ...
    @overload
    def __init__(self, array: Iterable[T_co] = (), shape: Optional[tuple[M_co, N_co]] = None) -> None: ...

    def __init__(self, array=(), shape=None):
        if isinstance(array, Grid):
            self._array = array._array
            self._shape = array._shape
            return
        self._array = tuple(array)
        if isinstance(array, AbstractGrid):
            self._shape = array.shape
        elif shape is None:
            self._shape = (1, len(self._array))
        else:
            self._shape = shape

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(array={self.array!r}, shape={self.shape!r})"

    def __hash__(self) -> int:
        return hash((self.array, self.shape))

    @overload
    def __getitem__(self, key: int) -> T_co: ...
    @overload
    def __getitem__(self, key: slice) -> Grid[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> Grid[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> Grid[Any, Literal[1], T_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> Grid[Any, Any, T_co]: ...
    @overload
    def __getitem__(self, key: Key[int, int]) -> T_co: ...
    @overload
    def __getitem__(self, key: Key[int, slice]) -> Grid[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, key: Key[slice, int]) -> Grid[Any, Literal[1], T_co]: ...
    @overload
    def __getitem__(self, key: Key[slice, slice]) -> Grid[Any, Any, T_co]: ...

    def __getitem__(self, key):
        array = self.array

        if isinstance(key, (tuple, Key)):
            row_key, col_key = key
            ncols = self.ncols

            if isinstance(row_key, slice):
                row_indices = self._resolve_matrix_slice(row_key, by=ROW)

                if isinstance(col_key, slice):
                    col_indices = self._resolve_matrix_slice(col_key, by=COL)
                    return Grid(
                        array=tuple(
                            array[row_index * ncols + col_index]
                            for row_index in row_indices
                            for col_index in col_indices
                        ),
                        shape=(len(row_indices), len(col_indices)),
                    )

                col_index = self._resolve_matrix_index(col_key, by=COL)
                return Grid(
                    array=tuple(
                        array[row_index * ncols + col_index]
                        for row_index in row_indices
                    ),
                    shape=(len(row_indices), 1),
                )

            row_index = self._resolve_matrix_index(row_key, by=ROW)

            if isinstance(col_key, slice):
                col_indices = self._resolve_matrix_slice(col_key, by=COL)
                return Grid(
                    array=tuple(
                        array[row_index * ncols + col_index]
                        for col_index in col_indices
                    ),
                    shape=(1, len(col_indices)),
                )

            col_index = self._resolve_matrix_index(col_key, by=COL)
            return array[row_index * ncols + col_index]

        if isinstance(key, slice):
            return Grid(array[key])

        return array[key]

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.array)

    def __reversed__(self) -> Iterator[T_co]:
        return reversed(self.array)

    def __contains__(self, value: object) -> bool:
        return value in self.array

    @property
    def array(self) -> tuple[T_co, ...]:
        return self._array

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self._shape  # type: ignore[return-value]


class AbstractGridPermutation(AbstractGrid[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ("_target",)

    def __init__(self, target: AbstractGrid[int, int, T_co]) -> None:
        self._target: AbstractGrid[int, int, T_co] = target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target={self.target!r})"

    @overload
    def __getitem__(self, key: int) -> T_co: ...
    @overload
    def __getitem__(self, key: slice) -> Grid[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> Grid[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> Grid[Any, Literal[1], T_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> Grid[Any, Any, T_co]: ...
    @overload
    def __getitem__(self, key: Key[int, int]) -> T_co: ...
    @overload
    def __getitem__(self, key: Key[int, slice]) -> Grid[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, key: Key[slice, int]) -> Grid[Any, Literal[1], T_co]: ...
    @overload
    def __getitem__(self, key: Key[slice, slice]) -> Grid[Any, Any, T_co]: ...

    def __getitem__(self, key):
        array = self.target.array

        if isinstance(key, (tuple, Key)):
            row_key, col_key = key

            permute_matrix_index = self._permute_matrix_index

            if isinstance(row_key, slice):
                row_indices = self._resolve_matrix_slice(row_key, by=ROW)

                if isinstance(col_key, slice):
                    col_indices = self._resolve_matrix_slice(col_key, by=COL)
                    return Grid(
                        array=tuple(
                            array[permute_matrix_index(row_index, col_index)]
                            for row_index in row_indices
                            for col_index in col_indices
                        ),
                        shape=(len(row_indices), len(col_indices)),
                    )

                col_index = self._resolve_matrix_index(col_key, by=COL)
                return Grid(
                    array=tuple(
                        array[permute_matrix_index(row_index, col_index)]
                        for row_index in row_indices
                    ),
                    shape=(len(row_indices), 1),
                )

            row_index = self._resolve_matrix_index(row_key, by=ROW)

            if isinstance(col_key, slice):
                col_indices = self._resolve_matrix_slice(col_key, by=COL)
                return Grid(
                    array=tuple(
                        array[permute_matrix_index(row_index, col_index)]
                        for col_index in col_indices
                    ),
                    shape=(1, len(col_indices)),
                )

            col_index = self._resolve_matrix_index(col_key, by=COL)
            return array[permute_matrix_index(row_index, col_index)]

        permute_vector_index = self._permute_vector_index

        if isinstance(key, slice):
            val_indices = self._resolve_vector_slice(key)
            return Grid(
                array=tuple(
                    array[permute_vector_index(val_index)]
                    for val_index in val_indices
                ),
                shape=(1, len(val_indices)),
            )

        val_index = self._resolve_vector_index(key)
        return array[permute_vector_index(val_index)]

    def __iter__(self) -> Iterator[T_co]:
        return self.values()

    def __reversed__(self) -> Iterator[T_co]:
        return self.values(reverse=True)

    def __contains__(self, value: object) -> bool:
        for x in self:
            if x is value or x == value:
                return True
        return False

    @property
    def array(self) -> Sequence[T_co]:
        return self

    @property
    def target(self) -> AbstractGrid[int, int, T_co]:
        return self._target

    @overload
    def n(self, by: Literal[Rule.ROW]) -> M_co: ...
    @overload
    def n(self, by: Literal[Rule.COL]) -> N_co: ...
    @overload
    def n(self, by: Rule) -> Union[M_co, N_co]: ...

    def n(self, by):
        return self.nrows if by is Rule.ROW else self.ncols

    def materialize(self) -> AbstractGrid[M_co, N_co, T_co]:
        return Grid(self)

    @abstractmethod
    def _permute_vector_index(self, val_index: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        raise NotImplementedError


class AbstractGridPermutationF(AbstractGridPermutation[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __init__(self, target: AbstractGrid[M_co, N_co, T_co]) -> None:
        super().__init__(target)
        self._target: AbstractGrid[M_co, N_co, T_co]

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self.target.shape

    @property
    def nrows(self) -> M_co:
        return self.target.nrows

    @property
    def ncols(self) -> N_co:
        return self.target.ncols

    @property
    def target(self) -> AbstractGrid[M_co, N_co, T_co]:
        return self._target


class AbstractGridPermutationR(AbstractGridPermutation[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __init__(self, target: AbstractGrid[N_co, M_co, T_co]) -> None:
        super().__init__(target)
        self._target: AbstractGrid[N_co, M_co, T_co]

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
    def target(self) -> AbstractGrid[N_co, M_co, T_co]:
        return self._target


@final
class GridTranspose(AbstractGridPermutationR[M_co, N_co, T_co]):

    __slots__ = ()

    def transpose(self) -> AbstractGrid[N_co, M_co, T_co]:
        return self.target

    def _permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        val_index = self._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )
        return val_index

    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        val_index = col_index * self.nrows + row_index
        return val_index


@final
class GridRowFlip(AbstractGridPermutationF[M_co, N_co, T_co]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> AbstractGrid[M_co, N_co, T_co]:
        return self.target if by is Rule.ROW else super().flip(by=by)

    def _permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        val_index = self._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )
        return val_index

    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        row_index = self.nrows - row_index - 1
        val_index = row_index * self.ncols + col_index
        return val_index


@final
class GridColFlip(AbstractGridPermutationF[M_co, N_co, T_co]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> AbstractGrid[M_co, N_co, T_co]:
        return self.target if by is Rule.COL else super().flip(by=by)

    def _permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        val_index = self._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )
        return val_index

    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        col_index = self.ncols - col_index - 1
        val_index = row_index * self.ncols + col_index
        return val_index


@final
class GridRotation90(AbstractGridPermutationR[M_co, N_co, T_co]):

    __slots__ = ()

    @overload
    def rotate(self, n: EvenNumber) -> AbstractGrid[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> AbstractGrid[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Union[AbstractGrid[M_co, N_co, T_co], AbstractGrid[N_co, M_co, T_co]]: ...
    @overload
    def rotate(self) -> AbstractGrid[N_co, M_co, T_co]: ...

    def rotate(self, n=1):
        return self.target if n % 4 == 3 else super().rotate(n)

    def _permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        val_index = self._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )
        return val_index

    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        row_index = self.nrows - row_index - 1
        val_index = col_index * self.nrows + row_index
        return val_index


@final
class GridRotation180(AbstractGridPermutationF[M_co, N_co, T_co]):

    __slots__ = ()

    @overload
    def rotate(self, n: EvenNumber) -> AbstractGrid[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> AbstractGrid[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Union[AbstractGrid[M_co, N_co, T_co], AbstractGrid[N_co, M_co, T_co]]: ...
    @overload
    def rotate(self) -> AbstractGrid[N_co, M_co, T_co]: ...

    def rotate(self, n=1):
        return self.target if n % 4 == 2 else super().rotate(n)

    def _permute_vector_index(self, val_index: int) -> int:
        val_index = len(self) - val_index - 1
        return val_index

    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        val_index = row_index * self.ncols + col_index
        val_index = self._permute_vector_index(
            val_index=val_index,
        )
        return val_index


@final
class GridRotation270(AbstractGridPermutationR[M_co, N_co, T_co]):

    __slots__ = ()

    @overload
    def rotate(self, n: EvenNumber) -> AbstractGrid[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> AbstractGrid[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Union[AbstractGrid[M_co, N_co, T_co], AbstractGrid[N_co, M_co, T_co]]: ...
    @overload
    def rotate(self) -> AbstractGrid[N_co, M_co, T_co]: ...

    def rotate(self, n=1):
        return self.target if n % 4 == 1 else super().rotate(n)

    def _permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        val_index = self._permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )
        return val_index

    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        col_index = self.ncols - col_index - 1
        val_index = col_index * self.nrows + row_index
        return val_index
