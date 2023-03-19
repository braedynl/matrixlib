from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Iterator, Reversible, Sequence
from typing import (Any, Generic, Literal, Optional, TypeVar, Union, final,
                    overload)

from typing_extensions import Self, TypeAlias

from .abc import Shaped
from .rule import COL, ROW, Rule

__all__ = [
    "EvenNumber",
    "OddNumber",
    "BaseGrid",
    "Grid",
]

T = TypeVar("T")
S = TypeVar("S")

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)

EvenNumber: TypeAlias = Literal[-8, -6, -4, -2, 0, 2, 4, 6, 8]
OddNumber: TypeAlias  = Literal[-7, -5, -3, -1, 1, 3, 5, 7]


@overload
def values(iterable: Reversible[T], /, *, reverse: Literal[True]) -> Iterator[T]: ...
@overload
def values(iterable: Iterable[T], /, *, reverse: bool = False) -> Iterator[T]: ...

def values(iterable, /, *, reverse=False):
    return (reversed if reverse else iter)(iterable)


class BaseGrid(Shaped[M_co, N_co], Sequence[T_co], Generic[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, BaseGrid):
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
    def __getitem__(self, key: slice) -> BaseGrid[Literal[1], Any, T_co]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> BaseGrid[Literal[1], Any, T_co]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> BaseGrid[Any, Literal[1], T_co]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> BaseGrid[Any, Any, T_co]: ...

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

    @abstractmethod
    def materialize(self) -> BaseGrid[M_co, N_co, T_co]:
        raise NotImplementedError

    def transpose(self) -> BaseGrid[N_co, M_co, T_co]:
        GridPermutation = GridTranspose
        return GridPermutation(self)

    def flip(self, *, by: Rule = Rule.ROW) -> BaseGrid[M_co, N_co, T_co]:
        GridPermutation = (GridRowFlip, GridColFlip)[by.value]
        return GridPermutation(self)

    @overload
    def rotate(self, n: EvenNumber) -> BaseGrid[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> BaseGrid[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Union[BaseGrid[M_co, N_co, T_co], BaseGrid[N_co, M_co, T_co]]: ...
    @overload
    def rotate(self) -> BaseGrid[N_co, M_co, T_co]: ...

    def rotate(self, n=1):
        if (n := n % 4):
            GridPermutation = (GridRotation90, GridRotation180, GridRotation270)[n - 1]
            return GridPermutation(self)
        return self

    def reverse(self) -> BaseGrid[M_co, N_co, T_co]:
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
            for row_index in values(row_indices, reverse=reverse):
                for col_index in values(col_indices, reverse=reverse):
                    yield self[row_index, col_index]
        else:
            for col_index in values(col_indices, reverse=reverse):
                for row_index in values(row_indices, reverse=reverse):
                    yield self[row_index, col_index]

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[BaseGrid[Literal[1], N_co, T_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[BaseGrid[M_co, Literal[1], T_co]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[BaseGrid[Any, Any, T_co]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[BaseGrid[Literal[1], N_co, T_co]]: ...

    def slices(self, *, by=Rule.ROW, reverse=False):
        indices = range(self.n(by))
        key = [None, None]
        key[(~by).value] = slice(None)
        for index in values(indices, reverse=reverse):
            key[(by).value] = index
            yield self[key]

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
class Grid(BaseGrid[M_co, N_co, T_co]):

    __slots__ = ("array", "shape")

    @overload
    def __init__(self, array: BaseGrid[M_co, N_co, T_co]) -> None: ...
    @overload
    def __init__(self, array: Iterable[T_co] = (), shape: Optional[tuple[M_co, N_co]] = None) -> None: ...

    def __init__(self, array=(), shape=None):
        self.array: tuple[T_co, ... ]  # type: ignore
        self.shape: tuple[M_co, N_co]  # type: ignore

        if type(array) is Grid:
            self.array = array.array
            self.shape = array.shape
            return

        self.array = tuple(array)
        if isinstance(array, BaseGrid):
            self.shape = array.shape
        elif shape is None:
            self.shape = (1, len(self.array))
        else:
            self.shape = shape

        if not __debug__:
            return

        nrows, ncols = self.shape
        nvals = len(self.array)

        if nrows < 0 or ncols < 0:
            raise ValueError("shape must contain non-negative values")
        if nvals != nrows * ncols:
            raise ValueError(f"cannot interpret size {nvals} iterable as shape ({nrows}, {ncols})")

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

    def __getitem__(self, key):
        array = self.array

        if isinstance(key, (tuple, list)):
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

    def materialize(self) -> Grid[M_co, N_co, T_co]:
        return self


class BaseGridPermutation(BaseGrid[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

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

    def __getitem__(self, key):
        array = self.target.array

        if isinstance(key, (tuple, list)):
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
    @abstractmethod
    def target(self) -> BaseGrid[Any, Any, T_co]:
        raise NotImplementedError

    def materialize(self) -> Grid[M_co, N_co, T_co]:
        return Grid(self)

    @overload
    def n(self, by: Literal[Rule.ROW]) -> M_co: ...
    @overload
    def n(self, by: Literal[Rule.COL]) -> N_co: ...
    @overload
    def n(self, by: Rule) -> Union[M_co, N_co]: ...

    def n(self, by):
        return self.nrows if by is Rule.ROW else self.ncols

    @abstractmethod
    def _permute_vector_index(self, val_index: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def _permute_matrix_index(self, row_index: int, col_index: int) -> int:
        raise NotImplementedError


class BaseGridPermutationF(BaseGridPermutation[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ("target",)

    def __init__(self, target: BaseGrid[M_co, N_co, T_co]) -> None:
        self.target: BaseGrid[M_co, N_co, T_co] = target

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self.target.shape

    @property
    def nrows(self) -> M_co:
        return self.target.nrows

    @property
    def ncols(self) -> N_co:
        return self.target.ncols


class BaseGridPermutationR(BaseGridPermutation[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ("target",)

    def __init__(self, target: BaseGrid[N_co, M_co, T_co]) -> None:
        self.target: BaseGrid[N_co, M_co, T_co] = target

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


@final
class GridTranspose(BaseGridPermutationR[M_co, N_co, T_co]):

    __slots__ = ()

    def transpose(self) -> BaseGrid[N_co, M_co, T_co]:
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
class GridRowFlip(BaseGridPermutationF[M_co, N_co, T_co]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> BaseGrid[M_co, N_co, T_co]:
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
class GridColFlip(BaseGridPermutationF[M_co, N_co, T_co]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> BaseGrid[M_co, N_co, T_co]:
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
class GridRotation90(BaseGridPermutationR[M_co, N_co, T_co]):

    __slots__ = ()

    @overload
    def rotate(self, n: EvenNumber) -> BaseGrid[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> BaseGrid[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Union[BaseGrid[M_co, N_co, T_co], BaseGrid[N_co, M_co, T_co]]: ...
    @overload
    def rotate(self) -> BaseGrid[N_co, M_co, T_co]: ...

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
class GridRotation180(BaseGridPermutationF[M_co, N_co, T_co]):

    __slots__ = ()

    @overload
    def rotate(self, n: EvenNumber) -> BaseGrid[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> BaseGrid[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Union[BaseGrid[M_co, N_co, T_co], BaseGrid[N_co, M_co, T_co]]: ...
    @overload
    def rotate(self) -> BaseGrid[N_co, M_co, T_co]: ...

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
class GridRotation270(BaseGridPermutationR[M_co, N_co, T_co]):

    __slots__ = ()

    @overload
    def rotate(self, n: EvenNumber) -> BaseGrid[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> BaseGrid[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Union[BaseGrid[M_co, N_co, T_co], BaseGrid[N_co, M_co, T_co]]: ...
    @overload
    def rotate(self) -> BaseGrid[N_co, M_co, T_co]: ...

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
