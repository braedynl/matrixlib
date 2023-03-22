from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Iterator, Reversible, Sequence
from typing import (Any, Literal, Optional, Protocol, TypeVar, Union, final,
                    overload)

from typing_extensions import TypeAlias

from .abc import Shaped
from .rule import COL, ROW, Rule

__all__ = ["EvenNumber", "OddNumber", "Mesh", "Grid"]

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


class MeshParts(Shaped[M_co, N_co], Protocol[M_co, N_co, T_co]):

    @property
    @abstractmethod
    def array(self) -> Sequence[T_co]:
        raise NotImplementedError


class Mesh(MeshParts[M_co, N_co, T_co], Sequence[T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, Mesh):
            return (
                self.shape == other.shape
                and
                all(x is y or x == y for x, y in zip(self, other))
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((tuple(self.array), tuple(self.shape)))

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> Mesh[Literal[1], Any, T_co]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> Mesh[Literal[1], Any, T_co]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> Mesh[Any, Literal[1], T_co]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> Mesh[Any, Any, T_co]: ...

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    @abstractmethod
    def materialize(self) -> Mesh[M_co, N_co, T_co]:
        raise NotImplementedError

    def transpose(self) -> Mesh[N_co, M_co, T_co]:
        return MeshTranspose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> Mesh[M_co, N_co, T_co]:
        MeshPermutation = (MeshRowFlip, MeshColFlip)[by.value]
        return MeshPermutation(self)

    @overload
    def rotate(self, n: EvenNumber) -> Mesh[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> Mesh[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Union[Mesh[M_co, N_co, T_co], Mesh[N_co, M_co, T_co]]: ...
    @overload
    def rotate(self) -> Mesh[N_co, M_co, T_co]: ...

    def rotate(self, n=1):
        if (n := n % 4):
            MeshPermutation = (MeshRotation090, MeshRotation180, MeshRotation270)[n - 1]
            return MeshPermutation(self)
        return self

    def reverse(self) -> Mesh[M_co, N_co, T_co]:
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
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[Mesh[Literal[1], N_co, T_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[Mesh[M_co, Literal[1], T_co]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[Mesh[Any, Any, T_co]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[Mesh[Literal[1], N_co, T_co]]: ...

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
            raise IndexError(f"there are {bound} values but index is {key}")
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


class MeshPermutationParts(MeshParts[M_co, N_co, T_co], Protocol[M_co, N_co, T_co]):

    @property
    @abstractmethod
    def target(self) -> Mesh[Any, Any, T_co]:
        raise NotImplementedError


class MeshPermutation(MeshPermutationParts[M_co, N_co, T_co], Mesh[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ("target",)

    def __init__(self, target: Mesh[Any, Any, T_co]) -> None:
        self.target: Mesh[Any, Any, T_co] = target

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


class MeshPermutationF(MeshPermutation[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __init__(self, target: Mesh[M_co, N_co, T_co]) -> None:
        super().__init__(target)
        self.target: Mesh[M_co, N_co, T_co]

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self.target.shape

    @property
    def nrows(self) -> M_co:
        return self.target.nrows

    @property
    def ncols(self) -> N_co:
        return self.target.ncols


class MeshPermutationR(MeshPermutation[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __init__(self, target: Mesh[N_co, M_co, T_co]) -> None:
        super().__init__(target)
        self.target: Mesh[N_co, M_co, T_co]

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
class MeshTranspose(MeshPermutationR[M_co, N_co, T_co]):

    __slots__ = ()

    def transpose(self) -> Mesh[N_co, M_co, T_co]:
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
class MeshRowFlip(MeshPermutationF[M_co, N_co, T_co]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> Mesh[M_co, N_co, T_co]:
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
class MeshColFlip(MeshPermutationF[M_co, N_co, T_co]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> Mesh[M_co, N_co, T_co]:
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
class MeshRotation090(MeshPermutationR[M_co, N_co, T_co]):

    __slots__ = ()

    @overload
    def rotate(self, n: EvenNumber) -> Mesh[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> Mesh[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Union[Mesh[M_co, N_co, T_co], Mesh[N_co, M_co, T_co]]: ...
    @overload
    def rotate(self) -> Mesh[N_co, M_co, T_co]: ...

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
class MeshRotation180(MeshPermutationF[M_co, N_co, T_co]):

    __slots__ = ()

    @overload
    def rotate(self, n: EvenNumber) -> Mesh[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> Mesh[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Union[Mesh[M_co, N_co, T_co], Mesh[N_co, M_co, T_co]]: ...
    @overload
    def rotate(self) -> Mesh[N_co, M_co, T_co]: ...

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
class MeshRotation270(MeshPermutationR[M_co, N_co, T_co]):

    __slots__ = ()

    @overload
    def rotate(self, n: EvenNumber) -> Mesh[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> Mesh[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Union[Mesh[M_co, N_co, T_co], Mesh[N_co, M_co, T_co]]: ...
    @overload
    def rotate(self) -> Mesh[N_co, M_co, T_co]: ...

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


@final
class Grid(Mesh[M_co, N_co, T_co]):

    __slots__ = ("array", "shape")

    @overload
    def __init__(self, array: Mesh[M_co, N_co, T_co]) -> None: ...
    @overload
    def __init__(self, array: Iterable[T_co] = (), shape: Optional[tuple[M_co, N_co]] = None) -> None: ...

    def __init__(self, array=(), shape=None):
        self.array: Sequence[T_co]     # type: ignore
        self.shape: tuple[M_co, N_co]  # type: ignore

        if isinstance(array, Mesh):
            self.shape = array.shape
            if type(array) is Grid:
                self.array = array.array
            else:
                self.array = tuple(array)
            return

        if isinstance(array, Sequence):
            self.array = array
        else:
            self.array = tuple(array)
        if shape:
            if __debug__:
                if any(n < 0 for n in shape):
                    raise ValueError("shape must contain non-negative values")
                if (
                    ((nrows := shape[0]) * (ncols := shape[1]))
                    !=
                    (size := len(self.array))
                ):
                    raise ValueError(f"cannot interpret size {size} iterable as shape {nrows} × {ncols}")
            self.shape = shape
        else:
            self.shape = (1, len(self.array))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(array={self.array!r}, shape={self.shape!r})"

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
