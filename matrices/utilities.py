from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import (Any, Generic, Literal, Optional, TypeVar, Union, final,
                    overload)

from typing_extensions import Self, TypeAlias

from .rule import COL, ROW, Rule

__all__ = [
    "EvenNumber",
    "OddNumber",
    "Mesh",
    "Grid",
    "RowGrid",
    "ColGrid",
]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)

EvenNumber: TypeAlias = Literal[-4, -2, 0, 2, 4]
OddNumber: TypeAlias  = Literal[-3, -1, 1, 3]


class Mesh(Sequence[T_co], Generic[M_co, N_co, T_co], metaclass=ABCMeta):

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

    def __len__(self) -> int:
        return self.nrows * self.ncols

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

    @property
    @abstractmethod
    def array(self) -> Sequence[T_co]:
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> tuple[M_co, N_co]:
        raise NotImplementedError

    @property
    def nrows(self) -> M_co:
        return self.shape[0]

    @property
    def ncols(self) -> N_co:
        return self.shape[1]

    @abstractmethod
    def materialize(self) -> Mesh[M_co, N_co, T_co]:
        raise NotImplementedError

    @abstractmethod
    def transpose(self) -> Mesh[N_co, M_co, T_co]:
        raise NotImplementedError

    @abstractmethod
    def flip(self, *, by: Rule = Rule.ROW) -> Mesh[M_co, N_co, T_co]:
        raise NotImplementedError

    @overload
    @abstractmethod
    def rotate(self, n: EvenNumber) -> Mesh[M_co, N_co, T_co]: ...
    @overload
    @abstractmethod
    def rotate(self, n: OddNumber) -> Mesh[N_co, M_co, T_co]: ...
    @overload
    @abstractmethod
    def rotate(self, n: int) -> Mesh[Any, Any, T_co]: ...
    @overload
    @abstractmethod
    def rotate(self) -> Mesh[N_co, M_co, T_co]: ...

    @abstractmethod
    def rotate(self, n=1):
        raise NotImplementedError

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

    @overload
    def values(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[T_co]: ...
    @overload
    def values(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[T_co]: ...
    @overload
    def values(self, *, by: Rule, reverse: bool = False) -> Iterator[T_co]: ...
    @overload
    def values(self, *, reverse: bool = False) -> Iterator[T_co]: ...

    def values(self, *, by=Rule.ROW, reverse=False):
        values = reversed if reverse else iter
        row_indices = range(self.nrows)
        col_indices = range(self.ncols)
        if by is Rule.ROW:
            for row_index in values(row_indices):
                for col_index in values(col_indices):
                    yield self[row_index, col_index]
        else:
            for col_index in values(col_indices):
                for row_index in values(row_indices):
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
        values  = reversed if reverse else iter
        indices = range(self.n(by))
        key = [None, None]
        key[(~by).value] = slice(None)
        for index in values(indices):
            key[(by).value] = index
            yield self[key]

    def resolve_vector_index(self, key: int) -> int:
        bound = len(self)
        if key < 0:
            key += bound
        if key < 0 or key >= bound:
            raise IndexError(f"there are {bound} values but index is {key}")
        return key

    def resolve_matrix_index(self, key: int, *, by: Rule = Rule.ROW) -> int:
        bound = self.n(by)
        if key < 0:
            key += bound
        if key < 0 or key >= bound:
            handle = by.handle
            raise IndexError(f"there are {bound} {handle}s but index is {key}")
        return key

    def resolve_vector_slice(self, key: slice) -> range:
        bound = len(self)
        return range(*key.indices(bound))

    def resolve_matrix_slice(self, key: slice, *, by: Rule = Rule.ROW) -> range:
        bound = self.n(by)
        return range(*key.indices(bound))


class Permutation(Mesh[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __hash__(self) -> int:
        return hash(self.materialize())

    @overload
    def __getitem__(self, key: int) -> T_co: ...
    @overload
    def __getitem__(self, key: slice) -> Mesh[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> Mesh[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> Mesh[Any, Literal[1], T_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> Mesh[Any, Any, T_co]: ...

    def __getitem__(self, key):
        array = self.target.array

        if isinstance(key, (tuple, list)):
            row_key, col_key = key

            permute_matrix_index = self.permute_matrix_index

            if isinstance(row_key, slice):
                row_indices = self.resolve_matrix_slice(row_key, by=ROW)

                if isinstance(col_key, slice):
                    col_indices = self.resolve_matrix_slice(col_key, by=COL)
                    return Grid(
                        array=tuple(
                            array[permute_matrix_index(row_index, col_index)]
                            for row_index in row_indices
                            for col_index in col_indices
                        ),
                        shape=(len(row_indices), len(col_indices)),
                    )

                col_index = self.resolve_matrix_index(col_key, by=COL)
                return ColGrid(
                    array=tuple(
                        array[permute_matrix_index(row_index, col_index)]
                        for row_index in row_indices
                    ),
                )

            row_index = self.resolve_matrix_index(row_key, by=ROW)

            if isinstance(col_key, slice):
                col_indices = self.resolve_matrix_slice(col_key, by=COL)
                return RowGrid(
                    array=tuple(
                        array[permute_matrix_index(row_index, col_index)]
                        for col_index in col_indices
                    ),
                )

            col_index = self.resolve_matrix_index(col_key, by=COL)
            return array[permute_matrix_index(row_index, col_index)]

        permute_vector_index = self.permute_vector_index

        if isinstance(key, slice):
            val_indices = self.resolve_vector_slice(key)
            return RowGrid(
                array=tuple(
                    array[permute_vector_index(val_index)]
                    for val_index in val_indices
                ),
            )

        val_index = self.resolve_vector_index(key)
        return array[permute_vector_index(val_index)]

    @property
    def array(self) -> Self:
        return self

    @property
    @abstractmethod
    def target(self) -> Mesh[Any, Any, T_co]:
        raise NotImplementedError

    def materialize(self) -> Mesh[M_co, N_co, T_co]:
        return Grid(self, self.shape)

    def transpose(self) -> Mesh[N_co, M_co, T_co]:
        return Transpose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> Mesh[M_co, N_co, T_co]:
        if by is Rule.ROW:
            return RowFlip(self)
        if by is Rule.COL:
            return ColFlip(self)
        raise RuntimeError

    @overload
    def rotate(self, n: EvenNumber) -> Mesh[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> Mesh[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Mesh[Any, Any, T_co]: ...
    @overload
    def rotate(self) -> Mesh[N_co, M_co, T_co]: ...

    def rotate(self, n=1):
        n = n % 4
        if n == 1:
            return Rotation90(self)
        if n == 2:
            return Rotation180(self)
        if n == 3:
            return Rotation270(self)
        return self

    @abstractmethod
    def permute_vector_index(self, val_index: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def permute_matrix_index(self, row_index: int, col_index: int) -> int:
        raise NotImplementedError


class ForwardPermutation(Permutation[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ("target")

    def __init__(self, target: Mesh[M_co, N_co, T_co]) -> None:
        self.target: Mesh[M_co, N_co, T_co] = target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target={self.target!r})"

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return self.target.shape

    @property
    def nrows(self) -> M_co:
        return self.target.nrows

    @property
    def ncols(self) -> N_co:
        return self.target.ncols


class ReversePermutation(Permutation[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ("target")

    def __init__(self, target: Mesh[N_co, M_co, T_co]) -> None:
        self.target: Mesh[N_co, M_co, T_co] = target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target={self.target!r})"

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
class Transpose(ReversePermutation[M_co, N_co, T_co]):

    __slots__ = ()

    def transpose(self) -> Mesh[N_co, M_co, T_co]:
        return self.target

    def permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        val_index = self.permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )
        return val_index

    def permute_matrix_index(self, row_index: int, col_index: int) -> int:
        val_index = col_index * self.nrows + row_index
        return val_index


@final
class RowFlip(ForwardPermutation[M_co, N_co, T_co]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> Mesh[M_co, N_co, T_co]:
        if by is Rule.ROW:
            return self.target
        return super().flip(by=by)

    def permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        val_index = self.permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )
        return val_index

    def permute_matrix_index(self, row_index: int, col_index: int) -> int:
        row_index = self.nrows - row_index - 1
        val_index = row_index * self.ncols + col_index
        return val_index


@final
class ColFlip(ForwardPermutation[M_co, N_co, T_co]):

    __slots__ = ()

    def flip(self, *, by: Rule = Rule.ROW) -> Mesh[M_co, N_co, T_co]:
        if by is Rule.COL:
            return self.target
        return super().flip(by=by)

    def permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        val_index = self.permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )
        return val_index

    def permute_matrix_index(self, row_index: int, col_index: int) -> int:
        col_index = self.ncols - col_index - 1
        val_index = row_index * self.ncols + col_index
        return val_index


@final
class Rotation90(ReversePermutation[M_co, N_co, T_co]):

    __slots__ = ()

    @overload
    def rotate(self, n: EvenNumber) -> Mesh[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> Mesh[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Mesh[Any, Any, T_co]: ...
    @overload
    def rotate(self) -> Mesh[N_co, M_co, T_co]: ...

    def rotate(self, n=1):
        if n % 4 == 3:
            return self.target
        return super().rotate(n)

    def permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        val_index = self.permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )
        return val_index

    def permute_matrix_index(self, row_index: int, col_index: int) -> int:
        row_index = self.nrows - row_index - 1
        val_index = col_index * self.nrows + row_index
        return val_index


@final
class Rotation180(ForwardPermutation[M_co, N_co, T_co]):

    __slots__ = ()

    @overload
    def rotate(self, n: EvenNumber) -> Mesh[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> Mesh[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Mesh[Any, Any, T_co]: ...
    @overload
    def rotate(self) -> Mesh[N_co, M_co, T_co]: ...

    def rotate(self, n=1):
        if n % 4 == 2:
            return self.target
        return super().rotate(n)

    def permute_vector_index(self, val_index: int) -> int:
        val_index = len(self) - val_index - 1
        return val_index

    def permute_matrix_index(self, row_index: int, col_index: int) -> int:
        val_index = row_index * self.ncols + col_index
        val_index = self.permute_vector_index(
            val_index=val_index,
        )
        return val_index


@final
class Rotation270(ReversePermutation[M_co, N_co, T_co]):

    __slots__ = ()

    @overload
    def rotate(self, n: EvenNumber) -> Mesh[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> Mesh[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Mesh[Any, Any, T_co]: ...
    @overload
    def rotate(self) -> Mesh[N_co, M_co, T_co]: ...

    def rotate(self, n=1):
        if n % 4 == 1:
            return self.target
        return super().rotate(n)

    def permute_vector_index(self, val_index: int) -> int:
        row_index, col_index = divmod(val_index, self.ncols)
        val_index = self.permute_matrix_index(
            row_index=row_index,
            col_index=col_index,
        )
        return val_index

    def permute_matrix_index(self, row_index: int, col_index: int) -> int:
        col_index = self.ncols - col_index - 1
        val_index = col_index * self.nrows + row_index
        return val_index


class Materialization(Mesh[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __hash__(self) -> int:
        return hash((self.array, self.shape))

    def __len__(self) -> int:
        return len(self.array)

    @overload
    def __getitem__(self, key: int) -> T_co: ...
    @overload
    def __getitem__(self, key: slice) -> Mesh[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> Mesh[Literal[1], Any, T_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> Mesh[Any, Literal[1], T_co]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> Mesh[Any, Any, T_co]: ...

    def __getitem__(self, key):
        array = self.array

        if isinstance(key, (tuple, list)):
            row_key, col_key = key
            ncols = self.ncols

            if isinstance(row_key, slice):
                row_indices = self.resolve_matrix_slice(row_key, by=ROW)

                if isinstance(col_key, slice):
                    col_indices = self.resolve_matrix_slice(col_key, by=COL)
                    return Grid(
                        array=tuple(
                            array[row_index * ncols + col_index]
                            for row_index in row_indices
                            for col_index in col_indices
                        ),
                        shape=(len(row_indices), len(col_indices)),
                    )

                col_index = self.resolve_matrix_index(col_key, by=COL)
                return ColGrid(
                    array=tuple(
                        array[row_index * ncols + col_index]
                        for row_index in row_indices
                    ),
                )

            row_index = self.resolve_matrix_index(row_key, by=ROW)

            if isinstance(col_key, slice):
                col_indices = self.resolve_matrix_slice(col_key, by=COL)
                return RowGrid(
                    array=tuple(
                        array[row_index * ncols + col_index]
                        for col_index in col_indices
                    ),
                )

            col_index = self.resolve_matrix_index(col_key, by=COL)
            return array[row_index * ncols + col_index]

        if isinstance(key, slice):
            return RowGrid(array[key])

        return array[key]

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.array)

    def __reversed__(self) -> Iterator[T_co]:
        return reversed(self.array)

    def __contains__(self, value: object) -> bool:
        return value in self.array

    def index(self, value: Any, start: int = 0, stop: Optional[int] = None) -> int:
        return self.array.index(value, start, stop)  # type: ignore[arg-type]

    def count(self, value: Any) -> int:
        return self.array.count(value)

    def materialize(self) -> Self:
        return self

    def transpose(self) -> Mesh[N_co, M_co, T_co]:
        return Transpose(self)

    def flip(self, *, by: Rule = Rule.ROW) -> Mesh[M_co, N_co, T_co]:
        if by is Rule.ROW:
            return RowFlip(self)
        if by is Rule.COL:
            return ColFlip(self)
        raise RuntimeError

    @overload
    def rotate(self, n: EvenNumber) -> Mesh[M_co, N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> Mesh[N_co, M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Mesh[Any, Any, T_co]: ...
    @overload
    def rotate(self) -> Mesh[N_co, M_co, T_co]: ...

    def rotate(self, n=1):
        n = n % 4
        if n == 1:
            return Rotation90(self)
        if n == 2:
            return Rotation180(self)
        if n == 3:
            return Rotation270(self)
        return self


@final
class Grid(Materialization[M_co, N_co, T_co], Generic[M_co, N_co, T_co]):

    __slots__ = ("array", "shape")

    def __init__(self, array: Iterable[T_co], shape: tuple[M_co, N_co]) -> None:
        self.array: tuple[T_co, ...] = tuple(array)
        if __debug__:
            if any(n < 0 for n in shape):  # type: ignore[operator]
                raise ValueError("shape must contain non-negative values")
            if (
                ((nrows := shape[0]) * (ncols := shape[1]))
                !=
                (size := len(self.array))
            ):
                raise ValueError(f"cannot interpret size {size} iterable as shape {nrows} × {ncols}")
        self.shape: tuple[M_co, N_co] = tuple(shape)  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Grid(array={self.array!r}, shape={self.shape!r})"


@final
class RowGrid(Materialization[Literal[1], N_co, T_co], Generic[N_co, T_co]):

    __slots__ = ("array")

    def __init__(self, array: Iterable[T_co]) -> None:
        self.array: tuple[T_co, ...] = tuple(array)

    def __repr__(self) -> str:
        return f"RowGrid(array={self.array!r})"

    @property
    def shape(self) -> tuple[Literal[1], N_co]:
        return (1, len(self.array))  # type: ignore[return-value]

    def transpose(self) -> Mesh[N_co, Literal[1], T_co]:
        return ColGrid(self.array)

    def flip(self, *, by: Rule = Rule.ROW) -> Mesh[Literal[1], N_co, T_co]:
        if by is Rule.ROW:
            return self
        return super().flip(by=by)

    @overload
    def rotate(self, n: EvenNumber) -> Mesh[Literal[1], N_co, T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> Mesh[N_co, Literal[1], T_co]: ...
    @overload
    def rotate(self, n: int) -> Mesh[Any, Any, T_co]: ...
    @overload
    def rotate(self) -> Mesh[N_co, Literal[1], T_co]: ...

    def rotate(self, n=1):
        if n % 4 == 3:
            return self.transpose()
        return super().rotate(n)


@final
class ColGrid(Materialization[M_co, Literal[1], T_co], Generic[M_co, T_co]):

    __slots__ = ("array")

    def __init__(self, array: Iterable[T_co]) -> None:
        self.array: tuple[T_co, ...] = tuple(array)

    def __repr__(self) -> str:
        return f"ColGrid(array={self.array!r})"

    @property
    def shape(self) -> tuple[M_co, Literal[1]]:
        return (len(self.array), 1)  # type: ignore[return-value]

    def transpose(self) -> Mesh[Literal[1], M_co, T_co]:
        return RowGrid(self.array)

    def flip(self, *, by: Rule = Rule.ROW) -> Mesh[M_co, Literal[1], T_co]:
        if by is Rule.COL:
            return self
        return super().flip(by=by)

    @overload
    def rotate(self, n: EvenNumber) -> Mesh[M_co, Literal[1], T_co]: ...
    @overload
    def rotate(self, n: OddNumber) -> Mesh[Literal[1], M_co, T_co]: ...
    @overload
    def rotate(self, n: int) -> Mesh[Any, Any, T_co]: ...
    @overload
    def rotate(self) -> Mesh[Literal[1], M_co, T_co]: ...

    def rotate(self, n=1):
        if n % 4 == 1:
            return self.transpose()
        return super().rotate(n)
