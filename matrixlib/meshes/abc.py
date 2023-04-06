from __future__ import annotations

import operator
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Generic, Literal, SupportsIndex, TypeVar, Union, overload

from typing_extensions import override

from ..rule import Rule

__all__ = ["Mesh", "InnerMesh", "OuterMesh"]

T_co = TypeVar("T_co", covariant=True)
M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


def resolve_index(key: SupportsIndex, bound: int) -> int:
    index = operator.index(key)
    if index < 0:
        index += bound
    if index < 0 or index >= bound:
        raise IndexError
    return index


def resolve_slice(key: slice, bound: int) -> range:
    return range(*key.indices(bound))


class Mesh(Generic[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    def __len__(self) -> int:
        return self.nrows * self.ncols

    def __iter__(self) -> Iterator[T_co]:
        return self.values()

    def __reversed__(self) -> Iterator[T_co]:
        return self.values(reverse=True)

    def __contains__(self, value: object) -> bool:
        for val in self:
            if val == value:
                return True
        return False

    @property
    def shape(self) -> tuple[M_co, N_co]:
        return (self.nrows, self.ncols)

    @property
    @abstractmethod
    def nrows(self) -> M_co:
        raise NotImplementedError

    @property
    @abstractmethod
    def ncols(self) -> N_co:
        raise NotImplementedError

    @overload
    def n(self, by: Literal[Rule.ROW]) -> M_co: ...
    @overload
    def n(self, by: Literal[Rule.COL]) -> N_co: ...
    @overload
    def n(self, by: Rule) -> Union[M_co, N_co]: ...

    def n(self, by: Rule) -> Union[M_co, N_co]:
        return self.shape[by.value]

    @abstractmethod
    def inner_get(self, index: int) -> T_co:
        raise NotImplementedError

    @abstractmethod
    def outer_get(self, row_index: int, col_index: int) -> T_co:
        raise NotImplementedError

    def resolve_inner_index(self, key: SupportsIndex) -> int:
        bound = len(self)
        try:
            index = resolve_index(key, bound)
        except IndexError:
            raise IndexError(f"there are {bound} values, but index is {key}") from None
        else:
            return index

    def resolve_inner_slice(self, key: slice) -> range:
        bound = len(self)
        return resolve_slice(key, bound)

    def resolve_outer_index(self, key: SupportsIndex, *, by: Rule) -> int:
        bound = self.n(by)
        try:
            index = resolve_index(key, bound)
        except IndexError:
            handle = by.handle
            raise IndexError(f"there are {bound} {handle}s, but index is {key}") from None
        else:
            return index

    def resolve_outer_slice(self, key: slice, *, by: Rule) -> range:
        bound = self.n(by)
        return resolve_slice(key, bound)

    def values(self, *, by: Rule = Rule.ROW, reverse: bool = False) -> Iterator[T_co]:
        nrows = self.nrows
        ncols = self.ncols
        if reverse:
            row_indices = range(nrows - 1, -1, -1)
            col_indices = range(ncols - 1, -1, -1)
        else:
            row_indices = range(nrows)
            col_indices = range(ncols)
        if by is Rule.ROW:
            for row_index in row_indices:
                for col_index in col_indices:
                    yield self.outer_get(row_index, col_index)
        else:
            for col_index in col_indices:
                for row_index in row_indices:
                    yield self.outer_get(row_index, col_index)


class InnerMesh(Mesh[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    @override
    def outer_get(self, row_index: int, col_index: int) -> T_co:
        index = row_index * self.ncols + col_index
        return self.inner_get(index)


class OuterMesh(Mesh[M_co, N_co, T_co], metaclass=ABCMeta):

    __slots__ = ()

    @override
    def inner_get(self, index: int) -> T_co:
        row_index, col_index = divmod(index, self.ncols)
        return self.outer_get(row_index, col_index)
