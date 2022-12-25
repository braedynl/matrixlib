from abc import ABCMeta, abstractmethod
from collections.abc import Collection, Iterator
from typing import Any, Generic, Literal, TypeVar, overload

__all__ = ["ShapeLike"]

M_co = TypeVar("M_co", covariant=True, bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)


class ShapeLike(Collection[M_co | N_co], Generic[M_co, N_co], metaclass=ABCMeta):

    __slots__: tuple[()]
    __match_args__: tuple[Literal["nrows"], Literal["ncols"]]

    def __eq__(self, other: Any) -> bool: ...
    def __len__(self) -> Literal[2]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: Literal[0]) -> M_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: Literal[1]) -> N_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> M_co | N_co: ...
    def __iter__(self) -> Iterator[M_co | N_co]: ...
    def __reversed__(self) -> Iterator[M_co | N_co]: ...
    def __contains__(self, value: Any) -> bool: ...

    @property
    def nrows(self) -> M_co: ...
    @property
    def ncols(self) -> N_co: ...
