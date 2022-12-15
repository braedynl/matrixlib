from abc import ABCMeta, abstractmethod
from collections.abc import Collection, Iterator
from typing import Any, Generic, Literal, TypeAlias, TypeVar, overload

__all__ = [
    "ShapeLike",
    "AnyShape",
    "AnyRowVectorShape",
    "AnyColVectorShape",
    "AnyVectorShape",
]

NRowsT_co = TypeVar("NRowsT_co", covariant=True, bound=int)
NColsT_co = TypeVar("NColsT_co", covariant=True, bound=int)


class ShapeLike(Collection[NRowsT_co | NColsT_co], Generic[NRowsT_co, NColsT_co], metaclass=ABCMeta):

    __match_args__: tuple[Literal["nrows"], Literal["ncols"]]

    def __eq__(self, other: Any) -> bool: ...
    def __len__(self) -> Literal[2]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: Literal[0]) -> NRowsT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: Literal[1]) -> NColsT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> NRowsT_co | NColsT_co: ...
    def __iter__(self) -> Iterator[NRowsT_co | NColsT_co]: ...
    def __reversed__(self) -> Iterator[NRowsT_co | NColsT_co]: ...
    def __contains__(self, value: Any) -> bool: ...

    @property
    def nrows(self) -> NRowsT_co: ...
    @property
    def ncols(self) -> NColsT_co: ...


AnyShape: TypeAlias = ShapeLike[int, int]
AnyRowVectorShape: TypeAlias = ShapeLike[Literal[1], int]
AnyColVectorShape: TypeAlias = ShapeLike[int, Literal[1]]
AnyVectorShape: TypeAlias = AnyRowVectorShape | AnyColVectorShape
