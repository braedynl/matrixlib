from abc import ABCMeta, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, Generic, Literal, TypeVar, overload

from .rule import Rule
from .shapes import ShapeLike
from .typeshed import (SupportsAbs, SupportsAdd, SupportsConjugate,
                       SupportsFloorDiv, SupportsMod, SupportsMul, SupportsNeg,
                       SupportsPos, SupportsPow, SupportsRAdd,
                       SupportsRFloorDiv, SupportsRMod, SupportsRMul,
                       SupportsRPow, SupportsRSub, SupportsRTrueDiv,
                       SupportsSub, SupportsTrueDiv)

__all__ = ["MatrixLike"]

T  = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")

DTypeT_co = TypeVar("DTypeT_co", covariant=True)
NRowsT_co = TypeVar("NRowsT_co", covariant=True, bound=int)
NColsT_co = TypeVar("NColsT_co", covariant=True, bound=int)


class MatrixLike(Sequence[DTypeT_co], Generic[DTypeT_co, NRowsT_co, NColsT_co], metaclass=ABCMeta):

    def __eq__(self, other: Any) -> bool: ...
    def __lt__(self, other: MatrixLike[DTypeT_co, NRowsT_co, NColsT_co]) -> bool: ...
    def __le__(self, other: MatrixLike[DTypeT_co, NRowsT_co, NColsT_co]) -> bool: ...
    def __gt__(self, other: MatrixLike[DTypeT_co, NRowsT_co, NColsT_co]) -> bool: ...
    def __ge__(self, other: MatrixLike[DTypeT_co, NRowsT_co, NColsT_co]) -> bool: ...
    def __len__(self) -> int: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> DTypeT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> MatrixLike[DTypeT_co, Literal[1], int]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> DTypeT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> MatrixLike[DTypeT_co, Literal[1], int]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> MatrixLike[DTypeT_co, int, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> MatrixLike[DTypeT_co, int, int]: ...
    def __iter__(self) -> Iterator[DTypeT_co]: ...
    def __reversed__(self) -> Iterator[DTypeT_co]: ...
    def __contains__(self, value: Any) -> bool: ...
    @overload
    @abstractmethod
    def __add__(self: MatrixLike[SupportsAdd[T1, T2], NRowsT_co, NColsT_co], other: MatrixLike[T1, NRowsT_co, NColsT_co]) -> MatrixLike[T2, NRowsT_co, NColsT_co]: ...
    @overload
    @abstractmethod
    def __add__(self: MatrixLike[T1, NRowsT_co, NColsT_co], other: MatrixLike[SupportsRAdd[T1, T2], NRowsT_co, NColsT_co]) -> MatrixLike[T2, NRowsT_co, NColsT_co]: ...
    @overload
    @abstractmethod
    def __sub__(self: MatrixLike[SupportsSub[T1, T2], NRowsT_co, NColsT_co], other: MatrixLike[T1, NRowsT_co, NColsT_co]) -> MatrixLike[T2, NRowsT_co, NColsT_co]: ...
    @overload
    @abstractmethod
    def __sub__(self: MatrixLike[T1, NRowsT_co, NColsT_co], other: MatrixLike[SupportsRSub[T1, T2], NRowsT_co, NColsT_co]) -> MatrixLike[T2, NRowsT_co, NColsT_co]: ...
    @overload
    @abstractmethod
    def __mul__(self: MatrixLike[SupportsMul[T1, T2], NRowsT_co, NColsT_co], other: MatrixLike[T1, NRowsT_co, NColsT_co]) -> MatrixLike[T2, NRowsT_co, NColsT_co]: ...
    @overload
    @abstractmethod
    def __mul__(self: MatrixLike[T1, NRowsT_co, NColsT_co], other: MatrixLike[SupportsRMul[T1, T2], NRowsT_co, NColsT_co]) -> MatrixLike[T2, NRowsT_co, NColsT_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self: MatrixLike[SupportsTrueDiv[T1, T2], NRowsT_co, NColsT_co], other: MatrixLike[T1, NRowsT_co, NColsT_co]) -> MatrixLike[T2, NRowsT_co, NColsT_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self: MatrixLike[T1, NRowsT_co, NColsT_co], other: MatrixLike[SupportsRTrueDiv[T1, T2], NRowsT_co, NColsT_co]) -> MatrixLike[T2, NRowsT_co, NColsT_co]: ...
    @overload
    @abstractmethod
    def __floordiv__(self: MatrixLike[SupportsFloorDiv[T1, T2], NRowsT_co, NColsT_co], other: MatrixLike[T1, NRowsT_co, NColsT_co]) -> MatrixLike[T2, NRowsT_co, NColsT_co]: ...
    @overload
    @abstractmethod
    def __floordiv__(self: MatrixLike[T1, NRowsT_co, NColsT_co], other: MatrixLike[SupportsRFloorDiv[T1, T2], NRowsT_co, NColsT_co]) -> MatrixLike[T2, NRowsT_co, NColsT_co]: ...
    @overload
    @abstractmethod
    def __mod__(self: MatrixLike[SupportsMod[T1, T2], NRowsT_co, NColsT_co], other: MatrixLike[T1, NRowsT_co, NColsT_co]) -> MatrixLike[T2, NRowsT_co, NColsT_co]: ...
    @overload
    @abstractmethod
    def __mod__(self: MatrixLike[T1, NRowsT_co, NColsT_co], other: MatrixLike[SupportsRMod[T1, T2], NRowsT_co, NColsT_co]) -> MatrixLike[T2, NRowsT_co, NColsT_co]: ...
    @overload
    @abstractmethod
    def __pow__(self: MatrixLike[SupportsPow[T1, T2], NRowsT_co, NColsT_co], other: MatrixLike[T1, NRowsT_co, NColsT_co]) -> MatrixLike[T2, NRowsT_co, NColsT_co]: ...
    @overload
    @abstractmethod
    def __pow__(self: MatrixLike[T1, NRowsT_co, NColsT_co], other: MatrixLike[SupportsRPow[T1, T2], NRowsT_co, NColsT_co]) -> MatrixLike[T2, NRowsT_co, NColsT_co]: ...
    @abstractmethod
    def __and__(self, other: MatrixLike[Any, NRowsT_co, NColsT_co]) -> MatrixLike[bool, NRowsT_co, NColsT_co]: ...
    @abstractmethod
    def __or__(self, other: MatrixLike[Any, NRowsT_co, NColsT_co]) -> MatrixLike[bool, NRowsT_co, NColsT_co]: ...
    @abstractmethod
    def __xor__(self, other: MatrixLike[Any, NRowsT_co, NColsT_co]) -> MatrixLike[bool, NRowsT_co, NColsT_co]: ...
    @abstractmethod
    def __neg__(self: MatrixLike[SupportsNeg[T], NRowsT_co, NColsT_co]) -> MatrixLike[T, NRowsT_co, NColsT_co]: ...
    @abstractmethod
    def __pos__(self: MatrixLike[SupportsPos[T], NRowsT_co, NColsT_co]) -> MatrixLike[T, NRowsT_co, NColsT_co]: ...
    @abstractmethod
    def __abs__(self: MatrixLike[SupportsAbs[T], NRowsT_co, NColsT_co]) -> MatrixLike[T, NRowsT_co, NColsT_co]: ...
    @abstractmethod
    def __invert__(self) -> MatrixLike[bool, NRowsT_co, NColsT_co]: ...

    @property
    @abstractmethod
    def shape(self) -> ShapeLike[NRowsT_co, NColsT_co]: ...
    @property
    def nrows(self) -> NRowsT_co: ...
    @property
    def ncols(self) -> NColsT_co: ...
    @property
    def size(self) -> int: ...

    @abstractmethod
    def eq(self, other: MatrixLike[Any, NRowsT_co, NColsT_co]) -> MatrixLike[bool, NRowsT_co, NColsT_co]: ...
    @abstractmethod
    def ne(self, other: MatrixLike[Any, NRowsT_co, NColsT_co]) -> MatrixLike[bool, NRowsT_co, NColsT_co]: ...
    @abstractmethod
    def lt(self, other: MatrixLike[DTypeT_co, NRowsT_co, NColsT_co]) -> MatrixLike[bool, NRowsT_co, NColsT_co]: ...
    @abstractmethod
    def le(self, other: MatrixLike[DTypeT_co, NRowsT_co, NColsT_co]) -> MatrixLike[bool, NRowsT_co, NColsT_co]: ...
    @abstractmethod
    def gt(self, other: MatrixLike[DTypeT_co, NRowsT_co, NColsT_co]) -> MatrixLike[bool, NRowsT_co, NColsT_co]: ...
    @abstractmethod
    def ge(self, other: MatrixLike[DTypeT_co, NRowsT_co, NColsT_co]) -> MatrixLike[bool, NRowsT_co, NColsT_co]: ...
    @abstractmethod
    def conjugate(self: MatrixLike[SupportsConjugate[T], NRowsT_co, NColsT_co]) -> MatrixLike[T, NRowsT_co, NColsT_co]: ...
    @overload
    @abstractmethod
    def slices(self, *, by: Literal[Rule.ROW]) -> Iterator[MatrixLike[DTypeT_co, Literal[1], NColsT_co]]: ...
    @overload
    @abstractmethod
    def slices(self, *, by: Literal[Rule.COL]) -> Iterator[MatrixLike[DTypeT_co, NRowsT_co, Literal[1]]]: ...
    @overload
    @abstractmethod
    def slices(self, *, by: Rule) -> Iterator[MatrixLike[DTypeT_co, int, int]]: ...
    @overload
    @abstractmethod
    def slices(self) -> Iterator[MatrixLike[DTypeT_co, Literal[1], NColsT_co]]: ...
