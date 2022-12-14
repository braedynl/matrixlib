from abc import abstractmethod
from collections.abc import Iterator
from typing import Any, Literal, Protocol, TypeVar, overload, runtime_checkable

from .rule import Rule

__all__ = ["ShapeLike"]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

S = TypeVar("S")
R = TypeVar("R")

NRows_co = TypeVar("NRows_co", bound=int, covariant=True)
NCols_co = TypeVar("NCols_co", bound=int, covariant=True)


class SupportsAdd(Protocol[T_contra, T_co]):
    def __add__(self, other: T_contra) -> T_co: ...
class SupportsSub(Protocol[T_contra, T_co]):
    def __sub__(self, other: T_contra) -> T_co: ...
class SupportsMul(Protocol[T_contra, T_co]):
    def __mul__(self, other: T_contra) -> T_co: ...
class SupportsTrueDiv(Protocol[T_contra, T_co]):
    def __truediv__(self, other: T_contra) -> T_co: ...
class SupportsFloorDiv(Protocol[T_contra, T_co]):
    def __floordiv__(self, other: T_contra) -> T_co: ...
class SupportsMod(Protocol[T_contra, T_co]):
    def __mod__(self, other: T_contra) -> T_co: ...
class SupportsPow(Protocol[T_contra, T_co]):
    def __pow__(self, other: T_contra) -> T_co: ...

class SupportsRAdd(Protocol[T_contra, T_co]):
    def __radd__(self, other: T_contra) -> T_co: ...
class SupportsRSub(Protocol[T_contra, T_co]):
    def __rsub__(self, other: T_contra) -> T_co: ...
class SupportsRMul(Protocol[T_contra, T_co]):
    def __rmul__(self, other: T_contra) -> T_co: ...
class SupportsRTrueDiv(Protocol[T_contra, T_co]):
    def __rtruediv__(self, other: T_contra) -> T_co: ...
class SupportsRFloorDiv(Protocol[T_contra, T_co]):
    def __rfloordiv__(self, other: T_contra) -> T_co: ...
class SupportsRMod(Protocol[T_contra, T_co]):
    def __rmod__(self, other: T_contra) -> T_co: ...
class SupportsRPow(Protocol[T_contra, T_co]):
    def __rpow__(self, other: T_contra) -> T_co: ...

class SupportsNeg(Protocol[T_co]):
    def __neg__(self) -> T_co: ...
class SupportsPos(Protocol[T_co]):
    def __pos__(self) -> T_co: ...
class SupportsAbs(Protocol[T_co]):
    def __abs__(self) -> T_co: ...
class SupportsConjugate(Protocol[T_co]):
    def conjugate(self) -> T_co: ...


@runtime_checkable
class ShapeLike(Protocol[NRows_co, NCols_co]):

    __match_args__: tuple[Literal["nrows"], Literal["ncols"]]

    def __eq__(self, other: Any) -> bool: ...
    def __len__(self) -> Literal[2]: ...
    @abstractmethod
    def __getitem__(self, key: int) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    def __reversed__(self) -> Iterator[int]: ...
    def __contains__(self, value: Any) -> bool: ...

    @property
    def nrows(self) -> NRows_co: ...
    @property
    def ncols(self) -> NCols_co: ...


# @runtime_checkable
# class MatrixLike(Protocol[T_co, Nrows, Ncols]):

#     def __eq__(self, other: Any) -> bool: ...

#     def __lt__(self, other: MatrixLike[T_co, Nrows, Ncols]) -> bool: ...
#     def __le__(self, other: MatrixLike[T_co, Nrows, Ncols]) -> bool: ...
#     def __gt__(self, other: MatrixLike[T_co, Nrows, Ncols]) -> bool: ...
#     def __ge__(self, other: MatrixLike[T_co, Nrows, Ncols]) -> bool: ...

#     def __len__(self) -> int: ...
#     @abstractmethod
#     @overload
#     def __getitem__(self, key: int) -> T_co: ...
#     @abstractmethod
#     @overload
#     def __getitem__(self, key: slice) -> MatrixLike[T_co, int, int]: ...
#     @abstractmethod
#     @overload
#     def __getitem__(self, key: tuple[int, int]) -> T_co: ...
#     @abstractmethod
#     @overload
#     def __getitem__(self, key: tuple[int, slice]) -> MatrixLike[T_co, int, int]: ...
#     @abstractmethod
#     @overload
#     def __getitem__(self, key: tuple[slice, int]) -> MatrixLike[T_co, int, int]: ...
#     @abstractmethod
#     @overload
#     def __getitem__(self, key: tuple[slice, slice]) -> MatrixLike[T_co, int, int]: ...
#     def __iter__(self) -> Iterator[T_co]: ...
#     def __reversed__(self) -> Iterator[T_co]: ...
#     def __contains__(self, value: Any) -> bool: ...

#     @abstractmethod
#     @overload
#     def __add__(
#         self: MatrixLike[SupportsAdd[S, R], Nrows, Ncols],
#         other: MatrixLike[S, Nrows, Ncols],
#     ) -> MatrixLike[R, Nrows, Ncols]: ...
#     @abstractmethod
#     @overload
#     def __add__(
#         self: MatrixLike[S, Nrows, Ncols],
#         other: MatrixLike[SupportsRAdd[S, R], Nrows, Ncols],
#     ) -> MatrixLike[R, Nrows, Ncols]: ...

#     @abstractmethod
#     def __and__(self, other: MatrixLike[Any]) -> MatrixLike[bool]: ...
#     @abstractmethod
#     def __or__(self, other: MatrixLike[Any]) -> MatrixLike[bool]: ...
#     @abstractmethod
#     def __xor__(self, other: MatrixLike[Any]) -> MatrixLike[bool]: ...

#     @abstractmethod
#     def __neg__(self: MatrixLike[SupportsNeg[R]]) -> MatrixLike[R]: ...
#     @abstractmethod
#     def __pos__(self: MatrixLike[SupportsPos[R]]) -> MatrixLike[R]: ...
#     @abstractmethod
#     def __abs__(self: MatrixLike[SupportsAbs[R]]) -> MatrixLike[R]: ...
#     @abstractmethod
#     def __invert__(self) -> MatrixLike[bool]: ...

#     @property
#     @abstractmethod
#     def shape(self) -> ShapeLike: ...
#     @property
#     def nrows(self) -> int: ...
#     @property
#     def ncols(self) -> int: ...
#     @property
#     def size(self) -> int: ...

#     @abstractmethod
#     def eq(self, other: MatrixLike[Any]) -> MatrixLike[bool]: ...
#     @abstractmethod
#     def ne(self, other: MatrixLike[Any]) -> MatrixLike[bool]: ...

#     @abstractmethod
#     def lt(self, other: MatrixLike[T_co]) -> MatrixLike[bool]: ...
#     @abstractmethod
#     def le(self, other: MatrixLike[T_co]) -> MatrixLike[bool]: ...
#     @abstractmethod
#     def gt(self, other: MatrixLike[T_co]) -> MatrixLike[bool]: ...
#     @abstractmethod
#     def ge(self, other: MatrixLike[T_co]) -> MatrixLike[bool]: ...

#     @abstractmethod
#     def conjugate(self: MatrixLike[SupportsConjugate[R]]) -> MatrixLike[R]: ...

#     @abstractmethod
#     def slices(self, *, by: Rule = Rule.ROW) -> Iterator[MatrixLike[T_co]]: ...
