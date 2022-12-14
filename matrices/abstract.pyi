from abc import ABCMeta, abstractmethod
from collections.abc import Collection, Iterator, Sequence
from typing import (Any, Generic, Literal, Protocol, TypeAlias, TypeVar,
                    overload)

from .rule import Rule

__all__ = [
    "ShapeLike",
    "MatrixLike",
    "AnyShape",
    "AnyRowVectorShape",
    "AnyColVectorShape",
    "AnyVectorShape",
]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

S = TypeVar("S")
R = TypeVar("R")

DType_co = TypeVar("DType_co", covariant=True)
NRows_co = TypeVar("NRows_co", covariant=True, bound=int)
NCols_co = TypeVar("NCols_co", covariant=True, bound=int)


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


class ShapeLike(Collection[NRows_co | NCols_co], Generic[NRows_co, NCols_co], metaclass=ABCMeta):

    __match_args__: tuple[Literal["nrows"], Literal["ncols"]]

    def __eq__(self, other: Any) -> bool: ...
    def __len__(self) -> Literal[2]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: Literal[0]) -> NRows_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: Literal[1]) -> NCols_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> NRows_co | NCols_co: ...
    def __iter__(self) -> Iterator[NRows_co | NCols_co]: ...
    def __reversed__(self) -> Iterator[NRows_co | NCols_co]: ...
    def __contains__(self, value: Any) -> bool: ...

    @property
    def nrows(self) -> NRows_co: ...
    @property
    def ncols(self) -> NCols_co: ...


AnyShape: TypeAlias = ShapeLike[int, int]
AnyRowVectorShape: TypeAlias = ShapeLike[Literal[1], int]
AnyColVectorShape: TypeAlias = ShapeLike[int, Literal[1]]
AnyVectorShape: TypeAlias = AnyRowVectorShape | AnyColVectorShape


class MatrixLike(Sequence[DType_co], Generic[DType_co, NRows_co, NCols_co], metaclass=ABCMeta):

    def __eq__(self, other: Any) -> bool: ...
    def __lt__(self, other: MatrixLike[DType_co, NRows_co, NCols_co]) -> bool: ...
    def __le__(self, other: MatrixLike[DType_co, NRows_co, NCols_co]) -> bool: ...
    def __gt__(self, other: MatrixLike[DType_co, NRows_co, NCols_co]) -> bool: ...
    def __ge__(self, other: MatrixLike[DType_co, NRows_co, NCols_co]) -> bool: ...
    def __len__(self) -> int: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> DType_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> MatrixLike[DType_co, Literal[1], int]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> DType_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> MatrixLike[DType_co, Literal[1], int]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> MatrixLike[DType_co, int, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> MatrixLike[DType_co, int, int]: ...
    def __iter__(self) -> Iterator[DType_co]: ...
    def __reversed__(self) -> Iterator[DType_co]: ...
    def __contains__(self, value: Any) -> bool: ...
    @overload
    @abstractmethod
    def __add__(
        self: MatrixLike[SupportsAdd[S, R], NRows_co, NCols_co],
        other: MatrixLike[S, NRows_co, NCols_co],
    ) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @overload
    @abstractmethod
    def __add__(
        self: MatrixLike[S, NRows_co, NCols_co],
        other: MatrixLike[SupportsRAdd[S, R], NRows_co, NCols_co],
    ) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @overload
    @abstractmethod
    def __sub__(
        self: MatrixLike[SupportsSub[S, R], NRows_co, NCols_co],
        other: MatrixLike[S, NRows_co, NCols_co],
    ) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @overload
    @abstractmethod
    def __sub__(
        self: MatrixLike[S, NRows_co, NCols_co],
        other: MatrixLike[SupportsRSub[S, R], NRows_co, NCols_co],
    ) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @overload
    @abstractmethod
    def __mul__(
        self: MatrixLike[SupportsMul[S, R], NRows_co, NCols_co],
        other: MatrixLike[S, NRows_co, NCols_co],
    ) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @overload
    @abstractmethod
    def __mul__(
        self: MatrixLike[S, NRows_co, NCols_co],
        other: MatrixLike[SupportsRMul[S, R], NRows_co, NCols_co],
    ) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @overload
    @abstractmethod
    def __truediv__(
        self: MatrixLike[SupportsTrueDiv[S, R], NRows_co, NCols_co],
        other: MatrixLike[S, NRows_co, NCols_co],
    ) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @overload
    @abstractmethod
    def __truediv__(
        self: MatrixLike[S, NRows_co, NCols_co],
        other: MatrixLike[SupportsRTrueDiv[S, R], NRows_co, NCols_co],
    ) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @overload
    @abstractmethod
    def __floordiv__(
        self: MatrixLike[SupportsFloorDiv[S, R], NRows_co, NCols_co],
        other: MatrixLike[S, NRows_co, NCols_co],
    ) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @overload
    @abstractmethod
    def __floordiv__(
        self: MatrixLike[S, NRows_co, NCols_co],
        other: MatrixLike[SupportsRFloorDiv[S, R], NRows_co, NCols_co],
    ) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @overload
    @abstractmethod
    def __mod__(
        self: MatrixLike[SupportsMod[S, R], NRows_co, NCols_co],
        other: MatrixLike[S, NRows_co, NCols_co],
    ) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @overload
    @abstractmethod
    def __mod__(
        self: MatrixLike[S, NRows_co, NCols_co],
        other: MatrixLike[SupportsRMod[S, R], NRows_co, NCols_co],
    ) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @overload
    @abstractmethod
    def __pow__(
        self: MatrixLike[SupportsPow[S, R], NRows_co, NCols_co],
        other: MatrixLike[S, NRows_co, NCols_co],
    ) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @overload
    @abstractmethod
    def __pow__(
        self: MatrixLike[S, NRows_co, NCols_co],
        other: MatrixLike[SupportsRPow[S, R], NRows_co, NCols_co],
    ) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @abstractmethod
    def __and__(self, other: MatrixLike[Any, NRows_co, NCols_co]) -> MatrixLike[bool, NRows_co, NCols_co]: ...
    @abstractmethod
    def __or__(self, other: MatrixLike[Any, NRows_co, NCols_co]) -> MatrixLike[bool, NRows_co, NCols_co]: ...
    @abstractmethod
    def __xor__(self, other: MatrixLike[Any, NRows_co, NCols_co]) -> MatrixLike[bool, NRows_co, NCols_co]: ...
    @abstractmethod
    def __neg__(self: MatrixLike[SupportsNeg[R], NRows_co, NCols_co]) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @abstractmethod
    def __pos__(self: MatrixLike[SupportsPos[R], NRows_co, NCols_co]) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @abstractmethod
    def __abs__(self: MatrixLike[SupportsAbs[R], NRows_co, NCols_co]) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @abstractmethod
    def __invert__(self) -> MatrixLike[bool, NRows_co, NCols_co]: ...

    @property
    @abstractmethod
    def shape(self) -> ShapeLike[NRows_co, NCols_co]: ...
    @property
    def nrows(self) -> NRows_co: ...
    @property
    def ncols(self) -> NCols_co: ...
    @property
    def size(self) -> int: ...

    @abstractmethod
    def eq(self, other: MatrixLike[Any, NRows_co, NCols_co]) -> MatrixLike[bool, NRows_co, NCols_co]: ...
    @abstractmethod
    def ne(self, other: MatrixLike[Any, NRows_co, NCols_co]) -> MatrixLike[bool, NRows_co, NCols_co]: ...
    @abstractmethod
    def lt(self, other: MatrixLike[DType_co, NRows_co, NCols_co]) -> MatrixLike[bool, NRows_co, NCols_co]: ...
    @abstractmethod
    def le(self, other: MatrixLike[DType_co, NRows_co, NCols_co]) -> MatrixLike[bool, NRows_co, NCols_co]: ...
    @abstractmethod
    def gt(self, other: MatrixLike[DType_co, NRows_co, NCols_co]) -> MatrixLike[bool, NRows_co, NCols_co]: ...
    @abstractmethod
    def ge(self, other: MatrixLike[DType_co, NRows_co, NCols_co]) -> MatrixLike[bool, NRows_co, NCols_co]: ...
    @abstractmethod
    def conjugate(self: MatrixLike[SupportsConjugate[R], NRows_co, NCols_co]) -> MatrixLike[R, NRows_co, NCols_co]: ...
    @overload
    @abstractmethod
    def slices(self, *, by: Literal[Rule.ROW]) -> Iterator[MatrixLike[DType_co, Literal[1], NCols_co]]: ...
    @overload
    @abstractmethod
    def slices(self, *, by: Literal[Rule.COL]) -> Iterator[MatrixLike[DType_co, NRows_co, Literal[1]]]: ...
    @overload
    @abstractmethod
    def slices(self) -> Iterator[MatrixLike[DType_co, Literal[1], NCols_co]]: ...
