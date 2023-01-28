from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import (Any, Generic, Literal, Optional, SupportsIndex, TypeVar,
                    overload)

from .shapes import ShapeLike
from .typeshed import (SupportsAbs, SupportsAdd, SupportsAnd,
                       SupportsClosedAdd, SupportsConjugate, SupportsDivMod,
                       SupportsDotProduct, SupportsFloorDiv, SupportsInvert,
                       SupportsLShift, SupportsMod, SupportsMul, SupportsNeg,
                       SupportsOr, SupportsPos, SupportsPow, SupportsRAdd,
                       SupportsRAnd, SupportsRDivMod, SupportsRDotProduct,
                       SupportsRFloorDiv, SupportsRLShift, SupportsRMod,
                       SupportsRMul, SupportsROr, SupportsRPow,
                       SupportsRRShift, SupportsRShift, SupportsRSub,
                       SupportsRTrueDiv, SupportsRXor, SupportsSub,
                       SupportsTrueDiv, SupportsXor)
from .utilities import Rule

__all__ = ["MatrixLike"]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

T1 = TypeVar("T1")
T2 = TypeVar("T2")

M = TypeVar("M", bound=int)
M_co = TypeVar("M_co", covariant=True, bound=int)

N = TypeVar("N", bound=int)
N_co = TypeVar("N_co", covariant=True, bound=int)

P = TypeVar("P", bound=int)
P_co = TypeVar("P_co", covariant=True, bound=int)

SupportsClosedAddT = TypeVar("SupportsClosedAddT", bound=SupportsClosedAdd)


class MatrixLike(Sequence[T_co], Generic[T_co, M_co, N_co], metaclass=ABCMeta):

    def __lt__(self, other: MatrixLike[T_co, M_co, N_co]) -> bool: ...
    def __le__(self, other: MatrixLike[T_co, M_co, N_co]) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...
    def __gt__(self, other: MatrixLike[T_co, M_co, N_co]) -> bool: ...
    def __ge__(self, other: MatrixLike[T_co, M_co, N_co]) -> bool: ...
    def __len__(self) -> int: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> MatrixLike[T_co, Literal[1], int]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> MatrixLike[T_co, Literal[1], int]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> MatrixLike[T_co, int, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> MatrixLike[T_co, int, int]: ...
    def __iter__(self) -> Iterator[T_co]: ...
    def __reversed__(self) -> Iterator[T_co]: ...
    def __contains__(self, value: Any) -> bool: ...
    @overload
    @abstractmethod
    def __add__(self: MatrixLike[SupportsAdd[T1, T2], M_co, N_co], other: MatrixLike[T1, M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __add__(self: MatrixLike[T1, M_co, N_co], other: MatrixLike[SupportsRAdd[T1, T2], M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __sub__(self: MatrixLike[SupportsSub[T1, T2], M_co, N_co], other: MatrixLike[T1, M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __sub__(self: MatrixLike[T1, M_co, N_co], other: MatrixLike[SupportsRSub[T1, T2], M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mul__(self: MatrixLike[SupportsMul[T1, T2], M_co, N_co], other: MatrixLike[T1, M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mul__(self: MatrixLike[T1, M_co, N_co], other: MatrixLike[SupportsRMul[T1, T2], M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self: MatrixLike[SupportsTrueDiv[T1, T2], M_co, N_co], other: MatrixLike[T1, M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self: MatrixLike[T1, M_co, N_co], other: MatrixLike[SupportsRTrueDiv[T1, T2], M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __floordiv__(self: MatrixLike[SupportsFloorDiv[T1, T2], M_co, N_co], other: MatrixLike[T1, M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __floordiv__(self: MatrixLike[T1, M_co, N_co], other: MatrixLike[SupportsRFloorDiv[T1, T2], M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mod__(self: MatrixLike[SupportsMod[T1, T2], M_co, N_co], other: MatrixLike[T1, M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mod__(self: MatrixLike[T1, M_co, N_co], other: MatrixLike[SupportsRMod[T1, T2], M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __divmod__(self: MatrixLike[SupportsDivMod[T1, T2], M_co, N_co], other: MatrixLike[T1, M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __divmod__(self: MatrixLike[T1, M_co, N_co], other: MatrixLike[SupportsRDivMod[T1, T2], M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __pow__(self: MatrixLike[SupportsPow[T1, T2], M_co, N_co], other: MatrixLike[T1, M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __pow__(self: MatrixLike[T1, M_co, N_co], other: MatrixLike[SupportsRPow[T1, T2], M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __lshift__(self: MatrixLike[SupportsLShift[T1, T2], M_co, N_co], other: MatrixLike[T1, M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __lshift__(self: MatrixLike[T1, M_co, N_co], other: MatrixLike[SupportsRLShift[T1, T2], M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rshift__(self: MatrixLike[SupportsRShift[T1, T2], M_co, N_co], other: MatrixLike[T1, M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rshift__(self: MatrixLike[T1, M_co, N_co], other: MatrixLike[SupportsRRShift[T1, T2], M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __and__(self: MatrixLike[SupportsAnd[T1, T2], M_co, N_co], other: MatrixLike[T1, M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __and__(self: MatrixLike[T1, M_co, N_co], other: MatrixLike[SupportsRAnd[T1, T2], M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __xor__(self: MatrixLike[SupportsXor[T1, T2], M_co, N_co], other: MatrixLike[T1, M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __xor__(self: MatrixLike[T1, M_co, N_co], other: MatrixLike[SupportsRXor[T1, T2], M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __or__(self: MatrixLike[SupportsOr[T1, T2], M_co, N_co], other: MatrixLike[T1, M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __or__(self: MatrixLike[T1, M_co, N_co], other: MatrixLike[SupportsROr[T1, T2], M_co, N_co]) -> MatrixLike[T2, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __matmul__(self: MatrixLike[SupportsDotProduct[T1, SupportsClosedAddT], M_co, N_co], other: MatrixLike[T1, N_co, P_co]) -> MatrixLike[SupportsClosedAddT, M_co, P_co]: ...
    @overload
    @abstractmethod
    def __matmul__(self: MatrixLike[T1, M_co, N_co], other: MatrixLike[SupportsRDotProduct[T1, SupportsClosedAddT], N_co, P_co]) -> MatrixLike[SupportsClosedAddT, M_co, P_co]: ...
    @abstractmethod
    def __neg__(self: MatrixLike[SupportsNeg[T1], M_co, N_co]) -> MatrixLike[T1, M_co, N_co]: ...
    @abstractmethod
    def __pos__(self: MatrixLike[SupportsPos[T1], M_co, N_co]) -> MatrixLike[T1, M_co, N_co]: ...
    @abstractmethod
    def __abs__(self: MatrixLike[SupportsAbs[T1], M_co, N_co]) -> MatrixLike[T1, M_co, N_co]: ...
    @abstractmethod
    def __invert__(self: MatrixLike[SupportsInvert[T1], M_co, N_co]) -> MatrixLike[T1, M_co, N_co]: ...

    @property
    @abstractmethod
    def shape(self) -> ShapeLike[M_co, N_co]: ...
    @property
    def nrows(self) -> M_co: ...
    @property
    def ncols(self) -> N_co: ...
    @property
    def size(self) -> int: ...

    @abstractmethod
    def equal(self, other: MatrixLike[Any, M_co, N_co]) -> MatrixLike[bool, M_co, N_co]: ...
    @abstractmethod
    def not_equal(self, other: MatrixLike[Any, M_co, N_co]) -> MatrixLike[bool, M_co, N_co]: ...
    @abstractmethod
    def lesser(self, other: MatrixLike[T_co, M_co, N_co]) -> MatrixLike[bool, M_co, N_co]: ...
    @abstractmethod
    def lesser_equal(self, other: MatrixLike[T_co, M_co, N_co]) -> MatrixLike[bool, M_co, N_co]: ...
    @abstractmethod
    def greater(self, other: MatrixLike[T_co, M_co, N_co]) -> MatrixLike[bool, M_co, N_co]: ...
    @abstractmethod
    def greater_equal(self, other: MatrixLike[T_co, M_co, N_co]) -> MatrixLike[bool, M_co, N_co]: ...
    @abstractmethod
    def logical_and(self, other: MatrixLike[Any, M_co, N_co]) -> MatrixLike[bool, M_co, N_co]: ...
    @abstractmethod
    def logical_or(self, other: MatrixLike[Any, M_co, N_co]) -> MatrixLike[bool, M_co, N_co]: ...
    @abstractmethod
    def logical_not(self) -> MatrixLike[bool, M_co, N_co]: ...
    @abstractmethod
    def conjugate(self: MatrixLike[SupportsConjugate[T1], M_co, N_co]) -> MatrixLike[T1, M_co, N_co]: ...
    @overload
    def slices(self, *, by: Literal[Rule.ROW]) -> Iterator[MatrixLike[T_co, Literal[1], N_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL]) -> Iterator[MatrixLike[T_co, M_co, Literal[1]]]: ...
    @overload
    def slices(self, *, by: Rule) -> Iterator[MatrixLike[T_co, int, int]]: ...
    @overload
    def slices(self) -> Iterator[MatrixLike[T_co, Literal[1], N_co]]: ...
    @abstractmethod
    def transpose(self) -> MatrixLike[T_co, N_co, M_co]: ...

    def _resolve_index(self, key: SupportsIndex, *, by: Optional[Rule] = None) -> int: ...
    def _resolve_slice(self, key: slice, *, by: Optional[Rule] = None) -> Iterable[int]: ...
    def _permute_index(self, index: tuple[int, int] | int) -> int: ...
