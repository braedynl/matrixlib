from abc import ABCMeta, abstractmethod
from collections.abc import (Callable, Collection, Iterable, Iterator,
                             Sequence, Sized)
from typing import (Any, Literal, Protocol, SupportsIndex, TypeVar, Union,
                    overload, runtime_checkable)

from .rule import Rule

__all__ = [
    "Shaped",
    "ShapedIterable",
    "ShapedCollection",
    "ShapedSequence",
    "MatrixLike",
    "ComplexMatrixLike",
    "RealMatrixLike",
    "IntegralMatrixLike",
    "check_friendly",
]

T_co = TypeVar("T_co", covariant=True)

M_co = TypeVar("M_co", bound=int, covariant=True)
N_co = TypeVar("N_co", bound=int, covariant=True)
P_co = TypeVar("P_co", bound=int, covariant=True)

ComplexT_co = TypeVar("ComplexT_co", covariant=True, bound=complex)
RealT_co = TypeVar("RealT_co", covariant=True, bound=float)
IntegralT_co = TypeVar("IntegralT_co", covariant=True, bound=int)


@runtime_checkable
class Shaped(Sized, Protocol[M_co, N_co]):

    def __len__(self) -> int: ...

    @property
    @abstractmethod
    def shape(self) -> tuple[M_co, N_co]: ...


@runtime_checkable
class ShapedIterable(Shaped[M_co, N_co], Iterable[T_co], Protocol[T_co, M_co, N_co]):
    ...


@runtime_checkable
class ShapedCollection(ShapedIterable[T_co, M_co, N_co], Collection[T_co], Protocol[T_co, M_co, N_co]):

    def __contains__(self, value: Any) -> bool: ...


class ShapedSequence(ShapedCollection[T_co, M_co, N_co], Sequence[T_co], metaclass=ABCMeta):

    __slots__: tuple[()]

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> ShapedSequence[T_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> ShapedSequence[T_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> ShapedSequence[T_co, Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> ShapedSequence[T_co, Any, Any]: ...


class MatrixLike(ShapedSequence[T_co, M_co, N_co], metaclass=ABCMeta):

    __slots__: tuple[()]
    __match_args__: tuple[Literal["array"], Literal["shape"]]

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> MatrixLike[T_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> MatrixLike[T_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> MatrixLike[T_co, Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> MatrixLike[T_co, Any, Any]: ...

    @property
    @abstractmethod
    def array(self) -> Sequence[T_co]: ...

    @property
    def nrows(self) -> M_co: ...
    @property
    def ncols(self) -> N_co: ...
    @property
    def size(self) -> int: ...

    @abstractmethod
    def equal(self, other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @abstractmethod
    def not_equal(self, other: MatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @abstractmethod
    def transpose(self) -> MatrixLike[T_co, N_co, M_co]: ...
    @abstractmethod
    def flip(self, *, by: Rule = Rule.ROW) -> MatrixLike[T_co, M_co, N_co]: ...
    @abstractmethod
    def reverse(self) -> MatrixLike[T_co, M_co, N_co]: ...

    @overload
    def n(self, by: Literal[Rule.ROW]) -> M_co: ...
    @overload
    def n(self, by: Literal[Rule.COL]) -> N_co: ...
    @overload
    def n(self, by: Rule) -> Union[M_co, N_co]: ...
    def values(self, *, by: Rule = Rule.ROW, reverse: bool = False) -> Iterator[T_co]: ...
    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[MatrixLike[T_co, Literal[1], N_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[MatrixLike[T_co, M_co, Literal[1]]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[MatrixLike[T_co, Any, Any]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[MatrixLike[T_co, Literal[1], N_co]]: ...

    def _resolve_vector_index(self, key: SupportsIndex) -> int: ...
    def _resolve_matrix_index(self, key: SupportsIndex, *, by: Rule = Rule.ROW) -> int: ...
    def _resolve_vector_slice(self, key: slice) -> Iterable[int]: ...
    def _resolve_matrix_slice(self, key: slice, *, by: Rule = Rule.ROW) -> Iterable[int]: ...


class ComplexMatrixLike(MatrixLike[ComplexT_co, M_co, N_co], metaclass=ABCMeta):

    __slots__: tuple[()]

    FRIENDLY_TYPES: tuple[type[ComplexMatrixLike], type[RealMatrixLike], type[IntegralMatrixLike]]

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> ComplexT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> ComplexMatrixLike[ComplexT_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> ComplexT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> ComplexMatrixLike[ComplexT_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> ComplexMatrixLike[ComplexT_co, Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> ComplexMatrixLike[ComplexT_co, Any, Any]: ...

    @overload
    @abstractmethod
    def __add__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __add__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __add__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __sub__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __sub__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __sub__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mul__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mul__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mul__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __matmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, N_co, P_co]) -> ComplexMatrixLike[complex, M_co, P_co]: ...
    @overload
    @abstractmethod
    def __matmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, N_co, P_co]) -> ComplexMatrixLike[complex, M_co, P_co]: ...
    @overload
    @abstractmethod
    def __matmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, N_co, P_co]) -> ComplexMatrixLike[complex, M_co, P_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __radd__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __radd__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __radd__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rsub__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rsub__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rsub__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmatmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, P_co, M_co]) -> ComplexMatrixLike[complex, P_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmatmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, P_co, M_co]) -> ComplexMatrixLike[complex, P_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmatmul__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, P_co, M_co]) -> ComplexMatrixLike[complex, P_co, N_co]: ...
    @overload
    @abstractmethod
    def __rtruediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rtruediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rtruediv__(self: ComplexMatrixLike[complex, M_co, N_co], other: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @abstractmethod
    def __neg__(self: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...
    @abstractmethod
    def __abs__(self: ComplexMatrixLike[complex, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    def __pos__(self: ComplexMatrixLike[complex, M_co, N_co]) -> ComplexMatrixLike[complex, M_co, N_co]: ...

    @abstractmethod
    def transpose(self) -> ComplexMatrixLike[ComplexT_co, N_co, M_co]: ...
    @abstractmethod
    def flip(self, *, by: Rule = Rule.ROW) -> ComplexMatrixLike[ComplexT_co, M_co, N_co]: ...
    @abstractmethod
    def reverse(self) -> ComplexMatrixLike[ComplexT_co, M_co, N_co]: ...

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[ComplexMatrixLike[ComplexT_co, Literal[1], N_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[ComplexMatrixLike[ComplexT_co, M_co, Literal[1]]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[ComplexMatrixLike[ComplexT_co, Any, Any]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[ComplexMatrixLike[ComplexT_co, Literal[1], N_co]]: ...

    @abstractmethod
    def conjugate(self) -> ComplexMatrixLike[ComplexT_co, M_co, N_co]: ...

    def transjugate(self) -> ComplexMatrixLike[ComplexT_co, N_co, M_co]: ...


class RealMatrixLike(MatrixLike[RealT_co, M_co, N_co], metaclass=ABCMeta):

    __slots__: tuple[()]

    FRIENDLY_TYPES: tuple[type[RealMatrixLike], type[IntegralMatrixLike]]

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> RealT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> RealMatrixLike[RealT_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> RealT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> RealMatrixLike[RealT_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> RealMatrixLike[RealT_co, Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> RealMatrixLike[RealT_co, Any, Any]: ...

    @abstractmethod
    def __lt__(self, other: Union[RealMatrixLike, IntegralMatrixLike]) -> bool: ...
    @abstractmethod
    def __le__(self, other: Union[RealMatrixLike, IntegralMatrixLike]) -> bool: ...
    @abstractmethod
    def __gt__(self, other: Union[RealMatrixLike, IntegralMatrixLike]) -> bool: ...
    @abstractmethod
    def __ge__(self, other: Union[RealMatrixLike, IntegralMatrixLike]) -> bool: ...
    @overload
    @abstractmethod
    def __add__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __add__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __sub__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __sub__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mul__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mul__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __matmul__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, N_co, P_co]) -> RealMatrixLike[float, M_co, P_co]: ...
    @overload
    @abstractmethod
    def __matmul__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, N_co, P_co]) -> RealMatrixLike[float, M_co, P_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __truediv__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __floordiv__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __floordiv__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mod__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __mod__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __divmod__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> MatrixLike[tuple[float, float], M_co, N_co]: ...
    @overload
    @abstractmethod
    def __divmod__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> MatrixLike[tuple[float, float], M_co, N_co]: ...
    @overload
    @abstractmethod
    def __radd__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __radd__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rsub__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rsub__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmul__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmul__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmatmul__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, P_co, M_co]) -> RealMatrixLike[float, P_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmatmul__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, P_co, M_co]) -> RealMatrixLike[float, P_co, N_co]: ...
    @overload
    @abstractmethod
    def __rtruediv__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rtruediv__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rfloordiv__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rfloordiv__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmod__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rmod__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rdivmod__(self: RealMatrixLike[float, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> MatrixLike[tuple[float, float], M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rdivmod__(self: RealMatrixLike[float, M_co, N_co], other: RealMatrixLike[float, M_co, N_co]) -> MatrixLike[tuple[float, float], M_co, N_co]: ...
    @abstractmethod
    def __neg__(self: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @abstractmethod
    def __abs__(self: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    def __pos__(self: RealMatrixLike[float, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...

    @abstractmethod
    def transpose(self) -> RealMatrixLike[RealT_co, N_co, M_co]: ...
    @abstractmethod
    def flip(self, *, by: Rule = Rule.ROW) -> RealMatrixLike[RealT_co, M_co, N_co]: ...
    @abstractmethod
    def reverse(self) -> RealMatrixLike[RealT_co, M_co, N_co]: ...

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[RealMatrixLike[RealT_co, Literal[1], N_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[RealMatrixLike[RealT_co, M_co, Literal[1]]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[RealMatrixLike[RealT_co, Any, Any]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[RealMatrixLike[RealT_co, Literal[1], N_co]]: ...

    @overload
    @abstractmethod
    def lesser(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def lesser(self, other: RealMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def lesser_equal(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def lesser_equal(self, other: RealMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def greater(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def greater(self, other: RealMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def greater_equal(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def greater_equal(self, other: RealMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...

    def conjugate(self) -> RealMatrixLike[RealT_co, M_co, N_co]: ...
    def transjugate(self) -> RealMatrixLike[RealT_co, N_co, M_co]: ...


class IntegralMatrixLike(MatrixLike[IntegralT_co, M_co, N_co], metaclass=ABCMeta):

    __slots__: tuple[()]

    FRIENDLY_TYPES: tuple[type[IntegralMatrixLike]]

    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> IntegralT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice) -> IntegralMatrixLike[IntegralT_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> IntegralT_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, slice]) -> IntegralMatrixLike[IntegralT_co, Literal[1], Any]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, int]) -> IntegralMatrixLike[IntegralT_co, Any, Literal[1]]: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[slice, slice]) -> IntegralMatrixLike[IntegralT_co, Any, Any]: ...

    @abstractmethod
    def __lt__(self, other: IntegralMatrixLike) -> bool: ...
    @abstractmethod
    def __le__(self, other: IntegralMatrixLike) -> bool: ...
    @abstractmethod
    def __gt__(self, other: IntegralMatrixLike) -> bool: ...
    @abstractmethod
    def __ge__(self, other: IntegralMatrixLike) -> bool: ...
    @abstractmethod
    def __add__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __sub__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __mul__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __matmul__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, N_co, P_co]) -> IntegralMatrixLike[int, M_co, P_co]: ...
    @abstractmethod
    def __truediv__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @abstractmethod
    def __floordiv__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __mod__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __divmod__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> MatrixLike[tuple[int, int], M_co, N_co]: ...
    @abstractmethod
    def __lshift__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __rshift__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __and__(self: IntegralMatrixLike[bool, M_co, N_co], other: IntegralMatrixLike[bool, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __and__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __xor__(self: IntegralMatrixLike[bool, M_co, N_co], other: IntegralMatrixLike[bool, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __xor__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __or__(self: IntegralMatrixLike[bool, M_co, N_co], other: IntegralMatrixLike[bool, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __or__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __radd__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __rsub__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __rmul__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __rmatmul__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, P_co, M_co]) -> IntegralMatrixLike[int, P_co, N_co]: ...
    @abstractmethod
    def __rtruediv__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> RealMatrixLike[float, M_co, N_co]: ...
    @abstractmethod
    def __rfloordiv__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __rmod__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __rdivmod__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> MatrixLike[tuple[int, int], M_co, N_co]: ...
    @abstractmethod
    def __rlshift__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __rrshift__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rand__(self: IntegralMatrixLike[bool, M_co, N_co], other: IntegralMatrixLike[bool, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rand__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rxor__(self: IntegralMatrixLike[bool, M_co, N_co], other: IntegralMatrixLike[bool, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __rxor__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __ror__(self: IntegralMatrixLike[bool, M_co, N_co], other: IntegralMatrixLike[bool, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @overload
    @abstractmethod
    def __ror__(self: IntegralMatrixLike[int, M_co, N_co], other: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __neg__(self: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __abs__(self: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...
    @abstractmethod
    def __invert__(self: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...

    def __pos__(self: IntegralMatrixLike[int, M_co, N_co]) -> IntegralMatrixLike[int, M_co, N_co]: ...

    @abstractmethod
    def transpose(self) -> IntegralMatrixLike[IntegralT_co, N_co, M_co]: ...
    @abstractmethod
    def flip(self, *, by: Rule = Rule.ROW) -> IntegralMatrixLike[IntegralT_co, M_co, N_co]: ...
    @abstractmethod
    def reverse(self) -> IntegralMatrixLike[IntegralT_co, M_co, N_co]: ...

    @overload
    def slices(self, *, by: Literal[Rule.ROW], reverse: bool = False) -> Iterator[IntegralMatrixLike[IntegralT_co, Literal[1], N_co]]: ...
    @overload
    def slices(self, *, by: Literal[Rule.COL], reverse: bool = False) -> Iterator[IntegralMatrixLike[IntegralT_co, M_co, Literal[1]]]: ...
    @overload
    def slices(self, *, by: Rule, reverse: bool = False) -> Iterator[IntegralMatrixLike[IntegralT_co, Any, Any]]: ...
    @overload
    def slices(self, *, reverse: bool = False) -> Iterator[IntegralMatrixLike[IntegralT_co, Literal[1], N_co]]: ...

    @abstractmethod
    def lesser(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @abstractmethod
    def lesser_equal(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @abstractmethod
    def greater(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...
    @abstractmethod
    def greater_equal(self, other: IntegralMatrixLike[Any, M_co, N_co]) -> IntegralMatrixLike[bool, M_co, N_co]: ...

    def conjugate(self) -> IntegralMatrixLike[IntegralT_co, M_co, N_co]: ...
    def transjugate(self) -> IntegralMatrixLike[IntegralT_co, N_co, M_co]: ...


S = TypeVar("S")

ComplexMatrixLikeT = TypeVar("ComplexMatrixLikeT", bound=ComplexMatrixLike)
ComplexMatrixLikeFriendT = TypeVar("ComplexMatrixLikeFriendT", bound=Union[ComplexMatrixLike, RealMatrixLike, IntegralMatrixLike])

RealMatrixLikeT = TypeVar("RealMatrixLikeT", bound=RealMatrixLike)
RealMatrixLikeFriendT = TypeVar("RealMatrixLikeFriendT", bound=Union[RealMatrixLike, IntegralMatrixLike])

IntegralMatrixLikeT = TypeVar("IntegralMatrixLikeT", bound=IntegralMatrixLike)
IntegralMatrixLikeFriendT = TypeVar("IntegralMatrixLikeFriendT", bound=IntegralMatrixLike)


@overload
def check_friendly(
    method: Callable[[ComplexMatrixLikeT, ComplexMatrixLikeFriendT], S],
    /,
) -> Callable[[ComplexMatrixLikeT, ComplexMatrixLikeFriendT], S]: ...
@overload
def check_friendly(
    method: Callable[[RealMatrixLikeT, RealMatrixLikeFriendT], S],
    /,
) -> Callable[[RealMatrixLikeT, RealMatrixLikeFriendT], S]: ...
@overload
def check_friendly(
    method: Callable[[IntegralMatrixLikeT, IntegralMatrixLikeFriendT], S],
    /,
) -> Callable[[IntegralMatrixLikeT, IntegralMatrixLikeFriendT], S]: ...
