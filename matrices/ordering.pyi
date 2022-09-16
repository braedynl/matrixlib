import sys
import typing
from typing import Any, Protocol, TypeVar, Union

from .generic import GenericMatrix

class SupportsDunderLT(Protocol):
    def __lt__(self, other: Any) -> Any: ...

class SupportsDunderGT(Protocol):
    def __gt__(self, other: Any) -> Any: ...

class SupportsDunderLE(Protocol):
    def __le__(self, other: Any) -> Any: ...

class SupportsDunderGE(Protocol):
    def __ge__(self, other: Any) -> Any: ...

SupportsComparison = Union[SupportsDunderLT, SupportsDunderGT, SupportsDunderLE, SupportsDunderGE]
SupportsComparisonT = TypeVar("SupportsComparisonT", bound=SupportsComparison)


class OrderingMatrix(GenericMatrix[SupportsComparisonT]):
    if sys.version_info >= (3, 11):
        Self = typing.Self
    else:
        Self = TypeVar("Self", bound="OrderingMatrix")

    def __lt__(self: Self, other: GenericMatrix[Any]) -> GenericMatrix[Any]: ...
    def __gt__(self: Self, other: GenericMatrix[Any]) -> GenericMatrix[Any]: ...
    def __le__(self: Self, other: GenericMatrix[Any]) -> GenericMatrix[Any]: ...
    def __ge__(self: Self, other: GenericMatrix[Any]) -> GenericMatrix[Any]: ...
