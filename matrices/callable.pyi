import sys
import typing
from collections.abc import Callable
from typing import Any, TypeVar

from .generic import GenericMatrix

T = TypeVar("T")


if sys.version_info >= (3, 10):
    from typing import ParamSpec

    P = ParamSpec("P")

    # Mypy thinks there's an error in this ParamSpec usage, but will still
    # infer everything correctly - don't know what's going on with that
    class CallableMatrix(GenericMatrix[Callable[P, T]]):  # type: ignore[misc]
        if sys.version_info >= (3, 11):
            Self = typing.Self
        else:
            Self = TypeVar("Self", bound="CallableMatrix")

        def __call__(self: Self, *args: P.args, **kwargs: P.kwargs) -> GenericMatrix[T]: ...

else:

    class CallableMatrix(GenericMatrix[Callable[..., T]]):
        Self = TypeVar("Self", bound="CallableMatrix")

        def __call__(self: Self, *args: Any, **kwargs: Any) -> GenericMatrix[T]: ...
