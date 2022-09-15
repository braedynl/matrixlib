import sys
from collections.abc import Callable
from typing import Any, TypeVar

from .base import BaseMatrix

T = TypeVar("T")


if sys.version_info >= (3, 10):
    from typing import ParamSpec

    P = ParamSpec("P")

    # Mypy thinks there's an error in this ParamSpec usage, but will still
    # infer everything correctly - don't know what's going on with that
    class CallableMatrix(BaseMatrix[Callable[P, T]]):  # type: ignore[misc]
        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> BaseMatrix[T]: ...

else:

    class CallableMatrix(BaseMatrix[Callable[..., T]]):
        def __call__(self, *args: Any, **kwargs: Any) -> BaseMatrix[T]: ...
