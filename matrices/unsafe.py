from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import TypeVar

__all__ = ["SafetyWarning", "unsafe"]

CallableT = TypeVar("CallableT", bound=Callable)


class SafetyWarning(UserWarning):
    """Warning emitted upon use of an unsafe function

    Like all warnings, instances ``SafetyWarning`` can be suppressed with a
    filter, or by using the ``warnings.catch_warnings()`` context manager.
    Suppression of a ``SafetyWarning`` implies that you are aware of the unsafe
    function's behavior and/or ramifications.
    """

    __slots__ = ()


def unsafe(func: CallableT, /) -> CallableT:
    """Designate a function as being unsafe

    Returns a wrapper of ``func`` that emits a ``SafetyWarning`` on invocation,
    encouraging its user to read the documentation, and acknowledge potentially
    malicious behavior by using a filter or wrapping the function call with
    ``warnings.catch_warnings()``.
    """

    qualified_name = func.__qualname__

    @functools.wraps(func)
    def unsafe_wrapper(*args, **kwargs):
        warnings.warn(
            f"function '{qualified_name}' is unsafe; read its documentation for tips on proper usage",
            category=SafetyWarning,
        )
        return func(*args, **kwargs)

    return unsafe_wrapper  # type: ignore[return-value]
