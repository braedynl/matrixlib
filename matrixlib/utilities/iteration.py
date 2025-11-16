from __future__ import annotations

__all__ = [
    "interleave",
    "repeat_call",
    "optional_reversed",
]

import collections
import itertools
from collections.abc import Callable, Reversible
from typing import Iterable, Iterator


def interleave[T](iterables: Iterable[Iterable[T]], counts: Iterable[int]) -> Iterator[T]:
    """Return an iterator that, for each integer N in ``counts``, yields N
    elements from the parallel iterable of ``iterables``, repeatedly, until all
    have been exhausted.

    Raises ``ValueError`` if the length of ``iterables`` does not match the
    length of ``counts``.
    """
    sentinel = object()
    requests = collections.deque(zip(counts, map(iter, iterables), strict=True))

    while requests:
        request = requests.popleft()
        count = request[0]
        if count:
            iterator = request[1]
            for _ in range(count):
                result = next(iterator, sentinel)
                if result is sentinel:
                    break
                else:
                    yield result  # type: ignore
            else:
                requests.append(request)


def repeat_call[T](function: Callable[[], T], times: int | None = None) -> Iterator[T]:
    """Return an iterator that calls and yields the result of ``function``
    ``times`` times.

    Yields indefinitely if ``times`` is ``None``.
    """
    if times is None:
        args = itertools.repeat(())
    else:
        args = itertools.repeat((), times)
    return itertools.starmap(function, args)


def optional_reversed[T](reversible: Reversible[T], *, reverse: bool = False) -> Iterator[T]:
    """Return the iterator of an object, optionally its reverse iterator."""
    if reverse:
        return reversed(reversible)
    return iter(reversible)
