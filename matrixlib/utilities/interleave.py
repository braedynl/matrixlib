from __future__ import annotations

__all__ = ["interleave"]

import collections
from typing import Iterable, Iterator


def interleave[T](iterables: Iterable[Iterable[T]], leave_counts: Iterable[int]) -> Iterator[T]:
    """Return an iterator that, for each integer N in ``leave_counts``, yields
    N elements from the parallel iterable of ``iterables``, repeatedly, until
    all have been exhausted.
    """

    sentinel = object()
    requests = collections.deque(zip(leave_counts, map(iter, iterables), strict=True))

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
