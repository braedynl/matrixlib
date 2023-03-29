.. _guide-access:

Access
======

In being a matrix type, there are a few ways to access its values. This part of the API will be familiar to those who have worked with NumPy.

Again, it's important to emphasize that the ``Matrix`` type behaves like a ``Sequence[T]``. You can access its values similar to how you would for most other ``Sequence[T]`` objects:

>>> from matrixlib import Matrix
>>>
>>> a = Matrix([
...     1, 2, 3,
...     4, 5, 6,
... ], shape=(2, 3))
>>>
>>> for i in range(len(a)):
...     print(a[i])
...
1
2
3
4
5
6

Integer access (what we typically call "1D indexing") is the fastest way to retrieve values through ``__getitem__()``, as the call is almost directly passed to the underlying ``Sequence`` object, which is often a ``tuple`` (note the word "often", here - more on this later).

In many circumstances, of course, you'll probably want the values. The ``__iter__()`` and ``__reversed__()`` methods work as you would expect, given the example above:

>>> for x in a:
...     print(x)
...
1
2
3
4
5
6
>>> for x in reversed(a):
...     print(x)
...
6
5
4
3
2
1

Prefer these methods over using the ``range(len(a))`` idiom - they are often *much* faster, and require less writing. If you need to iterate over multiple matrices at once, use built-in ``zip()``.

This default access order has a name: `"row-major order" <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_. This ordering is how the ``Matrix`` type aligns its values in memory, which is the primary reason for why access routines in this manner are so quick. Other orderings are exposed as an independent method, called ``values()``:

>>> from matrixlib import ROW, COL
>>>
>>> for x in a.values(by=ROW):  # Row-major order
...     print(x)
...
1
2
3
4
5
6
>>> for x in a.values(by=COL):  # Column-major order
...     print(x)
...
1
4
2
5
3
6
>>> for x in a.values(by=ROW, reverse=True):
...     print(x)
...
6
5
4
3
2
1
>>> for x in a.values(by=COL, reverse=True):
...     print(x)
...
6
3
5
2
4
1

This method is generally slower than using ``__iter__()`` and ``__reversed__()`` - even for the row-major ordering case - as it goes through additional layers of abstraction in the current implementation.

Slicing a matrix can also be accomplished in one dimension - the returned matrix will always have the shape :math:`1 \times N` (where :math:`N` is the slice length), making it a quick and easy way to retrieve a "flattened" copy of some matrix:

>>> print(a[:])
|        1        2        3        4        5        6 |
(1 × 6)

Though the "flattening" aspect of this capability is not too useful, since all matrices can be interpreted as their flattened selves, no matter the shape.

There are many circumstances in which you'd probably want the rows or columns of the matrix - the ``__getitem__()`` method exposes additional "2D indexing" capabilities in a similar fashion to NumPy arrays:

>>> print(a[0, 1])  # Row 0, column 1
2
>>>
>>> print(a[0, :])  # Row 0, all columns
|        1        2        3 |
(1 × 3)
>>>
>>> print(a[:, 0])  # All rows, column 0
|        1 |
|        4 |
(2 × 1)
>>>
>>> print(a[:, :])  # All rows, all columns
|        1        2        3 |
|        4        5        6 |
(2 × 3)

Similar to ``values()``, there exists a ``slices()`` method that iterates through the rows or columns of the matrix, optionally in reverse, as well:

>>> for x in a.slices(by=ROW):
...     print(x)
...
|        1        2        3 |
(1 × 3)
|        4        5        6 |
(1 × 3)
>>> for x in a.slices(by=COL):
...     print(x)
...
|        1 |
|        4 |
(2 × 1)
|        2 |
|        5 |
(2 × 1)
|        3 |
|        6 |
(2 × 1)
>>> for x in a.slices(by=ROW, reverse=True):
...     print(x)
...
|        4        5        6 |
(1 × 3)
|        1        2        3 |
(1 × 3)
>>> for x in a.slices(by=COL, reverse=True):
...     print(x)
...
|        3 |
|        6 |
(2 × 1)
|        2 |
|        5 |
(2 × 1)
|        1 |
|        4 |
(2 × 1)

There are no such "fast-pathing" methods for retrieving rows and columns, like there is ``__iter__()`` and ``__reversed__()`` as fast-pathing methods to ``values()``. Using ``__getitem__()`` to manually retrieve rows or columns is roughly time-equivalent to using ``slices()``, but you should prefer ``slices()`` for better readability.
