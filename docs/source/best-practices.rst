.. _best-practices:

Best Practices
==============

Operations
----------

Your choice of matrix may not always have methods to accomplish a specific task - this is fairly common, as the library's intent is to be general purpose. Element, row, or column-wise operations can be defined as free-standing functions, or methods of a sub-class - it's entirely up to you.

For functions that take a matrix as input, we generally recommend making it shape-agnostic, such that you may use it on the matrix itself, or its rows and columns. Many standard library functions, such as ``sum()`` and ``statistics.mean()``, are inherently written this way, as they only require their argument to be iterable:

>>> from matrixlib import ROW, COL, RealMatrix
>>>
>>> a = RealMatrix(
...     [
...         1, 2, 3,
...         4, 5, 6,
...     ],
...     shape=(2, 3),
... )
>>>
>>> sum(a)  # Sum over the elements
21
>>>
>>> b = RealMatrix(  # Sum over the rows
...     (
...         sum(row)
...         for row in a.slices(by=ROW)
...     ),
...     shape=COL,
... )
>>> print(b)
|        6 |
|       15 |
(2 × 1)
>>>
>>> c = RealMatrix(  # Sum over the columns
...     (
...         sum(col)
...         for col in a.slices(by=COL)
...     ),
...     shape=ROW,
... )
>>> print(c)
|        5        7        9 |
(1 × 3)
>>>

Depending on how generically-defined the function is, you can sometimes get away with passing slices to it directly:

>>> b = sum(a.slices(by=COL))
>>> print(b)
|        6 |
|       15 |
(2 × 1)
>>>
>>> c = sum(a.slices(by=ROW))
>>> print(c)
|        5        7        9 |
(1 × 3)

Be sure that the function *expressly* supports this kind of behavior, however, as it may be an unexpected consequence of the function's implementation (and thus, may be subject to change).

For functions that take a matrix as *both* input *and* output, you must traverse one more layer of depth to pass items to the ``Matrix`` constructor (``from_nesting()`` can avoid this, but the shape is not as clearly notated at the time of construction - prefer using ``__init__()`` when possible):

>>> d = RealMatrix(
...     (
...         val
...         for row in a.slices(by=ROW)
...         for val in row.reverse()
...     ),
...     shape=a.shape,
... )
>>> print(d)
|        3        2        1 |
|        6        5        4 |
(2 × 3)
>>>
>>> e = RealMatrix.from_nesting(
...     (
...         row.reverse()
...         for row in a.slices(by=ROW)
...     ),
... )
>>> print(e)
|        3        2        1 |
|        6        5        4 |
(2 × 3)
>>>

Typing
------

WIP
