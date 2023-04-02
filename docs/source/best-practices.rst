.. _best-practices:

Best Practices
==============

Keeping your matrix code clean and maintainable.

Operations
----------

Your choice of matrix may not always have methods to accomplish a specific task - this is fairly common, as the library tries its best not to over-extend on capabilities that may eventually require too much maintanence. Element, row, or column-wise operations can be defined as free-standing functions, or methods of a sub-class - it's entirely up to you, but, we tend to prefer free-standing functions so that other matrix types can be used.

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

Depending on your choice of ``Matrix`` type, and how generically-defined the function is, you can sometimes get away with passing slices to it directly:

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

For functions that take a matrix as both input *and* output, you must traverse one more layer of depth to pass items to the ``Matrix`` constructor (``from_nesting()`` can avoid this, but the shape is not as clearly notated at the time of construction):

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

Typing
------

Python's type system is a rapidly-changing area of work - we believe that ``Matrix``'s typing API is prepared for future system changes, however, and we encourage users to take advantage of it. In our opinion, it makes matrix-oriented code *much* clearer to read, because you can describe value types and shapes without ever consulting documentation.

For functions that operate on matrices of any type or shape, use a ``TypeVar``:

.. code-block::

    from typing import TypeVar

    from matrixlib import Matrix, ComplexMatrix

    T = TypeVar("T")
    M = TypeVar("M", bound=int)
    N = TypeVar("N", bound=int)

    def shape(x: Matrix[M, N, T]) -> tuple[M, N]:
        return x.shape

    a = ComplexMatrix[L[2], L[3], complex](
        (
            1+1j, 2+2j, 3+3j,
            4+4j, 5+5j, 6+6j,
        ),
        shape=(2, 3),
    )

    s = shape(a)

    reveal_type(s)  # Revealed type is "Tuple[Literal[2], Literal[3]]"

We often name our type variables ``M``, ``N``, and ``T`` to fill the number of rows, number of columns, and value type arguments. Note that ``M`` and ``N`` must be bound to instances of ``int`` - ``T`` may also need to be bounded depending on your matrix type:

* ``Matrix`` is un-bounded
* ``ComplexMatrix`` is bounded to built-in ``complex``
* ``RealMatrix`` is bounded to built-in ``float``
* ``IntegerMatrix`` is bounded to built-in ``int``

We typically name the type variable ``C``, ``R``, and ``I``, respectively, for the latter sub-classes. If you have type variables with those names already, we generally fallback to using ``ComplexT``, ``RealT``, and ``IntegerT``.

There are some situations where you may only know both dimensions of the matrix at runtime (e.g., streams of data from a network). While it's tempting to use a raw ``int`` as ``M`` or ``N``, prefer a ``NewType`` instead:

.. code-block::

    import random
    from typing import Final
    from typing import Literal as L
    from typing import NewType

    from matrixlib import ROW, RealMatrix

    UNKNOWN_SIZE: Final = random.randint(1, 50)  # An unknown size

    UnknownN = NewType("UnknownN", int)  # A type hint for the unknown size

    a = RealMatrix[L[1], UnknownN, float](
        (
            random.random()
            for _ in range(UNKNOWN_SIZE)
        ),
        shape=ROW,
    )

    b = RealMatrix[L[1], UnknownN, float](
        (
            random.random()
            for _ in range(UNKNOWN_SIZE)
        ),
        shape=ROW,
    )

    c = a + b

    reveal_type(c)  # Revealed type is "matrixlib.builtins.RealMatrix[Literal[1], main.UnknownN, builtins.float]"

This causes ``a.__add__()`` to ensure that ``b`` has ``UnknownN`` as its second dimension, similar to a literal ``int``. The same will apply to other binary functions like it.

In general, avoid using ``int`` dimensions whenever possible. Having them can let the type checker approve matrices of varying shape as arguments to equally-shaped parameters, as an example of just one problem they can cause:

.. code-block::

    from matrixlib import IntegerMatrix

    a = IntegerMatrix[int, int, int](
        (
            1, 2, 3,
            4, 5, 6,
        ),
        shape=(2, 3),
    )

    b = IntegerMatrix[int, int, int](
        (
            1, 2, 3, 4,
            5, 6, 7, 8,
        ),
        shape=(2, 4),
    )

    c = a + b

    reveal_type(c)  # Revealed type is "matrixlib.builtins.IntegerMatrix[builtins.int, builtins.int, builtins.int]"
