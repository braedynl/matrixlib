.. _guide-typing:

Typing
======

So far, you've probably seen matrices constructed without much regard for the types they contain, or what shape they have. Python's typing system is opt-in, and we've written the guide in a manner that respects this - as, not all Python developers want to use the system.

For those that *do* use the typing system, we provide the ability to check both the matrix's values, *and shapes*, for validation by type checkers. You can check for shape descrepancies prior to execution, and even let the Python compiler *remove* shape validation at runtime, in the expectation that you've checked for correct shapes using the typing system, or some other method of deduction.

.. note::

    The typing API has only (at this time) been tested with `mypy <https://mypy.readthedocs.io/en/stable/>`_ 1.0 and later. We plan to test with other type checkers in the near future.

We typically instantiate a ``Matrix`` like this, when typing:

.. code-block::

    from typing import Literal as L

    from matrices import ROW, COL, Matrix

    a = Matrix[L[2], L[3], int]([
        1, 2, 3,
        4, 5, 6,
    ], shape=(2, 3))

    reveal_type(a)  # Revealed type is "matrices.builtins.Matrix[Literal[2], Literal[3], builtins.int]"

Things to note, here:

* The arrangement of type arguments is ``<Nrows>, <Ncols>, <ValueType>``. By convention, we refer to each type argument as ``M``, ``N``, and ``T``, respectively.
* In order to properly "fix" your ``Matrix`` to some shape, you must provide a literal ``int`` as a type argument to ``M`` and ``N``, since type checkers will not assume literals by default.
* We typically import ``typing.Literal`` as the alias, ``L``, for less writing.

You can describe functions that take matrices of specific shape, letting the type checker provide an error for you:

.. code-block::

    def f(x: Matrix[L[5], L[5], float]) -> float:
        return sum(x)

    print(f(a))  # Argument 1 to "f" has incompatible type "Matrix[Literal[2], Literal[3], int]";
                 # expected "Matrix[Literal[5], Literal[5], float]"

All methods that come along with the ``Matrix`` class are typed to be *as specific as possible, down to the literal level*. You don't often need to type yourself, as most will deduce them for you:

.. code-block::

    for row in a.slices(by=ROW):
        reveal_type(row)  # Revealed type is "matrices.builtins.Matrix[Literal[1], Literal[3], builtins.int]"

    for col in a.slices(by=COL):
        reveal_type(col)  # Revealed type is "matrices.builtins.Matrix[Literal[2], Literal[1], builtins.int]"

All vectorized operations (such as ``Matrix.equal()``, ``ComplexMatrix.__add__()``, ``RealMatrix.__mod__()``, etc.) are typed as expecting matrix arguments of equal shape. Internally, a built-in ``map`` object is created and passed to a matrix constructor, where, after unpacking the ``map`` results, the shape is expected to have non-negative values, and a product equal to the number of unpacked results. This check can be redundant when you've independently verified the shapes yourself, through use of type checkers.

In accordance with this possibility, runtime-checking of correct shapes is a debug-only operation, and can be removed entirely when compiling with `the -O flag <https://docs.python.org/3/using/cmdline.html#cmdoption-O>`_. This is much more noticeable when operating on hundreds of matrices, and not too significant otherwise:

.. code-block::

    from random import random
    from timeit import repeat
    from typing import Literal as L

    from matrices import RealMatrix

    a = RealMatrix[L[3], L[3], float]((random() for _ in range(9)), (3, 3))
    b = RealMatrix[L[3], L[3], float]((random() for _ in range(9)), (3, 3))

    def f():
        c = a + b

    print(repeat(f, globals=globals()))

    # Without -O: [4.3656925,  4.327279, 4.3131848, 4.3171339, 4.320178200000001]
    # With -O:    [4.2557397, 4.1544341, 4.1569906, 4.190821700000001, 4.1678824]

If you want to learn more about typing, `the mypy documentation provides a great introduction <https://mypy.readthedocs.io/en/stable/>`_, and is the type checker that we use the most.
