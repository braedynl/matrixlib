.. _faq:

FAQ
===

Frequently asked questions and answers.

General
-------

The documentation talks about vectors a lot, but where are the vectors?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This library and its documentation thinks of vectors as simply being a certain kind of matrix. A "row vector" is a matrix of shape :math:`1 \times N`, while a "column vector" is a matrix of shape :math:`M \times 1`.

You'll find that the shape, :math:`1 \times N`, is a common fallback used when the matrix is flattened in some manner.

The recommended way of creating a row or column vector is to pass its content and a ``Rule`` member to the constructor:

>>> from matrixlib import Matrix, ROW, COL
>>>
>>> a = Matrix((1, 2, 3), shape=ROW)
>>> print(a)
|        1        2        3 |
(1 × 3)
>>>
>>> b = Matrix((1, 2, 3), shape=COL)
>>> print(b)
|        1 |
|        2 |
|        3 |
(3 × 1)

.. seealso::

    * :ref:`Construction <guide-construction>`
    * :ref:`Rules <guide-rules>`

How can I create a matrix with more dimensions (e.g., a tensor)?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You could simulate higher ranks by creating a matrix-of-matrices, but doing so is discouraged. If you truly require more dimensions, then `NumPy <https://numpy.org/>`_ should be your choice of tooling.

How can I write/add/remove entries of a matrix?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can't - all of the built-in matrix types are immutable.

For most cases, you can mutate a ``list`` as necessary, and cast it to a ``Matrix`` instance after performing your mutations - similar to collecting ``str`` instances in a ``list`` and using ``str.join()`` to concatenate them.

How can I re-shape a matrix?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass the matrix and desired shape to any ``Matrix`` constructor:

.. code-block::

    a = Matrix[L[2], L[3], int](...)
    b = IntegerMatrix[L[3], L[2], int](a, shape=(3, 2))  # Cast and re-shape as a 3 × 2 IntegerMatrix

This operation is :math:`O(1)` if the matrix is material.

Re-shapes should generally be avoided, as the new arrangement of values can be hard to infer (this is also why no parallel to NumPy's ``ndarray.reshape()`` method exists).

What's the difference between ``__eq__()`` and ``equal()``, ``__ne__()`` and ``not_equal()``, etc.?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The special comparison methods "summarize" their results by performing a `lexicographical comparison <https://en.wikipedia.org/wiki/Lexicographic_order>`_ (done by comparing values first, shapes second), while the non-special variants simply perform an element-wise comparison, and yield all of the results as a new matrix.

The special comparisons can operate with matrices of unequal shape, while the non-special comparisons expect equal shapes. The special comparisons can (and often will) `short-circuit <https://en.wikipedia.org/wiki/Short-circuit_evaluation>`_.

Why is it taking so long to index this matrix?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you find that a matrix has slow access times, it's likely that the matrix was created from a permuting operation that constructed a *sequence view* (what we call "non-material"), rather than a *sequence* (what we call "material").

Sequence views are (usually) constructed by the following methods:

* ``transpose()``
* ``flip()``
* ``rotate()``
* ``reverse()``

Sequence views, themselves, are a type of sequence - and so a "stacking" of sequence views can occur when these operations are combined. This can preserve large amounts of memory, but comes at the cost of access time.

If you believe that the access times are insufficient for your purposes, you can sacrifice the extra memory by materializing it. There is no need to materialize manually-constructed matrices (they're *always* material), or matrices produced by standard arithmetic operations (such as ``equal()``, ``__add__()``, ``__matmul__()``, etc.). The method should note in its documentation if a sequence view is used.

.. seealso::

    * :ref:`Material State <guide-material-state>`

My matrix operation is taking too long. How do I speed it up?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most matrix operations are done about as fast as you can go in raw Python - the ``Matrix`` type uses a built-in sequence (``tuple``) to store data when it needs to. Methods like ``Matrix.__iter__()``, ``Matrix.__reversed__()``, ``Matrix.__getitem__()``, etc. are just thin wrappers of ``tuple.__iter__()``, ``tuple.__reversed__()``, ``tuple.__getitem__()``, etc. (these aren't as thin when views are created, but still pretty quick).

So, in a sense, you're really just asking how to speed up Python in general. A quick and easy way to do that is to:

* Update your Python version: Python 3.11 introduced `lazy frames <https://docs.python.org/3/whatsnew/3.11.html#cheaper-lazy-python-frames>`_, `inlined function calls <https://docs.python.org/3/whatsnew/3.11.html#inlined-python-function-calls>`_, and `specialized operations <https://docs.python.org/3/whatsnew/3.11.html#pep-659-specializing-adaptive-interpreter>`_, all of which greatly benefit this library.
* Use PyPy: the `PyPy project <https://www.pypy.org/>`_ is an alternative Python implementation that uses a `JIT compiler <https://en.wikipedia.org/wiki/Just-in-time_compilation>`_. It is much faster than the default Python implementation, even on version 3.11.

Design Decisions
----------------

Why is there no ``StringMatrix`` type?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The inclusion of some kind of ``StringMatrix`` is being considered for a future version. The ``Matrix`` type is meant to be easily sub-classed, and so it's not too hard to implement a version for yourself (you can take a look at the numeric matrices to see a typical sub-class implementation).

The current debate over its interface revolves around methods with identical names. The ``str`` type implements a ``count()`` method, for example, but ``Matrix`` also implements a ``count()`` method. Should the value type's methods exist in a sub-namespace? Should we alter the value type's method names, such that we can include them at the matrix-class level? Experimentation with the interface is being done with regard to these questions.

Why are matrices immutable?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a number of benefits you get for being immutable:

* Immutable objects are inherently `thread-safe <https://en.wikipedia.org/wiki/Thread_safety>`_.
* Copying immutable objects is usually an :math:`O(1)` operation, since they act as their own copy. This can also preserve memory, as copying immutable objects, in the way it's typically done in Python, simply gives you a new reference to the object.
* Immutable objects can be made hashable, allowing for their use as ``dict`` keys or ``set`` elements.

Mutable sequences are typically needed during *construction time*, often when you can't know the number of incoming values. In such cases, we recommend using a ``list`` to build-up a sequence that can later be "casted" to a ``Matrix``. The ``Matrix`` type provides construction routines from both one and two-dimensional sequences via ``__init__()`` and ``from_nesting()``, respectively.

Why the name "transjugate"?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a lot of similar APIs, it's common to name the transpose and conjugate transpose operations as properties ``T`` and ``H``, respectively. We deliberately chose to avoid this, as it goes against common naming conventions in Python.

Contenders for the operation name included:

* ``conjugate_transpose()``
* ``hermitian_transpose()``
* ``adjoint()``

We argued that the first two are too long, however, and the last could be confused with the `adjugate <https://en.wikipedia.org/wiki/Adjugate_matrix>`_, which sometimes goes by the name "adjoint", or "classical adjoint".

Because of said conflicts, we went with an admittedly obscure name, "transjugate", since it isn't terribly long, and better expresses its functionality over "adjoint".

Why can you not broadcast rows and columns?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Broadcasting, `particularly NumPy's concept of broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_, was knowingly left out of the API design, as we do not believe it to be an intuitive operation. We argue that it's much more readable when written out as a loop, which is made incredibly easy with the ``slices()`` method.

NumPy-style broadcasting is, however, supported with scalar values. Sub-class implementors are advised to support scalar broadcasting as well, when applicable (and practical) to an operation.

Why are the type arguments arranged as ``Matrix[M, N, T]``, rather than ``Matrix[T, M, N]``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the latter ordering of type arguments might make more sense, given the ordering of constructor arguments:

.. code-block::

    a = Matrix[int, L[2], L[3]](
        (
            1, 2, 3,   # Value types appear first...
            4, 5, 6,
        ),
        shape=(2, 3),  # while the dimensions appear second
    )

We prioritzed the potential for less writing by arranging the type arguments in a way that will be compatible with `PEP 696 <https://peps.python.org/pep-0696/>`_ (likely to be implemented in Python 3.12), which specifies that type variables can default when omitted from the type argument list. Meaning that, in the future, you'll be able to write matrices like this:

.. code-block::

    a = Matrix[L[2], L[3]](
        (
            1, 2, 3,  # T is inferred to be `int` - you need only describe the shape
            4, 5, 6,
        ),
        shape=(2, 3),
    )

The type variable used in the implementation of ``Matrix``, ``T_co``, will likely default to ``object`` when PEP 696 is implemented. This would mean:

.. code-block::

    Matrix[L[2], L[3]] == Matrix[L[2], L[3], object]

The same principle will apply to sub-classes of ``Matrix``:

.. code-block::

    ComplexMatrix[L[2], L[3]] == ComplexMatrix[L[2], L[3], complex]
    RealMatrix[L[2], L[3]]    == RealMatrix[L[2], L[3], float]
    IntegerMatrix[L[2], L[3]] == IntegerMatrix[L[2], L[3], int]

.. seealso::

    * :ref:`Typing <guide-typing>`

Why are the numeric matrices constrained to only built-in numeric types?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The acception of any numeric type (upper-bounded to their respective domain, of course) was, and continues to be a desire for the library. Unfortunately, `the numeric tower <https://docs.python.org/3/library/numbers.html>`_ does not make a lot of typing guarantees that are circulatable for use as upper bounds. Discussion of the subject matter `has long been stale <https://github.com/python/mypy/issues/2636>`_, and remains unresolved for the time being. If there is better support for user-made numeric types in the future, the numeric matrices will have their type arguments widened.
