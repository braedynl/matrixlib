.. _guide-material-state:

Material State
==============

The last subject we'll cover in this guide is in regard to some behind-the-scenes details of the ``Matrix`` type that can have an impact on more complex uses of the API. Before we talk about these potential impacts, we must explain how the ``Matrix`` type is implemented.

In essence, the ``Matrix`` type is really just a wrapper of another type, called a ``Mesh``. ``Mesh`` is an `abstract base class <https://docs.python.org/3/library/abc.html>`_ that provides a layout for what we deem the "core" operations of the hybrid one and two-rank sequence interface. Matrices *do not know* what kind of ``Mesh`` they're holding - they only know that it's *some* object that implements the ``Mesh`` interface.

This `compositional relationship <https://en.wikipedia.org/wiki/Object_composition>`_ allows us to "change" the memory layout of some ``Matrix`` instances, which is why you've probably not seen much discussion of the class' memory layout until now - it depends on a lot internal conditions, and it very much needs its own page to fully describe.

In short:

    We consider a ``Matrix`` instance to be **material** if its mesh stores a non-mesh sequence. We consider a ``Matrix`` instance to be **non-material** if its mesh stores a reference to another mesh.

In the current implementation, you are guaranteed a *material* ``Matrix`` in the following circumstances:

* Construction from an iterable-shape pairing, or from ``from_nesting()``
* All forms of slicing (though this may change in a future version)
* Both unary, and binary arithmetic and comparison operations, such as ``Matrix.equal()``, ``ComplexMatrix.__add__()``, ``RealMatrix.__mod__()``, ``IntegerMatrix.__and__()``, etc.

You *may* receive a *non-material* ``Matrix`` in the following circumstances:

* Construction from another ``Matrix`` instance
* Use of ``transpose()``, ``flip()``, ``rotate()``, and ``reverse()`` (known as the "permuting methods")

The ``transpose()`` operation, and those like it, internally create a *mesh view* onto the operating ``Matrix`` instance's mesh - the operation itself is :math:`O(1)` because of this - in a sense, you pay the cost of transposition in parts when you access it, you *do not* pay the full cost at the moment of creation. When you combine these methods, the mesh views can *stack*.

.. code-block::

    from matrixlib import IntegerMatrix

    a = IntegerMatrix(
        [
            1, 2, 3,
            4, 5, 6,
        ],
        shape=(2, 3),
    )

    b = a.transpose()

    c = b.flip()

The above sequence of permuting operations is internally creating a chain of meshes, crudely shown by this model:

.. code-block::

    +-------a-------+              +-----------+
    |               |              |           |
    | IntegerMatrix +------------->|   Mesh    |
    |               |              |           |
    +-------+-------+              +-----------+
                                         ^
                                         |
                                         |
    +-------b-------+              +-----+-----+
    |               |              |           |
    | IntegerMatrix +------------->| Mesh View |
    |               |              |           |
    +-------+-------+              +-----------+
                                         ^
                                         |
                                         |
    +-------c-------+              +-----+-----+
    |               |              |           |
    | IntegerMatrix +------------->| Mesh View |
    |               |              |           |
    +---------------+              +-----------+

The matrix, ``a``, stores a ``Mesh`` that holds its elements as an actual in-memory sequence - it has the fastest possible access times. The matrix, ``b``, is a transposition of ``a``, and creates a ``Mesh`` view on ``a``'s ``Mesh`` - it has slower access times than ``a``. The matrix, ``c``, is a row-flip of ``b``, and creates a ``Mesh`` view on ``b``'s ``Mesh`` view - it has slower access times than ``b``.

In general, a matrix's access time is proportional to the number of permutations that proceeded its creation - the more permutations, the slower its access times will be.

So why do this? Well, Python objects are *big* - much bigger than objects in other languages. Views are a way to avoid allocation of potentially massive ``Matrix`` objects at the cost of time. We believe that the common case is just to apply one permutation, and in such scenarios, you're not sacrificing *that* much time for huge memory savings.

There are some circumstances where you *do* want to sacrifice memory for the sake of time, however. If you find yourself using a permuted matrix in a lot of different places, you may want to **materialize** it for better speeds. You can accomplish this via the ``materialize()`` method:

.. code-block::

    from matrixlib import IntegerMatrix

    a = IntegerMatrix(
        [
            1, 2, 3,
            4, 5, 6,
        ],
        shape=(2, 3),
    )

    b = a.transpose()

    c = b.flip().materialize()

The above example now produces a chain alike the following:

.. code-block::

    +-------a-------+              +-----------+
    |               |              |           |
    | IntegerMatrix +------------->|   Mesh    |
    |               |              |           |
    +-------+-------+              +-----------+
                                         ^
                                         |
                                         |
    +-------b-------+              +-----+-----+
    |               |              |           |
    | IntegerMatrix +------------->| Mesh View |
    |               |              |           |
    +-------+-------+              +-----------+



    +-------c-------+              +-----------+
    |               |              |           |
    | IntegerMatrix +------------->|   Mesh    |
    |               |              |           |
    +---------------+              +-----------+

The matrix, ``c``, now has its own in-memory sequence stored within its mesh - it has access times identical to that of a ``Matrix`` created from an iterable-shape pair.

Materialization does *not* affect material matrices. A new ``Matrix`` instance with a reference to the same mesh is constructed by the current implementation.
