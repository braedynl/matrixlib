.. _guide-construction:

Construction
============

Throughout the remainder of the user guide, you'll probably see matrices constructed like this:

>>> from matrices import Matrix
>>>
>>> a = Matrix([
...     1, 2, 3,
...     4, 5, 6,
... ], shape=(2, 3))

This uses the ``Matrix`` constructor. It takes an iterable of values (a "flattened" iterable), and the shape to interpret it as - raising ``ValueError`` if the shape's values are negative, or if the shape's product does not equal the iterable's length. Matrices are first and foremost a ``Sequence[T]`` - *not* a ``Sequence[Sequence[T]]`` - and the constructor helps to reinforce that.

There are more benefits to this type of construction style beyond the reinforcement, however. ``Matrix`` internally stores this data one-dimensionally, and having the user input a one-dimensional iterable means the constructor does not have to bother flattening it first. This style is also (in my opinion, at least) much more readable than having a constructor flatten, and infer the shape from a two-dimensional iterable, whose shape can be ambiguous at first glance.

There is another, more subtle reason why this construction style is beneficial: it additionally allows for complete, empty matrix representation. When you create a matrix-like ``list``, for example...

>>> matrix = [
...     [1, 2, 3],
...     [4, 5, 6],
...     [7, 8, 9],
... ]

you cannot represent a matrix of shape :math:`0 \times N`, where :math:`N > 0`. By forcing the user to pair a one-dimensional object with a matching shape, this edge case is accounted for:

>>> b = Matrix([], shape=(0, 3))  # len([]) == 0 * 3 == 0  ✔ OK
>>> print(b)
Empty matrix (0 × 3)
>>>
>>> c = Matrix([], shape=(3, 0))  # len([]) == 3 * 0 == 0  ✔ OK
>>> print(c)
Empty matrix (3 × 0)

While constructing a ``Matrix`` one-dimensionally is the recommended way of doing things, there does exist a method to construct matrices two-dimensionally, in cases where you don't already have a one-dimensional iterable:

>>> d = Matrix.from_nesting([
...     [1, 2, 3],
...     [4, 5, 6],
...     [7, 8, 9],
... ])

The shape is inferred in this circumstance, but bear in mind the drawbacks explained above - again, we only recommend using this method when you don't have immediate access to a one-dimensional version of the iterable.
