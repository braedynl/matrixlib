.. _guide-rules:

Rules
=====

Before we move on to the rest of the API, we have to talk about a different class that comes with the library: the ``Rule`` class.

If you're coming from NumPy, you're probably familiar with the concept of `axes <https://numpy.org/doc/stable/glossary.html#term-axis>`_ - rules can be thought of as a parallel to axes, only we package the concept as an `enum <https://docs.python.org/3/library/enum.html#enum.Enum>`_, rather than as integers. If you're unfamiliar with axes, they're essentially just numbers that correspond to an array dimension: for a two-dimensional array, the elements of "axis 0" are the rows, and the elements of "axis 1" are the columns.

Since we're confined to two dimensions, we decided to alter the nomenclature a bit. There exists "row-rule" and "column-rule" - no more, no less.

>>> from matrixlib import ROW, COL, Matrix, Rule
>>>
>>> Rule.ROW
<Rule.ROW: 0>
>>>
>>> Rule.COL
<Rule.COL: 1>
>>>

The value of each rule member maps to an integer that accesses its dimension from a matrix shape (or any two-element sequence). A member can be "inverted" as a means to retrieve the opposite dimension in places where you don't know which dimension you're handling (particularly useful for implementing ``Matrix`` functions):

>>> ~Rule.ROW
<Rule.COL: 1>
>>>
>>> ~Rule.COL
<Rule.ROW: 0>

The ``Matrix`` type is "rule-aware" - that is, the ``Matrix`` type explicitly uses the ``Rule`` class in many parts of its definition, and expects the user to provide ``Rule`` members as arguments to some functions - usually ones that can be interpreted as being done "row-wise" or "column-wise". The ``flip()`` method is a great example of this - instead of splitting the method into two (e.g., ``flip_rows()``, ``flip_cols()``), it expects a ``Rule`` member to dictate row or column-wise interpretation:

>>> a = Matrix(
...     [
...         1, 2, 3,
...         4, 5, 6,
...     ],
...     shape=(2, 3),
... )
>>>
>>> print(a)
|        1        2        3 |
|        4        5        6 |
(2 × 3)
>>>
>>> b = a.flip(by=ROW)
>>> print(b)
|        4        5        6 |
|        1        2        3 |
(2 × 3)
>>>
>>> c = a.flip(by=COL)
>>> print(c)
|        3        2        1 |
|        6        5        4 |
(2 × 3)

There was one method of ``Matrix`` construction that we neglected to mention in the previous segment, involving rules: the ``shape`` argument accepts a ``Rule`` member as a means to interpret the ``array`` as a row or column vector:

>>> d = Matrix(range(5), shape=ROW)
>>> print(d)
|        0        1        2        3        4 |
(1 × 5)
>>>
>>> e = Matrix(range(5), shape=COL)
>>> print(e)
|        0 |
|        1 |
|        2 |
|        3 |
|        4 |
(5 × 1)

This is actually the preferred method of constructing a vector, as it does not require you to retrieve the iterable's length in some fashion.
