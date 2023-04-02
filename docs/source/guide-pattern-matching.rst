.. _guide-pattern-matching:

Pattern Matching
================

A somewhat obscure feature that we support is the ability to structurally-match a ``Matrix`` instance through Python 3.10's match statement.

Because ``Matrix`` is a type of ``Sequence``, you can match an instance of one by using a sequence pattern:

.. code-block::

    from matrixlib import IntegerMatrix

    a = IntegerMatrix(
        [
            1, 2, 3,
            4, 5, 6,
        ],
        shape=(2, 3),
    )

    match a:
        case [a, b, c, 4, 5, 6]:
            print(a, b, c)  # Prints "1 2 3"

While it may look like we're matching a ``list``, the bracketed notation, in this context, denotes an instance of ``Sequence``. Thus, there is no notion of shape here - the pattern we gave would also match ``Matrix(range(1, 6 + 1), shape=(1, 6))``, for example.

To match both values *and* shape, you must use a ``Matrix`` explicitly - we provide a ``__match_args__`` variable whose fields are the instance's ``array`` and ``shape`` (in that order) for such purposes:

.. code-block::

    from matrixlib import IntegerMatrix

    a = IntegerMatrix(
        [
            1, 2, 3,
            4, 5, 6,
        ],
        shape=(2, 3),
    )

    match a:
        case IntegerMatrix(
            [
                a, b, c,
                4, 5, 6,
            ],
            shape=(2, 3),
        ):
            print(a, b, c)  # Prints "1 2 3"

Again, the syntax can be a bit misleading: when invoking a class under a ``case`` statement, you're describing a pattern for the class' ``__match_args__`` variable - *not* its constructor - though we've intentionally made its match arguments align with the constructor due to the semantic resemblance.

`PEP 636 <https://peps.python.org/pep-0636/>`_ provides more details on pattern matching, if wanting to learn more.
