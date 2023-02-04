Utilities
=========

.. autoclass:: matrices.utilities.Rule(Enum)

    .. autoattribute:: ROW

    .. autoattribute:: COL

    .. py:method:: __invert__(self: Literal[Rule.ROW]) -> Literal[Rule.COL]
        :noindex:
    .. py:method:: __invert__(self: Literal[Rule.COL]) -> Literal[Rule.ROW]
        :noindex:
    .. automethod:: __invert__(self) -> Rule

    .. autoproperty:: handle(self) -> Literal["row", "column"]

.. .. py:function:: matrices.utilities.checked_map(func: Callable[[T1], T], a: MatrixLike[T1, M, N]) -> Iterator[T]
..     :noindex:
.. .. py:function:: matrices.utilities.checked_map(func: Callable[[T1, T2], T], a: MatrixLike[T1, M, N], b: MatrixLike[T2, M, N]) -> Iterator[T]
..     :noindex:
.. .. py:function:: matrices.utilities.checked_map(func: Callable[[T1, T2, T3], T], a: MatrixLike[T1, M, N], b: MatrixLike[T2, M, N], c: MatrixLike[T3, M, N]) -> Iterator[T]
..     :noindex:
.. .. py:function:: matrices.utilities.checked_map(func: Callable[[T1, T2, T3, T4], T], a: MatrixLike[T1, M, N], b: MatrixLike[T2, M, N], c: MatrixLike[T3, M, N], d: MatrixLike[T4, M, N]) -> Iterator[T]
..     :noindex:
.. .. py:function:: matrices.utilities.checked_map(func: Callable[[T1, T2, T3, T4, T5], T], a: MatrixLike[T1, M, N], b: MatrixLike[T2, M, N], c: MatrixLike[T3, M, N], d: MatrixLike[T4, M, N], e: MatrixLike[T5, M, N]) -> Iterator[T]
..     :noindex:
.. autofunction:: matrices.utilities.checked_map(func: Callable[..., T], a: MatrixLike[Any, M, N], *bx: MatrixLike[Any, M, N]) -> Iterator[T]

.. autofunction:: matrices.utilities.logical_and(a: Any, b: Any, /) -> bool

.. autofunction:: matrices.utilities.logical_or(a: Any, b: Any, /) -> bool

.. autofunction:: matrices.utilities.logical_not(a: Any, /) -> bool
