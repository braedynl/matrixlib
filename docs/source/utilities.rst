Utilities
=========

.. autoclass:: matrices.utilities.Rule(Enum)

    .. autoattribute:: ROW

    .. autoattribute:: COL

    .. automethod:: __invert__() -> Literal[Rule.ROW, Rule.COL]
    
    .. autoproperty:: handle() -> Literal["row", "column"]

.. autofunction:: matrices.utilities.checked_map(func: Callable[..., T], a: MatrixLike[Any, M, N], *bx: MatrixLike[Any, M, N]) -> Iterator[T]

.. autofunction:: matrices.utilities.logical_and(a: Any, b: Any, /) -> bool

.. autofunction:: matrices.utilities.logical_or(a: Any, b: Any, /) -> bool

.. autofunction:: matrices.utilities.logical_not(a: Any, /) -> bool
