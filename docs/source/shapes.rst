Shapes
======

.. autoclass:: matrices.shapes.ShapeLike(Collection[Union[M_co, N_co]], Generic[M_co, N_co], metaclass=ABCMeta)

    .. automethod:: __str__(self) -> str

    .. automethod:: __lt__(self, other: ShapeLike[int, int]) -> bool

    .. automethod:: __le__(self, other: ShapeLike[int, int]) -> bool

    .. automethod:: __eq__(self, other: Any) -> bool

    .. automethod:: __ne__(self, other: Any) -> bool

    .. automethod:: __gt__(self, other: ShapeLike[int, int]) -> bool

    .. automethod:: __ge__(self, other: ShapeLike[int, int]) -> bool

    .. automethod:: __len__(self) -> Literal[2]

    .. py:method:: __getitem__(self, key: Literal[0]) -> M_co
        :abstractmethod:
        :noindex:
    .. py:method:: __getitem__(self, key: Literal[1]) -> N_co
        :abstractmethod:
        :noindex:
    .. automethod:: __getitem__(self, key: int) -> Union[M_co, N_co]

    .. automethod:: __iter__(self) -> Iterator[Union[M_co, N_co]]

    .. automethod:: __reversed__(self) -> Iterator[Union[M_co, N_co]]

    .. automethod:: __contains__(self, value: Any) -> bool

    .. autoproperty:: nrows(self) -> M_co

    .. autoproperty:: ncols(self) -> N_co

    .. automethod:: compare(self, other: ShapeLike[int, int]) -> Literal[-1, 0, 1]

.. autoclass:: matrices.shapes.Shape(ShapeLike[M, N])

    .. automethod:: __init__(self, nrows: M, ncols: N) -> None

    .. automethod:: __repr__(self) -> str

    .. py:method:: __getitem__(self, key: Literal[0]) -> M
        :noindex:
    .. py:method:: __getitem__(self, key: Literal[1]) -> N
        :noindex:
    .. automethod:: __getitem__(self, key: SupportsIndex) -> Union[M, N]

    .. py:method:: __setitem__(self, key: Literal[0], value: M) -> None
        :noindex:
    .. py:method:: __setitem__(self, key: Literal[1], value: N) -> None
        :noindex:
    .. automethod:: __setitem__(self, key: SupportsIndex, value: Union[M, N]) -> None

    .. automethod:: __deepcopy__(self: ShapeT, memo: Optional[dict[int, Any]] = None) -> ShapeT

    .. automethod:: __copy__(self: ShapeT) -> ShapeT

    .. automethod:: copy(self: ShapeT) -> ShapeT
