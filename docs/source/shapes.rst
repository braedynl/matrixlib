Shapes
======

Abstract
--------

.. autoclass:: matrices.shapes.ShapeLike(Collection[Union[M_co, N_co]], Generic[M_co, N_co], metaclass=ABCMeta)

    .. automethod:: __str__() -> str

    .. automethod:: __lt__(other: ShapeLike[int, int]) -> bool

    .. automethod:: __le__(other: ShapeLike[int, int]) -> bool

    .. automethod:: __eq__(other: Any) -> bool

    .. automethod:: __ne__(other: Any) -> bool

    .. automethod:: __gt__(other: ShapeLike[int, int]) -> bool

    .. automethod:: __ge__(other: ShapeLike[int, int]) -> bool

    .. automethod:: __getitem__(key: int) -> Union[M_co, N_co]

    .. automethod:: __iter__() -> Iterator[Union[M_co, N_co]]

    .. automethod:: __reversed__() -> Iterator[Union[M_co, N_co]]

    .. automethod:: __contains__(value: Any) -> bool

    .. autoproperty:: nrows() -> M_co

    .. autoproperty:: ncols() -> N_co

    .. automethod:: compare(other: ShapeLike[int, int]) -> Literal[-1, 0, 1]

Concrete
--------

.. autoclass:: matrices.shapes.Shape(ShapeLike[M, N])

    .. automethod:: __init__(nrows: M, ncols: N) -> None

    .. automethod:: __repr__() -> str

    .. automethod:: __getitem__(key: SupportsIndex) -> Union[M, N]

    .. automethod:: __setitem__(key: SupportsIndex, value: Union[M, N]) -> None

    .. automethod:: __deepcopy__(memo: Optional[dict[int, Any]] = None) -> ShapeT

    .. automethod:: __copy__() -> ShapeT

    .. automethod:: copy() -> ShapeT
