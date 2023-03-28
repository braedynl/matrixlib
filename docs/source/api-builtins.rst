.. _api-builtins:

Built-ins
=========

.. automodule:: matrices.builtins

    .. autoclass:: Matrix
        :special-members: __init__, __repr__, __str__, __eq__, __hash__, __len__, __getitem__, __iter__, __reversed__, __contains__, __deepcopy__, __copy__
        :members: from_nesting, array, shape, nrows, ncols, index, count, transpose, flip, rotate, reverse, materialize, n, values, slices, format, equal, not_equal

    .. autoclass:: ComplexMatrix
        :special-members: __getitem__, __add__, __sub__, __mul__, __matmul__, __truediv__, __radd__, __rsub__, __rmul__, __rtruediv__, __neg__, __pos__, __abs__
        :members: transpose, flip, rotate, reverse, materialize, slices, conjugate, transjugate

    .. autoclass:: RealMatrix
        :special-members: __lt__, __le__, __gt__, __ge__, __getitem__, __add__, __sub__, __mul__, __matmul__, __truediv__, __floordiv__, __mod__, __divmod__, __radd__, __rsub__, __rmul__, __rtruediv__, __rfloordiv__, __rmod__, __rdivmod__, __neg__, __pos__
        :members: transpose, flip, rotate, reverse, materialize, slices, conjugate, transjugate, compare, lesser, lesser_equal, greater, greater_equal

    .. autoclass:: IntegerMatrix
        :special-members: __getitem__, __add__, __sub__, __mul__, __matmul__, __floordiv__, __mod__, __divmod__, __lshift__, __rshift__, __and__, __xor__, __or__, __radd__, __rsub__, __rmul__, __rfloordiv__, __rmod__, __rdivmod__, __rlshift__, __rrshift__, __rand__, __rxor__, __ror__, __neg__, __pos__, __abs__, __invert__
        :members: transpose, flip, rotate, reverse, materialize, slices, conjugate, transjugate
