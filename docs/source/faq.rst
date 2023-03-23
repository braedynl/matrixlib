FAQ
===

The documentation talks about vectors a lot, but where are the vectors?
-----------------------------------------------------------------------

This library and its documentation thinks of vectors as simply being a certain kind of matrix. A "row vector" is a matrix of shape :math:`1 \times N`, while a "column vector" is a matrix of shape :math:`M \times 1`.

You'll find that the shape, :math:`1 \times N`, is a common fallback used when the matrix is flattened in some manner.

Why the name "transjugate"?
---------------------------

In a lot of similar APIs, it's common to name the transpose and conjugate transpose operations as properties ``T`` and ``H``, respectively. We deliberately chose to avoid this, as it goes against common naming conventions in Python.

Contenders for the operation name included:

* ``conjugate_transpose()``
* ``hermitian_transpose()``
* ``adjoint()``

We argued that the first two are too long, however, and the last could be confused with the `adjugate <https://en.wikipedia.org/wiki/Adjugate_matrix>`_, which sometimes goes by the name "adjoint", or "classical adjoint".

Due to all of this, we went with an admittedly obscure name, "transjugate", since it isn't terribly long, and better expresses its functionality over "adjoint".

How can I create a matrix with more dimensions (e.g., a tensor)?
----------------------------------------------------------------

You could simulate higher ranks by creating a matrix-of-matrices, but doing so is discouraged. If you truly require more dimensions, then `NumPy <https://numpy.org/>`_ should be your choice of tooling.

Why can you not broadcast rows and columns?
-------------------------------------------

Broadcasting, `particularly NumPy's concept of broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_, was knowingly left out of the API design, as we do not believe it to be an intuitive operation. We argue that it's much more readable when written out as a loop, which is made incredibly easy with the ``slices()`` method.

NumPy-style broadcasting is, however, supported with scalar values. Sub-class implementors are advised to support scalar broadcasting as well, when applicable to an operation.
