FAQ
===

The documentation talks about vectors a lot, but where are the vectors?
-----------------------------------------------------------------------

This library and its documentation thinks of vectors as simply being a certain kind of matrix. A "row vector" is a matrix of shape :math:`1 \times N`, while a "column vector" is a matrix of shape :math:`M \times 1`.

You'll find that the shape, :math:`1 \times N`, is a common fallback or requirement for some methods due to the internal row-major ordering of elements within the ``FrozenMatrix`` class. Some documentation will refer to this as being a row vector, or a matrix of said shape.

How can I create a matrix with more dimensions (e.g., a tensor)?
----------------------------------------------------------------

You could simulate higher-dimensional shapes by creating a matrix-of-matrices, but doing so is discouraged. If you truly require higher-dimensional shapes, then `NumPy <https://numpy.org/>`_ should be your choice of tooling.

Some data storage requirements may be more appropriately configured as a matrix of tuples. An RGB image, for example, could be modeled in this way (and may be more readable as a result):

.. code-block:: python

    from matrices import FrozenMatrix
    from typing import NamedTuple, Literal as L

    class Pixel(NamedTuple):
        r: int
        g: int
        b: int

    img = FrozenMatrix[Pixel, L[5], L[5]](
    [
        Pixel(r=0, g=50, b=50), ...,
        ...,                    ...,
    ],
    shape=(5, 5))

    ...
