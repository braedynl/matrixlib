.. _index:

Matrices-Py
===========

General-purpose matrices for the layman.

Implements a family of general-purpose matrix types, with comprehensive type-checking capabilities, and seamless integration with core Python services.

>>> from math import isclose
>>> from typing import Literal as L
>>>
>>> from matrices import RealMatrix, ROW
>>> 
>>> def norm(a: RealMatrix[int, int, float]) -> float:
...     return sum(a * a) ** 0.5
... 
>>> a = RealMatrix[L[3], L[3], float]([
...     1, 2, 3,
...     4, 5, 6,
...     7, 8, 9,
... ], shape=(3, 3))
>>> 
>>> b = RealMatrix[L[3], L[3], float].from_nesting(
...     row / norm(row)
...     for row in a.slices(by=ROW)
... )
>>>
>>> print(b)
| 0.26726… 0.53452… 0.80178… |
| 0.45584… 0.56980… 0.68376… |
| 0.50257… 0.57436… 0.64616… |
(3 × 3)
>>>
>>> assert all(isclose(norm(row), 1) for row in b.slices(by=ROW))

Getting Started
---------------

This project is available through `pip <https://pip.pypa.io/en/stable/>`_ (requires Python 3.9 or later, 3.11 recommended):

.. code-block:: console

    $ pip install matrices-py

.. warning::

    Matrices-Py is currently in its infancy, and may see future changes that are not always backwards compatible.

    The current iteration of this library is in **beta**. Further testing is being conducted at the moment.

Contributing
------------

This project is currently maintained by `Braedyn L <https://github.com/braedynl>`_. Feel free to report bugs or make a pull request through this repository.

License
-------

Distributed under the MIT license. See the `LICENSE <https://github.com/braedynl/matrices-py/blob/main/LICENSE>`_ file for more details.

.. toctree::
    :hidden:
    :caption: general

    faq

.. toctree::
    :hidden:
    :caption: api reference

    api-builtins
    api-rules

.. toctree::
    :hidden:
    :caption: user guide

    guide-prelude
    guide-construction
    guide-rules
    guide-access
    guide-typing
    guide-material-state

.. toctree::
    :hidden:
    :caption: meta

    changelog
