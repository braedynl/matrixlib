# Matrices-Py

Dynamic, vectorized sequence types in ≤ 2 dimensions.

This library aims to be a "natural" extension to the core Python tool set - similar to the [`fractions`](https://docs.python.org/3/library/fractions.html) module of the standard library, but centered around matrices. Everything contained within this library is, and will always be, written in Python alone.

The intent, here, is **not** to provide a substitute for [NumPy](https://numpy.org/). While there may be similarities in certain parts of the API, NumPy should always be the choice for efficient data storage and vectorization in larger applications - especially in more than two dimensions.

```python
>>> from math import isclose
>>> from matrices import RealMatrix, RealMatrixLike, ROW
>>>
>>> def norm(a: RealMatrixLike[float]) -> float:
...     return sum(a * a) ** 0.5
...
>>> a = RealMatrix[float]([
...     1, 2, 3,
...     4, 5, 6,
...     7, 8, 9,
... ], nrows=3, ncols=3)
>>>
>>> for i, row in enumerate(a.slices(by=ROW)):
...     a[i, :] = row / norm(row)
...
>>> print(a)
| 0.26726… 0.53452… 0.80178… |
| 0.45584… 0.56980… 0.68376… |
| 0.50257… 0.57436… 0.64616… |
(3 × 3)
>>>
>>> assert all(map(lambda row: isclose(norm(row), 1), a.slices(by=ROW)))
```

## Getting Started

This project is available through [pip](https://pip.pypa.io/en/stable/) (requires Python 3.10 or higher, 3.11 is recommended):

```
pip install matrices-py
```

Documentation can be found [here](https://github.com/braedynl/matrices-py/wiki), along with some examples and recipes.

**Note that this library is in its infancy, and may see future changes that are not always backwards compatible.**

## Contributing

This project is currently maintained by [Braedyn L](https://github.com/braedynl). Feel free to report bugs or make a pull request through this repository.

Before making a feature request, please check out the [FAQ](https://github.com/braedynl/matrices-py/wiki/FAQ) that's found in the documentation. Parts of this API that differ from others are usually different for a reason, and some of those reasons may be found there.

## License

Distributed under the MIT license. See the [LICENSE](https://github.com/braedynl/matrices-py/blob/main/LICENSE) file for more details.
