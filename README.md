# MatrixLib

General-purpose matrices for the layman.

Implements a family of general-purpose matrix types, with comprehensive type-checking capabilities, and seamless integration with core Python services.

```python
>>> from math import fsum, sqrt, isclose
>>> from typing import Any
>>> from typing import Literal as L
>>>
>>> from matrixlib import ROW, RealMatrix, IntegerMatrix
>>>
>>> def norm(a: RealMatrix[Any, Any, float]) -> float:
...     return sqrt(fsum(a * a))
...
>>> a = IntegerMatrix[L[3], L[3], int](
...     [
...         1, 2, 3,
...         4, 5, 6,
...         7, 8, 9,
...     ],
...     shape=(3, 3),
... )
>>>
>>> b = RealMatrix[L[3], L[3], float](
...     (
...         val
...         for row in a.slices(by=ROW)
...         for val in row / norm(row)
...     ),
...     shape=a.shape,
... )
>>>
>>> print(b)
| 0.26726… 0.53452… 0.80178… |
| 0.45584… 0.56980… 0.68376… |
| 0.50257… 0.57436… 0.64616… |
(3 × 3)
>>>
>>> assert all(isclose(norm(row), 1) for row in b.slices(by=ROW))
```

## Getting Started

This project is available through [pip](https://pip.pypa.io/en/stable/) (requires Python 3.9 or later, 3.11 recommended):

```
pip install matrixlib
```

**Warning**:  MatrixLib is currently in its infancy, and may see future changes that are not always backwards compatible.

The current iteration of this library is in **beta**. Further testing is being conducted at the moment.

## Contributing

This project is currently maintained by [Braedyn L](https://github.com/braedynl). Feel free to report bugs or make a pull request through this repository.

## License

Distributed under the MIT license. See the [LICENSE](LICENSE) file for more details.
