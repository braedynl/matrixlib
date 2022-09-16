# Changelog

## v0.2.0

### Enhanced Backwards (and Forwards) Compatibility

This library now supports Python 3.9 or higher. This was previously Python 3.10 or higher due to the usage of certain typing features. Typing has been re-worked to support 3.9, and is prepared to support 3.11, as well.

The faster you can upgrade to Python 3.11, the better, as Python will be receiving a number of [speed optimizations](https://docs.python.org/3.11/whatsnew/3.11.html#faster-cpython) in that version - optimizations that *greatly* benefit this library.

### Major Refactoring

Previously, there was one class implemented at the top-level module, `Matrix`. This class overloaded all of the arithmetic and comparison operators, but, the elements of the matrix were not necessarily compatible with these operations.

Container types should generally strive to be generic, while their abilities "bounded" by its elements. If the elements do not support the `+` operator, for example, I would argue that the matrix class should not provide an `__add__()` overload (since, all it would do is raise an exception).

For this reason, the library has been split into multiple matrix classes, each providing some convenience methods pertaining to the types they're restricted to contain.

### Re-thinking of Mapping Methods

There used to be three [mapping](https://en.wikipedia.org/wiki/Map_(higher-order_function)) methods provided by the (now non-existent) `Matrix` class: `flat_map()`, `map()`, and `collapse()`.

These methods no longer exist in any new matrix type. The reason being that there are now new matrix types - making the output matrix type somewhat ambiguous.

It's now recommended to use the built-in [`map()`](https://docs.python.org/3/library/functions.html#map) function with `__iter__()` or `slices()` to perform an element or row/column-wise operation, respectively. The call to `map()` can then be wrapped by a constructor of any matrix type.

While a bit more verbose, it provides the ability to redirect the mapping results from beyond a new matrix instance (which limits what the mapping results can be, in some cases). This change also better integrates matrices with Python, in general, as providing your own map method can be a bit of an anti-pattern, given that one exists in the global scope at all times.

### Alternate Constructors (New and Old!)

All new matrix types ultimately derive from one: `GenericMatrix`. It defines four alternate constructors in the form of class methods.

What used to be `Matrix.new()` in v0.1.0 is now `GenericMatrix.wrap()`. The functionality remains the same. An analogous class method for `Matrix.fill_like()` is now gone. Instead, use `GenericMatrix.fill(value, *matrix.shape)`.

Previously, there used to be two alternate constructors in the form of free functions: `matrix()` and `vector()`. With the addition of class extendability, having free functions do the job of a class method makes less sense. The `matrix()` function now exists as a class method called `GenericMatrix.infer()` - its functionality remains mostly identical. No analogous class method exists for `vector()` - using the constructor (and setting either of the two dimensions to 1) is now the recommended way to construct a vector-like matrix.

A new alternate constructor, `GenericMatrix.refer()`, has been added. This allows two matrices to share the same information. If a matrix, $B$, refers to another matrix, $A$, changes to $A$ will be reflected in $B$, and vice versa. This method can also be used as a kind of type cast, for those using type checkers:

```python
from matrices import GenericMatrix, IntegralMatrix

a = GenericMatrix([
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
], nrows=3, ncols=3)

b = a.copy()

# __eq__() will return a GenericMatrix[Any]. If you know that the elements will
# return booleans in their __eq__() implementation, you can refer it to an
# IntegralMatrix to add methods that work with Integral types
c = IntegralMatrix[bool].refer(a == b)
```

Referencing is performed in constant time ($O(1)$).

## v0.1.0

First official version of Matrices-Py.
