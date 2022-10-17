# Changelog

## v0.2.0

### Major Changes

#### Enhanced Backwards Compatibility

This library now supports Python 3.9 or higher. This was previously Python 3.10 or higher due to the usage of certain typing features.

The faster you can upgrade to Python 3.11, the better, as Python will be receiving a number of [speed optimizations](https://docs.python.org/3.11/whatsnew/3.11.html#faster-cpython) in that version - optimizations that *greatly* benefit this library.

#### Class Extendability

Previously, there was one class implemented at the top-level module, `Matrix`. This class overloaded all of the arithmetic and comparison operators, but, the elements of the matrix were not necessarily compatible with these operations.

Container types should strive to be generic, while their abilities "bounded" by its elements. If the elements do not support the `+` operator, for example, I would argue that the matrix class should not provide an `__add__()` overload (since, all it would do is raise an exception).

For this reason, the library has been split into multiple matrix classes, each providing some convenience methods pertaining to the types they're restricted to contain:
- `ComplexMatrix` - holds instances of complex-like objects
- `RealMatrix` - holds instances of real-like objects
- `IntegralMatrix` - holds instances of integral-like objects

With the addition of these subclasses, a number of protocols have been added to encompass built-in, and user-created numeric types.

These classes still suffer from the problem of type-hinting interactions with operators when custom numeric types are being used. When a built-in numeric type is used with its best corresponding matrix (e.g., `ComplexMatrix[complex]`, `RealMatrix[float]`, `IntegralMatrix[int]`), most operators will appropriately infer the output matrix's type argument.

#### A Re-Thinking of Mapping Methods

There used to be three [mapping](https://en.wikipedia.org/wiki/Map_(higher-order_function)) methods provided by the `Matrix` class: `flat_map()`, `map()`, and `collapse()`.

These methods no longer exist in any new matrix type. The reason being that there are now new matrix types - making the output matrix type somewhat ambiguous.

It's now recommended to use the built-in [`map()`](https://docs.python.org/3/library/functions.html#map) function with `__iter__()` or `slices()` to perform an element or row/column-wise operation, respectively. The call to `map()` can then be wrapped by a constructor of any matrix type.

While a bit more verbose, it provides the ability to redirect the mapping results from beyond a new matrix instance (which limits what the mapping results can be, in some cases). This change also better integrates matrices with Python, in general, as providing your own map method can be a bit of an anti-pattern, given that one exists in the global scope at all times.

#### Alternate Constructors (New and Old!)

All new matrix types ultimately derive from one: `Matrix`. It defines four alternate constructors in the form of class methods.

What used to be `Matrix.new()` in v0.1.0 is now `Matrix.wrap()`. The functionality remains the same. An analogous class method for `Matrix.fill_like()` is now gone. Instead, use `Matrix.fill(value, *matrix.shape)`.

Previously, there used to be two alternate constructors in the form of free functions: `matrix()` and `vector()`. With the addition of class extendability, having free functions do the job of a class method makes less sense. The `matrix()` function now exists as a class method called `Matrix.infer()` - its functionality remains identical. No analogous class method exists for `vector()` - using the constructor (and setting either of the two dimensions to 1) is now the recommended way to construct a vector-like matrix.

A new alternate constructor, `Matrix.refer()`, has been added. This allows two matrices to share the same information. If a matrix, $B$, refers to another matrix, $A$, changes to $A$ will be reflected in $B$, and vice versa. This method can also be used as a kind of type cast, for those using type checkers:

```python
from matrices import Matrix, IntegralMatrix

a = Matrix([
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
], nrows=3, ncols=3)

# If we know that some matrix will only contain certain types of elements, we
# might want to refer it do a different matrix subclass for more capabilities.

# Copying this matrix will return a Matrix[int]. We can refer the copy to an
# IntegralMatrix to add methods that work with integral numbers

b = IntegralMatrix[int].refer(a.copy())
```

Referencing is performed in constant time ( $O(1)$ ).

#### A Variety of Rule Changes

The `Rule` enum, as a concept, remains entirely intact. Many of its capabilities (i.e., its methods), however, have been moved to the `Shape` class. This change likely won't affect any existing code that has been using `Rule` members purely as a way to dictate direction in certain matrix operations.

The `Rule` enum remains a derivation of the `Enum` class, but is now, more specifically, an `IntEnum`. Users no longer have to reference the `value` attribute of a `Rule` member, as the member itself can be used as an integer.

The `inverse` property remains, and the `true_name` property (that's typically used for error messages) has been renamed to `handle`. These are the only two methods implemented by the library, with the movement of its other methods to `Shape`. `Rule` does, however, gain many other methods from its new `int` mix-in.

### Minor Changes

#### Overall

- Minor tweaks to documentation and error messages.

#### Directory

- The library has been split into a number of modules, as opposed to a single \_\_init\_\_.py module.
- Each module has a corresponding stub file. Stub files are used to provide better control over typing, as it is a constantly growing feature with each new version of Python (in other words, it has been too volatile).

#### `Shape`

- Now derives from `collections.abc.Collection`.
- Two shapes are now considered equal if they are either element-wise equivalent, or size equivalent when both represent a vector shape. The `equals()` method has been added to allow for element-wise equivalence checks alone.
- The `serialize()` method that was moved from `Rule` to `Shape` has been renamed to `sequence()`.

#### `Matrix`

- Most methods that only accepted a `Matrix` operand have been broadened to accept a `MatrixLike`.
- `__getitem__()` and `stack()` have been narrowed to only accept `MatrixLike` operands, as opposed to their original `Sequence` operands.
- Special methods that internally performed a kind of binary mapping process (e.g., `__eq__()`, `__and__()`, etc.) have been restricted by `Shape`'s new equality definition. Two matrices must now have equivalent shapes for a mapping method to go through.
- Parallel to `Shape`, an `equals()` method was added to allow for "traditional" element-wise equivalence checks.
- Certain utilities that were used in `Matrix`'s definition (e.g., `logical_and()`, `logical_or()`, etc.) have been moved to a separate module, utilities.py.
- All operator overloads that rely on objects implementing particular behaviors with respect to those operators have been moved to subclasses of `Matrix`, as was mentioned in the major changes. The operators that remain are `==`, `!=`, `&`, `|`, `^`, and `~`, as these operators are type-agnostic.

## v0.1.0

First official version of Matrices-Py.
