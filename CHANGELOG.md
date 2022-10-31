# Changelog

## v0.2.0

### Major Changes

#### Class Extendability

Previously, there was one class implemented at the top-level module, `Matrix`. This class overloaded all of the arithmetic and comparison operators, but, the elements of the matrix were not necessarily compatible with these operations.

Container types should strive to be generic, while their abilities "bounded" by its elements. If the elements do not support the `+` operator, for example, I would argue that the matrix class should not provide an `__add__()` overload (since, all it would do is raise an exception).

For this reason, the library has been split into multiple matrix classes, each providing some convenience methods pertaining to the types they're restricted to contain:
- `ComplexMatrix` - holds instances of complex-like objects
- `RealMatrix` - holds instances of real-like objects
- `IntegralMatrix` - holds instances of integral-like objects

With the addition of these subclasses, a number of protocols have been added to encompass built-in, and user-created numeric types.

These classes still suffer from the problem of type-hinting interactions with operators when custom numeric types are being used. When a built-in numeric type is used with its best corresponding matrix (e.g., `ComplexMatrix[complex]`, `RealMatrix[float]`, `IntegralMatrix[int]`), most operators will appropriately infer the output matrix's type argument.

#### A Re-Thinking on Mapping Methods

There used to be three [mapping](https://en.wikipedia.org/wiki/Map_(higher-order_function)) methods provided by the `Matrix` class: `flat_map()`, `map()`, and `collapse()`.

These methods no longer exist in any new matrix type. The reason being that there are now new matrix types - making the output matrix type somewhat ambiguous.

It's now recommended to use the built-in [`map()`](https://docs.python.org/3/library/functions.html#map) function with `__iter__()` or `slices()` to perform an element or row/column-wise operation, respectively. The call to `map()` can then be wrapped by a constructor of any matrix type.

While a bit more verbose, it provides the ability to redirect the mapping results from beyond a new matrix instance (which limits what the mapping results can be, in some cases). This change also better integrates matrices with Python, in general, as providing your own map method can be a bit of an anti-pattern.

#### Alternate Constructors (New and Old!)

All new matrix types ultimately derive from one: `Matrix`. It defines four alternate constructors in the form of class methods.

What used to be `Matrix.new()` in 0.1.0 is now `Matrix.wrap()`. The functionality remains the same. An analogous class method for `Matrix.fill_like()` is now gone. Instead, use `Matrix.fill(value, *matrix.shape)`.

Previously, there used to be two alternate constructors in the form of free functions: `matrix()` and `vector()`. With the addition of class extendability, having free functions do the job of a class method makes less sense. The `matrix()` function now exists as a class method called `Matrix.infer()` - its functionality remains identical. The default constructor, `__init__()`, now supports dimension inference, making it usable as a kind of `vector()` substitute when one of the two dimensions is set to 1:

```python
a = Matrix([1, 2, 3])           # Inferred shape is 1 × 3
b = Matrix([1, 2, 3], nrows=1)  # Inferred shape is 1 × 3
c = Matrix([1, 2, 3], ncols=1)  # Inferred shape is 3 × 1

# Equivalent ways of creating matrices `b` and `c`:
b = Matrix([1, 2, 3], ncols=3)
c = Matrix([1, 2, 3], nrows=3)
```

#### A Variety of Rule Changes

The `Rule` enum, as a concept, remains entirely intact. Many of its capabilities (i.e., its methods), however, have been moved to the `Shape` class. This change likely won't affect any existing code that has been using `Rule` members purely as a way to dictate direction in certain matrix operations.

The `Rule` enum remains a derivation of the `Enum` class, but is now, more specifically, an `IntEnum`. Users no longer have to reference the `value` attribute of a `Rule` member, as the member itself can be used as an integer.

The `inverse` property remains, and the `true_name` property (that's typically used for error messages) has been renamed to `handle`. These are the only two methods implemented by the library, with the movement of its other methods to `Shape`.

### Minor Changes

#### Overall

- General changes to documentation and error messages.
- Improvements to typing throughout the library.

#### Directory

- The library has been split into a number of modules, as opposed to a single \_\_init\_\_.py module.
- Each module has a corresponding stub file. Stub files are used to provide better control over typing, as it is a constantly growing feature with each new version of Python (in other words, it has been too volatile).

#### `Shape`

- Now derives from `collections.abc.Collection`.
- The `size` property was removed so as not to confuse it for an alias of `__len__()`. It's now recommended to use `shape.nrows * shape.ncols`, or `math.prod(shape)`.

#### `Matrix`

- All methods that only accepted a `Matrix` operand have been broadened to accept a `MatrixLike`.
- `__getitem__()` and `stack()` have been narrowed to only accept `MatrixLike` operands, as opposed to their original `Sequence` operands.
- Special methods that internally performed a kind of binary mapping process (e.g., `__eq__()`, `__and__()`, `mask()`, etc.) will now raise `ValueError` if the two operands have unequal shapes. In the case of non-matrix operands, `ValueError` will be raised if the matrix has size 0.
- All operator overloads that rely on objects implementing particular behaviors with respect to those operators have been moved to sub-classes of `Matrix`, as was mentioned in the major changes. The operators that remain are `==`, `!=`, `&`, `|`, `^`, and `~`, as these operators are type-agnostic.
- Comparison operators (both within `Matrix` and its sub-classes) will now reduce their return values to `bool`. An element-wise "un-reduced" version of each comparison operator is exposed as a separate method of the same name as the corresponding operator overload, but without the surrounding underscores (e.g., `__eq__()` is the reduced version, `eq()` is the un-reduced version).
- The bitwise operators (interpreted as being logical) now narrow their matrix results to an `IntegralMatrix`.
- `__setitem__()` now accepts a type `T` object during slice assignment, in which the object is repeated for each entry of the assignment selection.
- `reshape()` now supports dimension inference, allowing one, or even both, dimensions to be omitted and interpreted automatically. This change also applies to `__init__()`, as mentioned in the major changes (`__init__()` simply calls `reshape()`, internally).

#### Miscellaneous

- The function that checks for, and returns an appropriate iterator for binary `Matrix` operations is exposed as a newly-added function, `likewise()`.
- The functions that `Matrix` internally uses to map values in its implementation for the `&`, `|`, `^`, and `~` operators are now publicly exposed as `logical_and()`, `logical_or()`, `logical_xor()`, and `logical_not()`, respectively.
- The function that `ComplexMatrix` uses to map values in its `conjugate()` method is exposed as a newly-added function, `conjugate()` (parallel to built-in `abs()`).
- The `ROW` and `COL` members of the `Rule` enum now have equivalent global constants of the same name.
- Added some test-cases for newly-added `Matrix` sub-classes and related protocols.

## v0.1.0

First official version of Matrices-Py.
