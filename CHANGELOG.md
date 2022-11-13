# Changelog

## v0.2.3

### Major Changes

#### Pattern Matching

`Matrix` and `Shape` were each given a [`__match_args__`](https://docs.python.org/3/reference/datamodel.html#object.__match_args__) variable that enables instances of their classes to be used in a `match` statement:

```python
match matrix:
    case Matrix(data=[  # An exact data match
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
    ], nrows=3, ncols=3): ...
    case Matrix(data=[  # A partial data match
        0, 1, 2,
        _, 4, 5,
        _, _, 8,
    ], nrows=3, ncols=3): ...
    case Matrix(nrows=3, ncols=3): ...  # An exact shape match
    case Matrix(nrows=3, ncols=n): ...  # A partial shape match (by row)
    case Matrix(nrows=m, ncols=3): ...  # A partial shape match (by column)
```

Binding the data `list` to a name inside a `case` is discouraged, since this will expose a direct reference to the matrix's internal data `list`:

```python
match matrix:
    case Matrix(data, nrows=3, ncols=3):
        data.append(5)  # `matrix` is affected here!
```

A copy of the `list` object should be taken first before operating.

#### Customizable String Formatting

`Matrix` has been given a `__format__()` method that enables users to specify how much space should be alloted for rows, columns, and items. The specification syntax is alike:

```
[nrows_max[, ncols_max[, chars_max]]]
```

All fields default to 8, meaning that 8 rows, columns, and characters will be displayed before truncation:

```python
>>> m, n = 8, 8
>>> a = Matrix((randint(0, 9) for _ in range(m * n)), m, n)
>>>
>>> print(a)  # Invokes __str__(), which invokes __format__("")
|        1        3        1        8        8        3        6        8 |
|        0        6        5        2        8        7        4        6 |
|        2        4        5        3        0        4        5        2 |
|        4        7        2        8        0        0        2        8 |
|        0        0        6        1        4        6        4        9 |
|        8        6        1        6        8        4        6        0 |
|        6        5        6        0        2        1        3        3 |
|        1        0        3        9        3        8        6        9 |
(8 × 8)
```

A truncation is represented by the use of a horizontal ellipsis character, `…` (U+2026). The last row, column, or character is replaced in this circumstance, and is left up to the user to expand the view if desired:

```python
>>> m, n = 10, 10
>>> a = Matrix((random() for _ in range(m * n)), m, n)
>>>
>>> print(a)
| 0.90234… 0.71933… 0.11622… 0.41670… 0.92819… 0.81760… 0.79436…        … |
| 0.75183… 0.56194… 0.27910… 0.00578… 0.92726… 0.10620… 0.56841…        … |
| 0.13038… 0.12136… 0.73665… 0.26133… 0.95974… 0.18828… 0.04638…        … |
| 0.81630… 0.05358… 0.34026… 0.94982… 0.34014… 0.12534… 0.12827…        … |
| 0.07987… 0.24251… 0.34159… 0.92052… 0.03523… 0.59975… 0.61997…        … |
| 0.38625… 0.85097… 0.33970… 0.15241… 0.57177… 0.17904… 0.18483…        … |
| 0.30821… 0.62894… 0.45595… 0.12608… 0.35640… 0.27327… 0.06264…        … |
|        …        …        …        …        …        …        …        … |
(10 × 10)
```

```python
>>> # nrows_max=15, ncols_max=8, chars_max=10
>>> print(f"{a:15,,10}")
| 0.9023438… 0.7193378… 0.1162233… 0.4167054… 0.9281957… 0.8176032… 0.7943611…          … |
| 0.7518317… 0.5619490… 0.2791072… 0.0057882… 0.9272604… 0.1062029… 0.5684195…          … |
| 0.1303844… 0.1213696… 0.7366537… 0.2613351… 0.9597416… 0.1882844… 0.0463849…          … |
| 0.8163065… 0.0535828… 0.3402611… 0.9498205… 0.3401455… 0.1253470… 0.1282773…          … |
| 0.0798795… 0.2425148… 0.3415924… 0.9205257… 0.0352319… 0.5997546… 0.6199738…          … |
| 0.3862552… 0.8509707… 0.3397048… 0.1524167… 0.5717743… 0.1790440… 0.1848337…          … |
| 0.3082173… 0.6289432… 0.4559529… 0.1260872… 0.3564015… 0.2732758… 0.0626408…          … |
| 0.2220599… 0.9194986… 0.3350282… 0.7225253… 0.6887936… 0.8080292… 0.3125350…          … |
| 0.5726533… 0.8472804… 0.9607844… 0.6287346… 0.4111070… 0.9845180… 0.5918491…          … |
| 0.3278540… 0.7251117… 0.6442113… 0.1088052… 0.9302837… 0.0227163… 0.2182001…          … |
(10 × 10)
```

```python
>>> # nrows_max=15, ncols_max=3, chars_max=5
>>> print(f"{a:15,3,5}")
| 0.90… 0.71…     … |
| 0.75… 0.56…     … |
| 0.13… 0.12…     … |
| 0.81… 0.05…     … |
| 0.07… 0.24…     … |
| 0.38… 0.85…     … |
| 0.30… 0.62…     … |
| 0.22… 0.91…     … |
| 0.57… 0.84…     … |
| 0.32… 0.72…     … |
(10 × 10)
```

If the maximum amount of row or column space is less than or equal to 0, all rows and columns will be written to the string. If the item space is less than or equal to 0, it will fallback to the default of 8 (since, the maximum amount of space necessary is indeterminable without a second pass through the matrix).

### Minor Changes

#### Directory

- The original test.py module has been split into a sub-library, under tests/.

#### `Matrix`

- The iterable argument to `__init__()` was renamed to `data` (from `values`), so as to align it more with its `__match_args__`.

#### `Shape`

- Added a warning notice to the docstrings of mutable operations.

#### Miscellaneous

- A new function, `only()`, has been added to the utilities.py module. It returns the *only* contained object of a length 1 collection, and is now used by the methods `__complex__()`, `__float__()`, `__int__()`, and `__index__()` implemented by the numeric matrix family. Their error messages are now a bit less specific as a consequence.
- The first argument to `likewise()` was renamed to `x` (from `object`).
- Minor alterations to internal formatting.

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
