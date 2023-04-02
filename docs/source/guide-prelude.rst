.. _guide-prelude:

Prelude
=======

Fundamental to this library is the ``Matrix`` type. It is the root of all other ("built-in") matrix types, and is implemented in a way that may be strange to those who are used to thinking of matrices as two-dimensional arrays.

The NumPy ``ndarray``, despite its name, is internally represented as a contiguous, one-dimensional memory block, where it "fakes" the effect of being N-dimensional. ``Matrix``, too, uses a contiguous, one-dimensional memory block, but instead commits to the idea of higher-dimensional arrays as being a one-dimensional sequence by default:

>>> from matrixlib import Matrix
>>>
>>> a = Matrix(
...     (
...         1, 2, 3,
...         4, 5, 6,
...     ),
...     shape=(2, 3),
... )
>>>
>>> for x in a:
...     print(x)
...
1
2
3
4
5
6

This has many interesting consequences. For one, this enables all functions that accept a type of ``Iterable[T]`` (or even ``Sequence[T]``) to perform an element-wise operation across a matrix instance, without the need to flatten:

>>> sum(a)
21

There are, of course, functions you'll want to perform across rows and columns - the API *extends* the one-dimensional interface to include two-dimensional functionality for such tasks, and it does this in a manner that attempts to integrate with the rest of the Python language and philosophy.

If you can keep this premise in mind, then you can pretty easily look at the API Reference, alone, to get a fuller understanding of ``Matrix``'s capabilities. We do recommend taking at least a brief look at some other parts of the guide, however, as there are somewhat important "behind-the-scenes" details that can impact more advanced use cases of the library.
