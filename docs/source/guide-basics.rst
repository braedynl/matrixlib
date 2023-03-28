.. _guide-basics:

Basics
======

Fundamental to this library is the ``Matrix`` type. It is the root of all other ("built-in") matrix types, and is implemented in a way that may be strange to those who are used to thinking of matrices as two-dimensional arrays.

The NumPy ``ndarray``, despite its name, is internally represented as a contiguous, one-dimensional memory block, where it "fakes" the effect of being N-dimensional. ``Matrix``, too, uses a contiguous, one-dimensional memory block, but instead commits to the idea of matrices being a one-dimensional container by default:

>>> from matrices import Matrix
>>>
>>> a = Matrix([
...     1, 2, 3,
...     4, 5, 6,
... ], shape=(2, 3))
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

This has many interesting consequences. For one, this enables all functions that accept a type of ``Iterable[T]`` (or even ``Sequence[T]``) to perform an element-wise operation across a matrix instance:

>>> sum(a)
21

Meaning, we get to "inherit" a lot of functionality from the standard library that expect values at the shallowest depth, such as tools from the ``statistics`` module.

Due to matrices' interoperability with many of the standard library tools, a sub-module for mathematical/statistical functions does not exist, and one will likely never be created, as the ``math`` and ``statistics`` modules cover for their absence. There, of course, may be some functions you'll want to broadcast across rows or columns, which we'll discuss soon.

If what has been explained so far makes sense, then you really understand a majority of the API already. We do recommend taking at least a brief look at some other parts of the guide, however, as there are some important "behind-the-scenes" details that can impact more advanced use cases of the library.
