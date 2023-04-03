.. _changelog:

Changelog
=========

This document lists API changes for users and contributors under the labels "public" and "private", respectively. Changes made to documentation are not listed. The logs of a specific version describe changes respective to the version before. Any change that could break existing code is marked as "(*breaking change*)".

Version 0.3.2
-------------

Public
^^^^^^

* Added a docstring for ``Matrix.__len__()`` (this was an unintentional omission).
* Changed the type-hinting of ``Matrix.values()`` to show resemblance to ``Matrix.slices()``. Its functionality remains the same.

Private
^^^^^^^

* The typing sub-module now imports ``annotations`` from ``__future__``.
* The string returned by the ``__repr__()`` of all materialized ``Mesh`` sub-classes now invoke their constructor arguments by keyword.
* Changed the signature of ``Box.__init__()`` to accept a one-element ``tuple``, rather than an element. This brings it more in-line with the ``Row`` and ``Col`` meshes. ``Box`` still only stores a sole value, however - its ``array`` is constructed at access time, like in prior versions.

Version 0.3.1
-------------

Public
^^^^^^

* Added ``builtins.IntegralMatrix`` as a type alias of ``builtins.IntegerMatrix``, for those that work (and want consistency) with the ``numbers.Integral`` ABC. Do note that the class still only accepts sub-types of ``int``.
* The ``EvenNumber`` and ``OddNumber`` types used by ``Matrix.rotate()`` now covers literal integers from -16 to +16 - this was originally -4 to +4.
* The top-level module has been renamed from ``matrices`` to ``matrixlib``. The library's "branding" has been changed to reflect this across all relevant systems (GitHub, PyPI, etc.) (*breaking change*).
* The string returned by ``Matrix.__repr__()`` no longer invokes the ``array`` argument by keyword. The ``array`` argument itself can, and will continue to be invocable by keyword, however.

Private
^^^^^^^

* Added a ``typing`` sub-module for miscellaneous typing needs. This module is not exposed at the top level (though that may change in the future).
* The ``EvenNumber`` and ``OddNumber`` types have been moved to the ``typing`` sub-module. These are the only objects that exist within it as of right now.

Version 0.3.0
-------------

The first "official" version of the library.
