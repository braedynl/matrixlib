.. _changelog:

Changelog
=========

This document lists API changes for users and contributors under the labels "public" and "private", respectively. Changes made to documentation are not listed. The logs of a specific version describe changes respective to the version before. Any change that could break existing code is marked as "(*breaking change*)".

Version 0.3.1
-------------

Public
^^^^^^

Additions

* Added ``builtins.IntegralMatrix`` as a type alias of ``builtins.IntegerMatrix``, for those that work (and want consistency) with the ``numbers.Integral`` ABC. Do note that the class still only accepts sub-types of ``int``.
* The ``EvenNumber`` and ``OddNumber`` types used by ``Matrix.rotate()`` now covers literal integers from -16 to +16 - this was originally -4 to +4.

Alterations

* The top-level module has been renamed from ``matrices`` to ``matrixlib``. The library's "branding" has been changed to reflect this across all relevant systems (GitHub, PyPI, etc.) (*breaking change*).
* The string returned by ``Matrix.__repr__()`` no longer invokes the ``array`` argument by keyword. The ``array`` argument itself can, and will continue to be invocable by keyword, however.

Private
^^^^^^^

Additions

* Added a ``typing`` sub-module for miscellaneous typing needs. This module is not exposed at the top level (though that may change in the future).

Alterations

* The ``EvenNumber`` and ``OddNumber`` types have been moved to the ``typing`` sub-module. These are the only two definitions that exist in the module as of right now.

Version 0.3.0
-------------

The first "official" version of the library.
