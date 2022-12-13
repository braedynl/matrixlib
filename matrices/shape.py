import operator
from collections.abc import Collection, Iterator
from typing import Any, Optional, SupportsIndex, TypeVar

from views import View

from .protocols import ShapeLike
from .rule import Rule

__all__ = ["Shape"]


ShapeT = TypeVar("ShapeT", bound="Shape")

class Shape(Collection[int], ShapeLike):
    """A mutable collection type for storing matrix dimensions

    Instances of `Shape` support integer - but not slice - indexing. Negative
    values are accepted by writing operations, and is left up to the matrix
    implementation to consider.

    Shapes should not be written-to when exposed by a matrix object, unless
    stated otherwise by the matrix class' documentation.
    """

    __slots__ = ("_data",)
    __match_args__ = ("nrows", "ncols")

    def __init__(self, nrows: int = 0, ncols: int = 0) -> None:
        """Construct a shape from its two dimensions"""
        self._data = [nrows, ncols]

    def __repr__(self) -> str:
        """Return a canonical representation of the shape"""
        nrows, ncols = self
        return f"{self.__class__.__name__}(nrows={nrows!r}, ncols={ncols!r})"

    def __str__(self) -> str:
        """Return a string representation of the shape"""
        nrows, ncols = self
        return f"{nrows} × {ncols}"

    def __getitem__(self, key: SupportsIndex) -> int:
        try:
            value = self._data[key]
        except IndexError as error:
            raise IndexError("index out of range") from error
        else:
            return value

    def __setitem__(self, key: SupportsIndex, value: int) -> None:
        """Set the dimension corresponding to `key` with `value`"""
        try:
            self._data[key] = value
        except IndexError as error:
            raise IndexError("index out of range") from error

    def __iter__(self) -> Iterator[int]:
        yield from self._data

    def __reversed__(self) -> Iterator[int]:
        yield from reversed(self._data)

    def __contains__(self, value: Any) -> bool:
        return value in self._data

    def __deepcopy__(self: ShapeT, memo: Optional[dict[int, Any]] = None) -> ShapeT:
        """Return a copy of the shape"""
        return self.__class__(*self)  # Our components are (hopefully) immutable

    __copy__ = __deepcopy__

    @property
    def data(self) -> View[int]:  # TODO: docstring
        return View(self._data)

    @property
    def nrows(self) -> int:
        return self[0]

    @nrows.setter
    def nrows(self, value: int) -> None:
        self[0] = value

    @property
    def ncols(self) -> int:
        return self[1]

    @ncols.setter
    def ncols(self, value: int) -> None:
        self[1] = value

    def copy(self: ShapeT) -> ShapeT:
        """Return a copy of the shape"""
        return self.__class__(*self)

    def reverse(self: ShapeT) -> ShapeT:
        """Reverse the shape's dimensions in place"""
        self._data.reverse()
        return self

    def subshape(self: ShapeT, *, by: Rule = Rule.ROW) -> ShapeT:
        """Return the shape of any sub-matrix in the given rule's form"""
        shape = self.copy()
        shape[by] = 1
        return shape

    def resolve_index(self, key: SupportsIndex, *, by: Rule = Rule.ROW) -> int:
        """Return an index `key` as an equivalent integer, respective to a rule

        Raises `IndexError` if the key is out of range.
        """
        n = self[by]
        i = operator.index(key)
        i += n * (i < 0)
        if i < 0 or i >= n:
            raise IndexError(f"there are {n} {by.handle}s but index is {key}")
        return i

    def resolve_slice(self, key: slice, *, by: Rule = Rule.ROW) -> range:
        """Return a slice `key` as an equivalent sequence of indices,
        respective to a rule
        """
        n = self[by]
        return range(*key.indices(n))

    def sequence(self, index: int, *, by: Rule = Rule.ROW) -> tuple[int, int, int]:
        """Return the start, stop, and step values required to create a range
        or slice object of the given rule's shape beginning at `index`

        The input `index` must be positive - negative indices may produce
        unexpected results. This requirement is not checked for.
        """
        dy = not by

        major = self[by]
        minor = self[dy]

        major_step = by * major + dy
        minor_step = dy * minor + by

        start = minor_step * index
        stop  = major_step * minor + start

        return start, stop, major_step

    def range(self, index: int, *, by: Rule = Rule.ROW) -> range:
        """Return a range of indices that can be used to construct a sub-matrix
        of the rule's shape beginning at `index`
        """
        return range(*self.sequence(index, by=by))

    def slice(self, index: int, *, by: Rule = Rule.ROW) -> slice:
        """Return a slice that can be used to construct a sub-matrix of the
        rule's shape beginning at `index`
        """
        return slice(*self.sequence(index, by=by))


ShapeViewT = TypeVar("ShapeViewT", bound="ShapeView")

class ShapeView(Collection[int], ShapeLike):

    __slots__ = ("_target",)

    def __init__(self, target: ShapeLike) -> None:
        self._target = target

    def __repr__(self) -> str:
        """Return a canonical representation of the shape view"""
        return f"{self.__class__.__name__}(target={self._target!r})"

    __str__ = __repr__

    def __getitem__(self, key: SupportsIndex) -> int:
        key = operator.index(key)
        return self._target[key]

    def __iter__(self) -> Iterator[int]:
        yield from self._target

    def __reversed__(self) -> Iterator[int]:
        yield from reversed(self._target)

    def __contains__(self, value: Any) -> bool:
        return value in self._target

    def __deepcopy__(self: ShapeViewT, memo: Optional[dict[int, Any]] = None) -> ShapeViewT:
        """Return the view"""
        return self

    __copy__ = __deepcopy__
