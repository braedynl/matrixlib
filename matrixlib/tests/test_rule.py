from ..rule import Rule


class TestRule:

    def test_value(self) -> None:
        assert Rule.ROW.value == 0
        assert Rule.COL.value == 1

    def test_handle(self) -> None:
        assert Rule.ROW.handle == "row"
        assert Rule.COL.handle == "column"

    def test_invert(self) -> None:
        assert ~Rule.ROW is Rule.COL
        assert ~Rule.COL is Rule.ROW
