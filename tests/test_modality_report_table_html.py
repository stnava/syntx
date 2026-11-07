"""``syntx.viz.table_html`` is the public table renderer (same output as the private ``_table``)."""

from syntx.viz import table_html
from syntx.viz.modality_report import _table


def test_table_html_matches_private_renderer():
    rows = [{"a": 1, "b": "x", "Status": "pass"}, {"a": 2, "b": "y", "Status": "fail"}]
    out = table_html(rows, ["a", "b", "Status"])
    assert out == _table(rows, ["a", "b", "Status"])
    assert "<table" in out and "x" in out and "y" in out
