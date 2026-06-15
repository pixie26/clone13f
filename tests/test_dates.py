import pandas as pd
import pytest

from build_universe import SecDateParseError, _sec_date


def test_sec_date_parses_known_formats():
    raw = pd.Series(["2020-03-31", "20200331", "03/31/2020", "03-31-2020", "31-MAR-2020"])

    parsed = _sec_date(raw, column_name="TEST_DATE")

    assert parsed.tolist() == [pd.Timestamp("2020-03-31")] * 5


def test_sec_date_counts_malformed_values():
    raw = pd.Series(["03-31-2020", "not-a-date"])

    with pytest.raises(SecDateParseError, match="failed to parse 1/2"):
        _sec_date(raw, column_name="TEST_DATE")
