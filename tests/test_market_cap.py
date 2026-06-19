import pandas as pd
import pytest

import market_cap as mc
from market_cap import _future_split_factors, load_market_cap_table, market_caps_by_month


def test_load_market_cap_table_preserves_availability_for_later_asof_use(tmp_path):
    path = tmp_path / "caps.parquet"
    pd.DataFrame(
        {
            "month_end": ["2020-01-31", "2020-01-31"],
            "ticker": ["A", "B"],
            "market_cap": [100.0, 200.0],
            "available_date": ["2020-01-31", "2020-02-05"],
            "source": ["test", "test"],
            "strict_pit": [True, True],
        }
    ).to_parquet(path, index=False)

    table = load_market_cap_table(path)

    assert table["ticker"].tolist() == ["A", "B"]
    january = market_caps_by_month(table, [pd.Timestamp("2020-01-31")], max_stale_days=45)
    february = market_caps_by_month(table, [pd.Timestamp("2020-02-29")], max_stale_days=45)
    assert january[pd.Timestamp("2020-01-31")].to_dict() == {"A": 100.0}
    assert february[pd.Timestamp("2020-02-29")].to_dict() == pytest.approx({"A": 100.0, "B": 200.0})


def test_market_caps_by_month_uses_latest_available_nonstale_value():
    table = pd.DataFrame(
        {
            "month_end": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-01-31"]),
            "ticker": ["A", "A", "B"],
            "market_cap": [100.0, 120.0, 200.0],
            "available_date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-01-31"]),
        }
    )
    months = pd.to_datetime(["2020-02-29", "2020-03-31"])

    caps = market_caps_by_month(table, months, max_stale_days=45)

    assert caps[months[0]].to_dict() == pytest.approx({"A": 120.0, "B": 200.0})
    assert caps[months[1]].to_dict() == pytest.approx({"A": 120.0})


def test_future_split_factor_restores_asof_price_units():
    dates = pd.Series(pd.to_datetime(["2019-12-31", "2020-09-30"]))
    factors = _future_split_factors(dates, [(pd.Timestamp("2020-08-31"), 4.0)])

    assert factors.tolist() == [4.0, 1.0]


def test_market_cap_builder_checkpoints_and_reuses_covered_request(monkeypatch, tmp_path):
    calls = []

    def fake_download(ticker, start, end, **kwargs):
        calls.append((ticker, start, end))
        return pd.DataFrame(
            {
                "month_end": [pd.Timestamp("2020-01-31")],
                "ticker": [ticker],
                "market_cap": [100.0],
                "available_date": [pd.Timestamp("2020-01-31")],
                "source": [mc.SOURCE_YAHOO_SHARES_PROXY],
                "strict_pit": [False],
                "method_version": [mc.MARKET_CAP_METHOD_VERSION],
            }
        )

    monkeypatch.setattr(mc, "_yahoo_market_cap_one", fake_download)
    path = tmp_path / "market_caps.parquet"

    first = mc.fetch_market_cap_history(["A"], "2020-01-01", "2020-01-31", cache_path=path)
    second = mc.fetch_market_cap_history(["A"], "2020-01-01", "2020-01-31", cache_path=path)

    assert len(calls) == 1
    assert first[["ticker", "market_cap"]].to_dict("records") == [{"ticker": "A", "market_cap": 100.0}]
    assert second[["ticker", "market_cap"]].to_dict("records") == [{"ticker": "A", "market_cap": 100.0}]
