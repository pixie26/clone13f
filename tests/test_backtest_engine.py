import numpy as np
import pandas as pd
import pytest

import data_adapters as da
from data_adapters import (
    _is_yfinance_ticker,
    _load_price_cache,
    _normalise_close_frame,
    _parse_ken_french_monthly_csv,
    _select_openfigi_ticker,
    _write_price_cache,
    align_holdings_to_prices,
    map_holdings_to_tickers,
    mapping_diagnostics,
    priceable_holdings,
)
from engine import BacktestConfig, PortfolioConfig, UniverseConfig, attribution, rebalance_trace, run_backtest
from sweep import deflated_sharpe


def test_run_backtest_raises_on_missing_held_return():
    holdings = pd.DataFrame(
        [
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-01-31"),
                "accession_number": "a1",
                "submission_type": "13F-HR",
                "ticker": "A",
                "value": 100.0,
                "sec_type": "SH",
            }
        ]
    )
    prices = pd.DataFrame({"A": [0.01, np.nan]}, index=pd.to_datetime(["2020-01-31", "2020-02-29"]))

    cfg = BacktestConfig(
        universe=UniverseConfig(
            min_history_quarters=1,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        )
    )

    with pytest.raises(ValueError, match="Missing returns"):
        run_backtest(holdings, prices, cfg)


def test_rebalance_trace_reports_auditable_rebalance_fields():
    holdings = pd.DataFrame(
        [
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-01-15"),
                "accession_number": "a1",
                "submission_type": "13F-HR",
                "ticker": "A",
                "value": 80.0,
                "sec_type": "SH",
            },
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-01-15"),
                "accession_number": "a1",
                "submission_type": "13F-HR",
                "ticker": "B",
                "value": 20.0,
                "sec_type": "SH",
            },
            {
                "manager": "m2",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-01-20"),
                "accession_number": "a2",
                "submission_type": "13F-HR",
                "ticker": "A",
                "value": 70.0,
                "sec_type": "SH",
            },
            {
                "manager": "m2",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-01-20"),
                "accession_number": "a2",
                "submission_type": "13F-HR",
                "ticker": "C",
                "value": 30.0,
                "sec_type": "SH",
            },
        ]
    )
    prices = pd.DataFrame(
        {"A": [0.01], "B": [0.02], "C": [0.03]},
        index=pd.to_datetime(["2020-01-31"]),
    )
    cfg = BacktestConfig(
        universe=UniverseConfig(
            min_history_quarters=1,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(top_n_ideas=1, min_consensus_funds=2),
    )

    trace = rebalance_trace(holdings, prices, cfg)
    summary = trace["summary"].iloc[0]

    assert summary["rebalance_month"] == "2020-01-31"
    assert summary["selected_managers"] == 2
    assert summary["target_names"] == 1
    assert summary["effective_names"] == 1
    assert summary["turnover_one_way"] == pytest.approx(0.5)
    assert summary["cost_bps"] == pytest.approx(7.5)
    assert trace["holdings"]["ticker"].tolist() == ["A"]
    assert set(trace["managers"]["manager"]) == {"m1", "m2"}


def test_mapping_diagnostics_reports_unmapped_value():
    holdings = pd.DataFrame(
        {
            "cusip": ["111111111", "222222222"],
            "value": [75.0, 25.0],
        }
    )
    cmap = {"111111111": "AAA"}

    diag = mapping_diagnostics(holdings, cmap)
    mapped = map_holdings_to_tickers(holdings, cmap, strict=False)

    assert diag["cusips_mapped"] == 1
    assert diag["cusips_unmapped"] == 1
    assert diag["value_coverage"] == 0.75
    assert mapped["ticker"].tolist() == ["AAA"]


def test_priceable_holdings_drops_prn_and_bond_descriptions():
    holdings = pd.DataFrame(
        {
            "ticker": ["AAPL", "TTEK 2.25 08/15/28", "MSFT"],
            "sec_type": ["SH", "SH", "PRN"],
            "value": [1.0, 2.0, 3.0],
        }
    )

    filtered = priceable_holdings(holdings)

    assert filtered["ticker"].tolist() == ["AAPL"]
    assert filtered.attrs["price_filter_diagnostics"]["rows_dropped"] == 2


def test_yfinance_ticker_filter_rejects_non_us_vendor_symbols():
    bad = ["0HQK", "2655787D", "ACLXGBX", "ALLKGUSD", "ANETEUR", "A2O1", "AM6", "TTEK 2.25 08/15/28"]
    good = ["AAPL", "GOOGL", "BRK-B", "BF.B", "AAXJ"]

    assert not any(_is_yfinance_ticker(x) for x in bad)
    assert all(_is_yfinance_ticker(x) for x in good)


def test_openfigi_selector_prefers_us_equity_ticker():
    data = [
        {"ticker": "0HQK", "marketSector": "Equity", "exchCode": "LN"},
        {"ticker": "FANG", "marketSector": "Equity", "exchCode": "UW", "securityType2": "Common Stock"},
        {"ticker": "FANG 3.25 2029", "marketSector": "Corp", "securityType2": "Corporate Bond"},
    ]

    assert _select_openfigi_ticker(data) == "FANG"


def test_openfigi_selector_rejects_non_us_exchange():
    data = [
        {"ticker": "ALEX", "marketSector": "Equity", "exchCode": "LN", "securityType2": "Common Stock"},
    ]

    assert _select_openfigi_ticker(data) is None


def test_align_holdings_to_prices_drops_unpriced_tickers():
    holdings = pd.DataFrame(
        {
            "ticker": ["AAPL", "DEAD"],
            "value": [80.0, 20.0],
        }
    )
    prices = pd.DataFrame({"AAPL": [0.01]}, index=pd.to_datetime(["2020-01-31"]))

    aligned = align_holdings_to_prices(holdings, prices)

    assert aligned["ticker"].tolist() == ["AAPL"]
    assert aligned.attrs["price_alignment_diagnostics"]["tickers_without_prices"] == 1
    assert aligned.attrs["price_alignment_diagnostics"]["value_coverage"] == 0.8


def test_attribution_without_factors_reports_basic_metrics():
    ret = pd.Series([0.01] * 12, index=pd.date_range("2020-01-31", periods=12, freq="ME"))

    out = attribution(ret, pd.DataFrame(index=ret.index))

    assert out["n_months"] == 12
    assert out["note"] == "factor regression unavailable"
    assert "ann_return" in out
    assert "ann_alpha" not in out


def test_parse_ken_french_monthly_csv():
    text = (
        "header\n"
        ",Mkt-RF,SMB,HML,RMW,CMA,RF\n"
        "202501,1.0,2.0,3.0,4.0,5.0,0.1\n"
        "202502,-1.0,0.0,1.0,2.0,3.0,0.2\n"
        "\n"
        "Annual Factors: January-December\n"
    )

    parsed = _parse_ken_french_monthly_csv(text)

    assert parsed.index.tolist() == [pd.Timestamp("2025-01-31"), pd.Timestamp("2025-02-28")]
    assert parsed.loc[pd.Timestamp("2025-01-31"), "Mkt-RF"] == 0.01
    assert parsed.loc[pd.Timestamp("2025-02-28"), "RF"] == 0.002


def test_normalise_close_frame_accepts_ticker_first_yfinance_columns():
    cols = pd.MultiIndex.from_tuples(
        [("AAPL", "Open"), ("AAPL", "Close"), ("MSFT", "Close")]
    )
    raw = pd.DataFrame([[100.0, 101.0, 201.0]], columns=cols, index=pd.to_datetime(["2025-01-02"]))

    close = _normalise_close_frame(raw, ["AAPL", "MSFT"])

    assert close.columns.tolist() == ["AAPL", "MSFT"]
    assert close.iloc[0].tolist() == [101.0, 201.0]


def test_price_cache_preserves_existing_columns_and_filters_dates(tmp_path):
    cache_path = tmp_path / "prices.parquet"
    first = pd.DataFrame(
        {"AAPL": [100.0, 101.0]},
        index=pd.to_datetime(["2024-12-31", "2025-01-31"]),
    )
    second = pd.DataFrame(
        {"MSFT": [200.0]},
        index=pd.to_datetime(["2025-01-31"]),
    )

    _write_price_cache(cache_path, first)
    _write_price_cache(cache_path, second)
    loaded = _load_price_cache(cache_path, "2025-01-01", "2025-01-31")

    assert loaded.index.tolist() == [pd.Timestamp("2025-01-31")]
    assert loaded.columns.tolist() == ["AAPL", "MSFT"]
    assert loaded.loc[pd.Timestamp("2025-01-31"), "AAPL"] == 101.0
    assert loaded.loc[pd.Timestamp("2025-01-31"), "MSFT"] == 200.0


def test_fetch_prices_uses_chart_fallback_when_yfinance_probe_is_empty(monkeypatch, tmp_path):
    dates = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])

    def fake_yf_download_close(yf, batch, start, end):
        return pd.DataFrame()

    def fake_yfinance_probe(yf, start, end):
        return "probe_empty_close"

    def fake_chart_probe(start, end):
        return "chart_probe_ok=SPY,IWD,AAPL"

    def fake_chart_download_close(batch, start, end, *, max_workers=8):
        data = {ticker: [100.0, 110.0, 121.0] for ticker in batch}
        return pd.DataFrame(data, index=dates)

    monkeypatch.setattr(da, "_yf_download_close", fake_yf_download_close)
    monkeypatch.setattr(da, "_yfinance_probe", fake_yfinance_probe)
    monkeypatch.setattr(da, "_yahoo_chart_probe", fake_chart_probe)
    monkeypatch.setattr(da, "_yahoo_chart_download_close", fake_chart_download_close)

    returns = da.fetch_prices(
        ["AAPL", "MSFT"],
        "2025-01-01",
        "2025-03-31",
        batch_size=1,
        max_retries=0,
        max_consecutive_empty_batches=1,
        cache_path=tmp_path / "prices.parquet",
    )

    assert returns.columns.tolist() == ["AAPL", "MSFT"]
    assert returns.attrs["price_diagnostics"]["used_chart_fallback"] is True
    assert returns.attrs["price_diagnostics"]["tickers_with_close"] == 2
    assert returns.iloc[-1].tolist() == pytest.approx([0.1, 0.1])


def test_fetch_prices_drops_incomplete_monthly_return_columns(monkeypatch):
    dates = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])

    def fake_yf_download_close(yf, batch, start, end):
        return pd.DataFrame(
            {
                "AAPL": [100.0, 110.0, 121.0],
                "MSFT": [100.0, np.nan, 121.0],
            },
            index=dates,
        )

    monkeypatch.setattr(da, "_yf_download_close", fake_yf_download_close)

    returns = da.fetch_prices(
        ["AAPL", "MSFT"],
        "2025-01-01",
        "2025-03-31",
        batch_size=2,
        max_retries=0,
    )

    assert returns.columns.tolist() == ["AAPL"]
    assert returns.attrs["price_diagnostics"]["tickers_with_close"] == 2
    assert returns.attrs["price_diagnostics"]["tickers_with_complete_returns"] == 1
    assert returns.attrs["price_diagnostics"]["tickers_incomplete_returns"] == 1


def test_deflated_sharpe_reports_trials_when_oos_is_insufficient():
    out = deflated_sharpe(pd.Series([0.01] * 6), n_trials=16)

    assert out["note"] == "insufficient OOS"
    assert out["T"] == 6
    assert out["n_trials"] == 16
