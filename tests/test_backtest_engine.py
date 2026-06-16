import numpy as np
import pandas as pd
import pytest

from data_adapters import (
    _is_yfinance_ticker,
    _normalise_close_frame,
    _parse_ken_french_monthly_csv,
    _select_openfigi_ticker,
    align_holdings_to_prices,
    map_holdings_to_tickers,
    mapping_diagnostics,
    priceable_holdings,
)
from engine import BacktestConfig, UniverseConfig, attribution, run_backtest


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
