import numpy as np
import pandas as pd
import pytest

from data_adapters import map_holdings_to_tickers, mapping_diagnostics
from engine import BacktestConfig, UniverseConfig, run_backtest


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
