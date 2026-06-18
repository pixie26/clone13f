import numpy as np
import pandas as pd

from engine import BacktestConfig, PortfolioConfig, UniverseConfig, manager_characteristics, run_backtest
from manager_classifier import (
    ManagerClassifierConfig,
    _apply_dedicated_persistence,
    build_manager_classification,
    filter_selected_versions,
    frame_hash,
)


def _row(manager, name, period, filing, accession, ticker, value, sec_type="SH", is_fund_like=False):
    return {
        "manager": manager,
        "manager_name": name,
        "period_date": pd.Timestamp(period),
        "filing_date": pd.Timestamp(filing),
        "accession_number": accession,
        "submission_type": "13F-HR",
        "ticker": ticker,
        "issuer": ticker,
        "value": float(value),
        "sec_type": sec_type,
        "is_fund_like": bool(is_fund_like),
    }


def _factors(months):
    out = pd.DataFrame(index=months)
    for col in ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]:
        out[col] = np.linspace(0.001, 0.003, len(months))
    out["RF"] = 0.0
    return out


def test_static_detection_drops_dirty_names_but_keeps_canonical_compounders():
    months = pd.date_range("2020-05-31", periods=8, freq="ME")
    raw = pd.DataFrame(
        [
            _row("1", "Donor Advised Charitable Giving Inc", "2020-03-31", "2020-05-15", "a1", "AAPL", 100),
            _row("2", "Berkshire Hathaway Inc", "2020-03-31", "2020-05-15", "a2", "MSFT", 100),
        ]
    )
    chars = manager_characteristics(raw)
    prices = pd.DataFrame({"AAPL": 0.01, "MSFT": 0.01}, index=months)

    c = build_manager_classification(raw, raw, chars, months, prices, _factors(months))
    latest = c.sort_values("asof_month").groupby("manager").tail(1).set_index("manager")

    assert bool(latest.loc["0000000001", "dirty_flag"]) is True
    assert "donor_advised_or_charity" in latest.loc["0000000001", "dirty_reason"]
    assert bool(latest.loc["0000000002", "dirty_flag"]) is False


def test_all_mode_ignores_non_empty_override_and_preserves_returns():
    months = pd.date_range("2020-05-31", periods=3, freq="ME")
    holdings = pd.DataFrame(
        [
            _row("1", "Test Manager", "2020-03-31", "2020-05-15", "a1", "AAPL", 100),
            _row("2", "Other Manager", "2020-03-31", "2020-05-15", "a2", "MSFT", 100),
        ]
    )
    prices = pd.DataFrame({"AAPL": [0.01, 0.02, 0.03], "MSFT": [0.0, 0.0, 0.0]}, index=months)
    cfg = BacktestConfig(
        universe=UniverseConfig(
            min_history_quarters=1,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(idea_signal="level", min_consensus_funds=1, max_name_weight=1.0),
        manager_filter_mode="all",
    )
    chars = manager_characteristics(holdings)
    classification = build_manager_classification(holdings, holdings, chars, months, prices, _factors(months))
    overrides = pd.DataFrame({"manager": ["0000000001"], "action": ["deny"], "manager_type": [""], "note": ["test"]})

    base = run_backtest(holdings, prices, cfg, chars=chars)
    with_overrides = run_backtest(
        holdings,
        prices,
        cfg,
        chars=chars,
        manager_classification=classification,
        manager_overrides=overrides,
    )

    pd.testing.assert_series_equal(base, with_overrides)


def test_etf_share_uses_raw_book_before_security_level_filter():
    months = pd.date_range("2020-05-31", periods=4, freq="ME")
    raw = pd.DataFrame(
        [
            _row("1", "ETF Parking Manager", "2020-03-31", "2020-05-15", "a1", "SPY", 80, is_fund_like=True),
            _row("1", "ETF Parking Manager", "2020-03-31", "2020-05-15", "a1", "AAPL", 20),
        ]
    )
    filtered = raw[raw["ticker"].eq("AAPL")].copy()
    chars = manager_characteristics(filtered)
    prices = pd.DataFrame({"AAPL": 0.01}, index=months)

    c = build_manager_classification(raw, filtered, chars, months, prices, _factors(months))

    assert c["etf_share_raw"].max() == 0.8
    assert "high_etf_share_raw" in c.sort_values("asof_month").iloc[-1]["dirty_reason"]


def test_factor_r2_insufficient_status_when_history_or_names_are_too_short():
    months = pd.date_range("2020-05-31", periods=4, freq="ME")
    holdings = pd.DataFrame(
        [_row("1", "Short History Manager", "2020-03-31", "2020-05-15", "a1", "AAPL", 100)]
    )
    chars = manager_characteristics(holdings)
    prices = pd.DataFrame({"AAPL": [0.01, 0.02, 0.03, 0.04]}, index=months)

    c = build_manager_classification(holdings, holdings, chars, months, prices, _factors(months))

    assert c["factor_r2"].isna().all()
    assert set(c["factor_r2_status"]) == {"insufficient_factor_r2"}


def test_persistence_counts_calendar_quarters_not_rebalance_events():
    cfg = ManagerClassifierConfig(persistence_quarters=2)
    df = pd.DataFrame(
        {
            "manager": ["m1", "m1", "m1"],
            "asof_month": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-04-30"]),
            "raw_dedicated": [True, True, True],
        }
    )

    persistent = _apply_dedicated_persistence(df, cfg)

    assert persistent.tolist() == [False, False, True]


def test_filter_modes_and_deterministic_classification_hash():
    months = pd.date_range("2020-05-31", periods=4, freq="ME")
    raw = pd.DataFrame(
        [
            _row("1", "Donor Advised Charitable Giving Inc", "2020-03-31", "2020-05-15", "a1", "AAPL", 100),
            _row("2", "Normal Manager", "2020-03-31", "2020-05-15", "a2", "MSFT", 100),
        ]
    )
    chars = manager_characteristics(raw)
    prices = pd.DataFrame({"AAPL": 0.01, "MSFT": 0.01}, index=months)

    c1 = build_manager_classification(raw, raw, chars, months, prices, _factors(months))
    c2 = build_manager_classification(raw.sample(frac=1, random_state=1), raw, chars, months, prices, _factors(months))
    selected = chars[chars["filing_date"].le(months[-1])].sort_values("manager").groupby("manager").tail(1)

    filtered = filter_selected_versions(selected, months[-1], "exclude_dirty", c1)

    assert filtered["manager"].tolist() == ["0000000002"]
    assert frame_hash(c1) == frame_hash(c2)
