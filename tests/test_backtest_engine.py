import numpy as np
import pandas as pd
import pytest

import data_adapters as da
import engine as en
import report as rp
import sweep as sw
from data_adapters import (
    _is_yfinance_ticker,
    _load_openfigi_cache,
    _load_price_cache,
    _normalise_close_frame,
    _openfigi_id_type,
    _parse_ken_french_monthly_csv,
    _select_openfigi_ticker,
    _write_openfigi_cache,
    cusip_to_ticker,
    _write_price_cache,
    align_holdings_to_prices,
    load_openfigi_metadata,
    map_holdings_to_tickers,
    mapping_diagnostics,
    priceable_holdings,
)
from engine import (
    BacktestConfig,
    PortfolioConfig,
    UniverseConfig,
    _cap_weights,
    _cap_weights_with_groups,
    attribution,
    filter_universe_versions,
    rebalance_trace,
    run_backtest,
    target_weights_from_versions,
)
from run_example import _rebalance_summary_stats, load_security_groups, run as run_example_run, value_unit_continuity_diagnostics
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
        ),
        missing_price_policy="raise",
    )

    with pytest.raises(ValueError, match="Missing returns"):
        run_backtest(holdings, prices, cfg)


def test_run_backtest_exits_missing_held_return_by_default():
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

    ret = run_backtest(holdings, prices, cfg)

    assert ret.loc[pd.Timestamp("2020-02-29")] == 0.0


def test_rebalance_skips_target_without_current_month_price():
    holdings = pd.DataFrame(
        [
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2019-12-31"),
                "filing_date": pd.Timestamp("2020-01-15"),
                "accession_number": "a1",
                "submission_type": "13F-HR",
                "ticker": "A",
                "value": 100.0,
                "sec_type": "SH",
            },
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-02-15"),
                "accession_number": "a2",
                "submission_type": "13F-HR",
                "ticker": "A",
                "value": 100.0,
                "sec_type": "SH",
            },
        ]
    )
    prices = pd.DataFrame({"A": [0.0, np.nan]}, index=pd.to_datetime(["2020-01-31", "2020-02-29"]))
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

    trace = rebalance_trace(holdings, prices, cfg)
    feb = trace["summary"][trace["summary"]["rebalance_month"].eq("2020-02-29")].iloc[0]

    assert feb["target_names"] == 0
    assert feb["effective_names"] == 0


def test_cap_weights_enforces_feasible_name_cap():
    capped = _cap_weights(pd.Series({"A": 0.60, "B": 0.20, "C": 0.10, "D": 0.06, "E": 0.04}), 0.25)

    assert capped.sum() == pytest.approx(1.0)
    assert capped.max() <= 0.25 + 1e-12


def test_cap_weights_with_groups_limits_share_classes_by_issuer():
    raw = pd.Series({"GOOG": 0.35, "GOOGL": 0.25, "MSFT": 0.20, "AMZN": 0.12, "META": 0.08})
    groups = pd.Series({"GOOG": "ALPHABET", "GOOGL": "ALPHABET"})

    capped = _cap_weights_with_groups(raw, max_name_weight=0.40, max_issuer_weight=0.30, security_groups=groups)

    assert capped.sum() == pytest.approx(1.0)
    assert capped.max() <= 0.40 + 1e-12
    assert capped.loc[["GOOG", "GOOGL"]].sum() <= 0.30 + 1e-12


def test_value_tilt_only_scores_previously_filtered_candidates(monkeypatch):
    calls = []
    latest = pd.DataFrame(
        [
            {
                "manager": "too_large",
                "hist_q": 4,
                "aum": 100.0,
                "top10_weight": 0.90,
                "n_holdings": 5,
                "put_weight": 0.0,
                "active_share": np.nan,
                "turnover": 0.1,
                "period_date": pd.Timestamp("2020-03-31"),
                "bw": pd.Series({"A": 1.0}),
            },
            {
                "manager": "candidate",
                "hist_q": 4,
                "aum": 10.0,
                "top10_weight": 0.90,
                "n_holdings": 5,
                "put_weight": 0.0,
                "active_share": np.nan,
                "turnover": 0.1,
                "period_date": pd.Timestamp("2020-03-31"),
                "bw": pd.Series({"B": 1.0}),
            },
        ]
    )
    cfg = UniverseConfig(
        min_aum=0.0,
        max_aum=30.0,
        min_history_quarters=1,
        use_size_band=True,
        use_concentration=False,
        use_low_turnover=False,
        use_hedge_filter=False,
        use_value_tilt=True,
        value_tilt_min_pctl=0.50,
    )
    value_scores = pd.DataFrame({"A": [1.0], "B": [1.0]}, index=[pd.Timestamp("2020-03-31")])

    def fake_book_value_pctl(w, vscores):
        calls.append(tuple(w.index))
        return 1.0

    monkeypatch.setattr(en, "_book_value_pctl", fake_book_value_pctl)

    selected = filter_universe_versions(latest, cfg, value_scores)

    assert selected["manager"].tolist() == ["candidate"]
    assert calls == [("B",)]


def test_active_weight_signal_uses_relative_overweight_not_absolute_level():
    latest_versions = pd.DataFrame(
        [
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-05-15"),
                "accession_number": "a1",
                "bw": pd.Series({"SPY": 0.40, "A": 0.30, "B": 0.30}),
                "prev_bw": None,
            }
        ]
    )
    benchmark = pd.Series({"SPY": 0.50, "A": 0.05, "B": 0.45})
    cfg = PortfolioConfig(
        idea_signal="active_weight",
        top_n_ideas=1,
        min_consensus_funds=1,
        max_name_weight=1.0,
        min_active_weight_holdings=1,
    )

    target = target_weights_from_versions(latest_versions, cfg, benchmark)

    assert target.index.tolist() == ["A"]
    assert target.loc["A"] == pytest.approx(1.0)


def test_active_weight_signal_requires_minimum_book_breadth():
    latest_versions = pd.DataFrame(
        [
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-05-15"),
                "accession_number": "a1",
                "bw": pd.Series({"A": 0.80, "B": 0.20}),
                "prev_bw": None,
            }
        ]
    )
    benchmark = pd.Series({"A": 0.10, "B": 0.10, "C": 0.80})
    cfg = PortfolioConfig(
        idea_signal="active_weight",
        top_n_ideas=1,
        min_consensus_funds=1,
        max_name_weight=1.0,
        min_active_weight_holdings=3,
    )

    target = target_weights_from_versions(latest_versions, cfg, benchmark)

    assert target.empty


def test_active_weight_change_requires_positive_current_overweight():
    latest_versions = pd.DataFrame(
        [
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-05-15"),
                "accession_number": "a1",
                "bw": pd.Series({"A": 0.40, "B": 0.35, "C": 0.25}),
                "prev_bw": pd.Series({"A": 0.20, "B": 0.10, "C": 0.70}),
            }
        ]
    )
    benchmark = pd.Series({"A": 0.10, "B": 0.50, "C": 0.40})
    cfg = PortfolioConfig(
        idea_signal="active_weight_change",
        top_n_ideas=2,
        min_consensus_funds=1,
        max_name_weight=1.0,
        min_active_weight_holdings=3,
    )

    target = target_weights_from_versions(latest_versions, cfg, benchmark)

    assert target.index.tolist() == ["A"]
    assert target.loc["A"] == pytest.approx(1.0)


def test_active_weight_change_scores_current_overweight_not_weight_product():
    latest_versions = pd.DataFrame(
        [
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-05-15"),
                "accession_number": "a1",
                "bw": pd.Series({"A": 0.31, "B": 0.30, "C": 0.39}),
                "prev_bw": pd.Series({"A": 0.30, "B": 0.01, "C": 0.69}),
            }
        ]
    )
    benchmark = pd.Series({"A": 0.10, "B": 0.20, "C": 0.70})
    cfg = PortfolioConfig(
        idea_signal="active_weight_change",
        top_n_ideas=1,
        min_consensus_funds=1,
        max_name_weight=1.0,
        min_active_weight_holdings=3,
    )

    target = target_weights_from_versions(latest_versions, cfg, benchmark)

    assert target.index.tolist() == ["A"]


def test_active_weight_initiation_uses_new_positive_overweights_only():
    latest_versions = pd.DataFrame(
        [
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-05-15"),
                "accession_number": "a1",
                "bw": pd.Series({"A": 0.30, "B": 0.30, "C": 0.40}),
                "prev_bw": pd.Series({"A": 0.60, "B": 0.40}),
            }
        ]
    )
    benchmark = pd.Series({"A": 0.10, "B": 0.10, "C": 0.05})
    cfg = PortfolioConfig(
        idea_signal="active_weight_initiation",
        top_n_ideas=2,
        min_consensus_funds=1,
        max_name_weight=1.0,
        min_active_weight_holdings=3,
    )

    target = target_weights_from_versions(latest_versions, cfg, benchmark)

    assert target.index.tolist() == ["C"]
    assert target.loc["C"] == pytest.approx(1.0)


def test_max_portfolio_names_caps_aggregate_target_before_weight_caps():
    latest_versions = pd.DataFrame(
        [
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-05-15"),
                "accession_number": "a1",
                "bw": pd.Series({"A": 0.50, "B": 0.30, "C": 0.20}),
                "prev_bw": None,
            },
            {
                "manager": "m2",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-05-15"),
                "accession_number": "a2",
                "bw": pd.Series({"A": 0.40, "D": 0.35, "E": 0.25}),
                "prev_bw": None,
            },
        ]
    )
    cfg = PortfolioConfig(
        idea_signal="level",
        top_n_ideas=3,
        min_consensus_funds=1,
        max_portfolio_names=2,
        max_name_weight=1.0,
    )

    target = target_weights_from_versions(latest_versions, cfg)

    assert target.index.tolist() == ["A", "D"]
    assert target.sum() == pytest.approx(1.0)


def test_target_weights_applies_eligible_tickers_before_top_n_selection():
    latest_versions = pd.DataFrame(
        [
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-05-15"),
                "accession_number": "a1",
                "bw": pd.Series({"STALE": 0.40, "A": 0.30, "B": 0.20, "C": 0.10}),
                "prev_bw": None,
            }
        ]
    )
    cfg = PortfolioConfig(
        idea_signal="level",
        top_n_ideas=4,
        min_consensus_funds=1,
        max_portfolio_names=3,
        max_name_weight=1.0,
    )

    target = target_weights_from_versions(latest_versions, cfg, eligible_tickers=["A", "B", "C"])

    assert target.index.tolist() == ["A", "B", "C"]
    assert target.sum() == pytest.approx(1.0)


def test_iter_configs_supports_paired_aum_band_axis():
    base = BacktestConfig()

    configs = sw.iter_configs(
        base,
        {
            ("universe", "aum_band"): [("0.5-5B", 0.5e9, 5e9), ("15-30B", 15e9, 30e9)],
            ("portfolio", "top_n_ideas"): [3],
        },
    )

    labels = [label for label, _ in configs]
    cfgs = [cfg for _, cfg in configs]
    assert labels == [{"aum_band": "0.5-5B", "top_n_ideas": 3}, {"aum_band": "15-30B", "top_n_ideas": 3}]
    assert cfgs[0].universe.min_aum == 0.5e9
    assert cfgs[0].universe.max_aum == 5e9
    assert cfgs[1].universe.min_aum == 15e9
    assert cfgs[1].universe.max_aum == 30e9


def test_walk_forward_selects_on_active_sharpe_not_raw_market_return(monkeypatch):
    months = pd.date_range("2020-01-31", periods=3, freq="ME")
    prices = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=months)
    benchmark = pd.Series([0.05, 0.04, 0.03], index=months, name="SPY")
    base = BacktestConfig(
        universe=UniverseConfig(
            min_history_quarters=1,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(idea_signal="level"),
    )

    def fake_run_backtest(
        holdings,
        px,
        cfg,
        value_scores=None,
        benchmark_weights=None,
        chars=None,
        visible_versions_cache=None,
        security_groups=None,
        active_benchmark_weights_by_month=None,
    ):
        if cfg.portfolio.idea_signal == "active_weight":
            active = pd.Series([0.01, 0.02, 0.01], index=months).reindex(px.index)
        else:
            active = pd.Series([-0.02, -0.01, -0.02], index=months).reindex(px.index)
        return benchmark.reindex(px.index) + active

    monkeypatch.setattr(sw, "run_backtest", fake_run_backtest)

    _, log, _ = sw.walk_forward(
        pd.DataFrame(),
        prices,
        pd.DataFrame(index=months),
        base,
        {("portfolio", "idea_signal"): ["level", "active_weight"]},
        benchmark=benchmark,
        train_m=2,
        test_m=1,
        select_on="active_sharpe",
        chars=pd.DataFrame(),
        visible_versions_cache={months[0]: pd.DataFrame()},
    )

    assert log["idea_signal"].tolist() == ["active_weight"]


def test_walk_forward_can_reuse_precomputed_config_returns(monkeypatch):
    months = pd.date_range("2020-01-31", periods=3, freq="ME")
    prices = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=months)
    benchmark = pd.Series([0.0, 0.0, 0.0], index=months, name="SPY")
    base = BacktestConfig(
        universe=UniverseConfig(
            min_history_quarters=1,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(idea_signal="level"),
    )

    def fail_run_backtest(*args, **kwargs):
        raise AssertionError("walk_forward should use precomputed returns")

    monkeypatch.setattr(sw, "run_backtest", fail_run_backtest)
    precomputed = {
        (("idea_signal", "level"),): pd.Series([0.01, 0.02, 0.01], index=months),
    }

    oos, log, n_trials = sw.walk_forward(
        pd.DataFrame(),
        prices,
        pd.DataFrame(index=months),
        base,
        {("portfolio", "idea_signal"): ["level"]},
        benchmark=benchmark,
        train_m=2,
        test_m=1,
        select_on="active_sharpe",
        chars=pd.DataFrame(),
        visible_versions_cache={months[0]: pd.DataFrame()},
        precomputed_returns=precomputed,
    )

    assert n_trials == 1
    assert log["idea_signal"].tolist() == ["level"]
    assert oos.index.tolist() == [months[-1]]


def test_live_run_preflights_missing_spy_weight_file_before_data_build(tmp_path, monkeypatch):
    def fail_build_live_data(*args, **kwargs):
        raise AssertionError("live data build should not start when SPY weights are missing")

    monkeypatch.setattr("run_example.build_live_data", fail_build_live_data)

    missing = tmp_path / "missing_spy_weights.parquet"
    with pytest.raises(FileNotFoundError, match="active benchmark weight file is required"):
        run_example_run(
            "live",
            tmp_path,
            active_benchmark_source="spy_holdings",
            active_benchmark_weights_path=str(missing),
        )


def test_grid_eval_active_sharpe_skips_factor_attribution(monkeypatch):
    months = pd.date_range("2020-01-31", periods=3, freq="ME")
    prices = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=months)
    benchmark = pd.Series([0.0, 0.01, 0.0], index=months, name="SPY")
    base = BacktestConfig(
        universe=UniverseConfig(
            min_history_quarters=1,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(idea_signal="level"),
    )

    def fake_run_backtest(*args, **kwargs):
        return pd.Series([0.01, 0.02, 0.03], index=months)

    def fail_attribution(*args, **kwargs):
        raise AssertionError("active_sharpe grid should not run factor attribution")

    monkeypatch.setattr(sw, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(sw, "attribution", fail_attribution)

    grid = sw.grid_eval(
        pd.DataFrame(),
        prices,
        pd.DataFrame(index=months),
        base,
        {("portfolio", "idea_signal"): ["level"]},
        benchmark=benchmark,
        metric="active_sharpe",
        chars=pd.DataFrame(),
        visible_versions_cache={},
        use_selection_cache=False,
    )

    assert len(grid) == 1
    assert np.isfinite(grid.loc[0, "active_sharpe"])
    assert grid.loc[0, "ir"] == grid.loc[0, "active_sharpe"]
    assert pd.isna(grid.loc[0, "ann_alpha"])
    assert pd.isna(grid.loc[0, "alpha_t"])


def test_grid_eval_writes_partial_checkpoint(tmp_path, monkeypatch):
    months = pd.date_range("2020-01-31", periods=3, freq="ME")
    prices = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=months)
    benchmark = pd.Series([0.0, 0.01, 0.0], index=months, name="SPY")
    base = BacktestConfig(
        universe=UniverseConfig(
            min_history_quarters=1,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(idea_signal="level"),
    )

    def fake_run_backtest(*args, **kwargs):
        out = pd.Series([0.01, 0.02, 0.03], index=months)
        out.attrs["rebalance_summary"] = pd.DataFrame({"effective_names": [10], "valid_rebalance": [True]})
        return out

    monkeypatch.setattr(sw, "run_backtest", fake_run_backtest)

    sw.grid_eval(
        pd.DataFrame(),
        prices,
        pd.DataFrame(index=months),
        base,
        {("portfolio", "top_n_ideas"): [5, 10]},
        benchmark=benchmark,
        metric="active_sharpe",
        chars=pd.DataFrame(),
        visible_versions_cache={},
        use_selection_cache=False,
        include_returns=True,
        checkpoint_dir=tmp_path,
        checkpoint_every=1,
    )

    grid_path = tmp_path / "sweep_grid_partial.csv"
    returns_path = tmp_path / "sweep_returns_partial.csv"
    assert grid_path.exists()
    assert returns_path.exists()
    assert len(pd.read_csv(grid_path)) == 2
    assert {"config_id", "active_sharpe", "invested_month_frac"}.issubset(pd.read_csv(grid_path).columns)


def test_grid_eval_factor_metric_runs_attribution(monkeypatch):
    months = pd.date_range("2020-01-31", periods=3, freq="ME")
    prices = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=months)
    base = BacktestConfig(
        universe=UniverseConfig(
            min_history_quarters=1,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(idea_signal="level"),
    )
    calls = {"attribution": 0}

    def fake_run_backtest(*args, **kwargs):
        return pd.Series([0.01, 0.02, 0.03], index=months)

    def fake_attribution(*args, **kwargs):
        calls["attribution"] += 1
        return {"sharpe": 1.23, "ann_alpha": 0.04, "alpha_t": 2.0, "ir_vs_benchmark": 0.5}

    monkeypatch.setattr(sw, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(sw, "attribution", fake_attribution)

    grid = sw.grid_eval(
        pd.DataFrame(),
        prices,
        pd.DataFrame(index=months),
        base,
        {("portfolio", "idea_signal"): ["level"]},
        metric="sharpe",
        chars=pd.DataFrame(),
        visible_versions_cache={},
        use_selection_cache=False,
    )

    assert calls["attribution"] == 1
    assert grid.loc[0, "sharpe"] == 1.23
    assert grid.loc[0, "ann_alpha"] == 0.04


def test_marginal_ir_uses_active_ir_without_factor_attribution(monkeypatch):
    months = pd.date_range("2020-01-31", periods=3, freq="ME")
    prices = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=months)
    benchmark = pd.Series([0.0, 0.01, 0.0], index=months, name="SPY")
    cfg = BacktestConfig(
        universe=UniverseConfig(
            min_history_quarters=1,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(idea_signal="level"),
    )

    def fake_run_backtest(*args, **kwargs):
        return pd.Series([0.01, 0.02, 0.03], index=months)

    def fail_attribution(*args, **kwargs):
        raise AssertionError("marginal_ir should not run factor attribution")

    monkeypatch.setattr(en, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(en, "attribution", fail_attribution)

    out = en.marginal_ir(
        pd.DataFrame(),
        prices,
        pd.DataFrame(index=months),
        cfg,
        benchmark=benchmark,
        chars=pd.DataFrame(),
        visible_versions_cache={},
    )

    expected = en._active_ir_metric(pd.Series([0.01, 0.02, 0.03], index=months), benchmark)
    assert out["filter"].tolist() == ["(full stack)"]
    assert out.loc[0, "metric"] == expected


def test_selection_cache_backtest_matches_direct_run():
    months = pd.date_range("2020-01-31", periods=6, freq="ME")
    holdings = pd.DataFrame(
        [
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2019-12-31"),
                "filing_date": pd.Timestamp("2020-01-15"),
                "accession_number": "a1",
                "submission_type": "13F-HR",
                "ticker": "A",
                "value": 100.0,
                "sec_type": "SH",
            },
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-04-15"),
                "accession_number": "a2",
                "submission_type": "13F-HR",
                "ticker": "B",
                "value": 100.0,
                "sec_type": "SH",
            },
        ]
    )
    prices = pd.DataFrame(
        {
            "A": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            "B": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        },
        index=months,
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
        portfolio=PortfolioConfig(
            idea_signal="level",
            min_consensus_funds=1,
            max_name_weight=1.0,
            holding_horizon_q=0,
        ),
    )
    chars = en.manager_characteristics(holdings)
    visible_cache = en.build_visible_versions_cache(chars, prices.index)

    direct = run_backtest(holdings, prices, cfg, chars=chars, visible_versions_cache=visible_cache)
    selected = en.build_rebalance_selection_cache(
        holdings,
        prices,
        cfg,
        chars=chars,
        visible_versions_cache=visible_cache,
    )
    cached = en.run_backtest_from_selection_cache(prices, cfg, selected)

    pd.testing.assert_series_equal(cached, direct)


def test_active_selection_cache_requires_explicit_active_benchmark():
    months = pd.date_range("2020-01-31", periods=3, freq="ME")
    holdings = pd.DataFrame(
        [
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2019-12-31"),
                "filing_date": pd.Timestamp("2020-01-15"),
                "accession_number": "a1",
                "submission_type": "13F-HR",
                "ticker": "A",
                "value": 60.0,
                "sec_type": "SH",
            },
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2019-12-31"),
                "filing_date": pd.Timestamp("2020-01-15"),
                "accession_number": "a1",
                "submission_type": "13F-HR",
                "ticker": "B",
                "value": 40.0,
                "sec_type": "SH",
            },
        ]
    )
    prices = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.0, 0.0, 0.0]}, index=months)
    cfg = BacktestConfig(
        universe=UniverseConfig(
            min_history_quarters=1,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(
            idea_signal="active_weight",
            min_consensus_funds=1,
            max_name_weight=1.0,
            min_active_weight_holdings=1,
        ),
    )
    chars = en.manager_characteristics(holdings)
    visible_cache = en.build_visible_versions_cache(chars, prices.index)
    selected = en.build_rebalance_selection_cache(
        holdings,
        prices,
        cfg,
        chars=chars,
        visible_versions_cache=visible_cache,
    )

    with pytest.raises(ValueError, match="requires PIT active_benchmark_weights"):
        en.run_backtest_from_selection_cache(prices, cfg, selected)


def test_active_selection_cache_backtest_matches_direct_run_with_same_benchmark():
    months = pd.date_range("2020-01-31", periods=4, freq="ME")
    holdings = pd.DataFrame(
        [
            {
                "manager": manager,
                "period_date": pd.Timestamp("2019-12-31"),
                "filing_date": pd.Timestamp("2020-01-15"),
                "accession_number": manager,
                "submission_type": "13F-HR",
                "ticker": ticker,
                "value": value,
                "sec_type": "SH",
            }
            for manager, ticker, value in [
                ("m1", "A", 70.0),
                ("m1", "B", 30.0),
                ("m2", "A", 20.0),
                ("m2", "B", 80.0),
            ]
        ]
    )
    prices = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04], "B": [0.0, 0.0, 0.0, 0.0]}, index=months)
    cfg = BacktestConfig(
        universe=UniverseConfig(
            min_history_quarters=1,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(
            idea_signal="active_weight",
            min_consensus_funds=1,
            max_name_weight=1.0,
            min_active_weight_holdings=1,
        ),
    )
    chars = en.manager_characteristics(holdings)
    visible_cache = en.build_visible_versions_cache(chars, prices.index)
    selected = en.build_rebalance_selection_cache(
        holdings,
        prices,
        cfg,
        chars=chars,
        visible_versions_cache=visible_cache,
    )
    active_cache = en.build_active_benchmark_weights_cache(
        holdings,
        prices,
        chars=chars,
        visible_versions_cache=visible_cache,
    )

    direct = run_backtest(holdings, prices, cfg, chars=chars, visible_versions_cache=visible_cache)
    cached = en.run_backtest_from_selection_cache(prices, cfg, selected, active_cache)

    pd.testing.assert_series_equal(cached, direct)


def test_external_active_benchmark_direct_and_cached_paths_match():
    months = pd.date_range("2020-01-31", periods=4, freq="ME")
    holdings = pd.DataFrame(
        [
            {
                "manager": manager,
                "period_date": pd.Timestamp("2019-12-31"),
                "filing_date": pd.Timestamp("2020-01-15"),
                "accession_number": manager,
                "submission_type": "13F-HR",
                "ticker": ticker,
                "value": value,
                "sec_type": "SH",
            }
            for manager, ticker, value in [
                ("m1", "A", 70.0),
                ("m1", "B", 30.0),
                ("m2", "A", 20.0),
                ("m2", "B", 80.0),
            ]
        ]
    )
    prices = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04], "B": [0.0, 0.0, 0.0, 0.0]}, index=months)
    cfg = BacktestConfig(
        active_benchmark_source="spy_holdings",
        universe=UniverseConfig(
            min_history_quarters=1,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(
            idea_signal="active_weight",
            min_consensus_funds=1,
            max_name_weight=1.0,
            min_active_weight_holdings=1,
        ),
    )
    external = {month: pd.Series({"A": 0.60, "B": 0.40}) for month in months}
    chars = en.manager_characteristics(holdings)
    visible_cache = en.build_visible_versions_cache(chars, prices.index)
    selected = en.build_rebalance_selection_cache(
        holdings,
        prices,
        cfg,
        chars=chars,
        visible_versions_cache=visible_cache,
    )

    direct = run_backtest(
        holdings,
        prices,
        cfg,
        chars=chars,
        visible_versions_cache=visible_cache,
        active_benchmark_weights_by_month=external,
    )
    cached = en.run_backtest_from_selection_cache(prices, cfg, selected, external)

    pd.testing.assert_series_equal(cached, direct)


def test_external_active_benchmark_missing_weights_fails_loudly():
    months = pd.date_range("2020-01-31", periods=2, freq="ME")
    holdings = pd.DataFrame(
        [
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2019-12-31"),
                "filing_date": pd.Timestamp("2020-01-15"),
                "accession_number": "m1",
                "submission_type": "13F-HR",
                "ticker": "A",
                "value": 100.0,
                "sec_type": "SH",
            }
        ]
    )
    prices = pd.DataFrame({"A": [0.01, 0.02]}, index=months)
    cfg = BacktestConfig(
        active_benchmark_source="spy_holdings",
        universe=UniverseConfig(
            min_history_quarters=1,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(idea_signal="active_weight", max_name_weight=1.0, min_active_weight_holdings=1),
    )

    with pytest.raises(ValueError, match="requires active_benchmark_weights_by_month"):
        run_backtest(holdings, prices, cfg)


def test_holding_horizon_is_measured_in_quarters_not_rebalance_events():
    rows = [
        ("m1", "2019-12-31", "2020-01-15", "a1", "A"),
        ("m1", "2020-03-31", "2020-04-15", "a2", "B"),
        ("m2", "2020-03-31", "2020-05-15", "a3", "C"),
        ("m3", "2020-03-31", "2020-06-15", "a4", "D"),
        ("m4", "2020-06-30", "2020-07-15", "a5", "E"),
    ]
    holdings = pd.DataFrame(
        [
            {
                "manager": manager,
                "period_date": pd.Timestamp(period),
                "filing_date": pd.Timestamp(filing),
                "accession_number": accession,
                "submission_type": "13F-HR",
                "ticker": ticker,
                "value": 100.0,
                "sec_type": "SH",
            }
            for manager, period, filing, accession, ticker in rows
        ]
    )
    prices = pd.DataFrame(
        0.0,
        index=pd.date_range("2020-01-31", "2020-07-31", freq="ME"),
        columns=list("ABCDE"),
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
        portfolio=PortfolioConfig(top_n_ideas=1, min_consensus_funds=1, max_name_weight=1.0, holding_horizon_q=2),
    )

    trace = rebalance_trace(holdings, prices, cfg)
    july_holdings = trace["holdings"][trace["holdings"]["rebalance_month"].eq("2020-07-31")]

    assert "A" in july_holdings["ticker"].tolist()
    assert bool(july_holdings.loc[july_holdings["ticker"].eq("A"), "is_carried"].iat[0]) is True


def test_min_portfolio_names_marks_rebalance_invalid_and_moves_to_cash():
    months = pd.date_range("2020-01-31", periods=3, freq="ME")
    holdings = pd.DataFrame(
        [
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2019-12-31"),
                "filing_date": pd.Timestamp("2020-01-15"),
                "accession_number": "a1",
                "submission_type": "13F-HR",
                "ticker": "A",
                "value": 100.0,
                "sec_type": "SH",
            }
        ]
    )
    prices = pd.DataFrame({"A": [0.10, 0.10, 0.10]}, index=months)
    cfg = BacktestConfig(
        universe=UniverseConfig(
            min_history_quarters=1,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(
            top_n_ideas=1,
            min_consensus_funds=1,
            min_portfolio_names=2,
            max_name_weight=1.0,
        ),
    )

    ret = run_backtest(holdings, prices, cfg, capture_rebalance=True)
    trace = rebalance_trace(holdings, prices, cfg)

    assert ret.eq(0.0).all()
    assert bool(ret.attrs["rebalance_summary"].loc[0, "valid_rebalance"]) is False
    assert "target_names_below_min_portfolio_names" in ret.attrs["rebalance_summary"].loc[0, "note"]
    assert trace["summary"].loc[0, "effective_names"] == 0
    assert trace["holdings"].empty


def test_stale_manager_versions_are_excluded_at_later_rebalances():
    months = pd.date_range("2020-01-31", "2021-02-28", freq="ME")
    holdings = pd.DataFrame(
        [
            {
                "manager": "stopped",
                "period_date": pd.Timestamp("2019-12-31"),
                "filing_date": pd.Timestamp("2020-01-15"),
                "accession_number": "old",
                "submission_type": "13F-HR",
                "ticker": "OLD",
                "value": 100.0,
                "sec_type": "SH",
            },
            {
                "manager": "fresh",
                "period_date": pd.Timestamp("2020-12-31"),
                "filing_date": pd.Timestamp("2021-02-15"),
                "accession_number": "new",
                "submission_type": "13F-HR",
                "ticker": "NEW",
                "value": 100.0,
                "sec_type": "SH",
            },
        ]
    )
    prices = pd.DataFrame(0.0, index=months, columns=["OLD", "NEW"])
    cfg = BacktestConfig(
        universe=UniverseConfig(
            min_history_quarters=1,
            max_stale_filing_months=6,
            max_stale_period_months=6,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(top_n_ideas=1, min_consensus_funds=1, max_name_weight=1.0),
    )

    trace = rebalance_trace(holdings, prices, cfg)
    last_summary = trace["summary"].iloc[-1]
    last_holdings = trace["holdings"][trace["holdings"]["rebalance_month"].eq("2021-02-28")]

    assert last_summary["visible_managers"] == 2
    assert last_summary["stale_managers_dropped"] == 1
    assert last_summary["stale_filing_managers"] == 1
    assert last_summary["stale_period_managers"] == 1
    assert last_summary["selected_managers"] == 1
    assert last_holdings["ticker"].tolist() == ["NEW"]


def test_active_benchmark_cache_excludes_stale_manager_versions():
    months = pd.date_range("2020-01-31", "2021-02-28", freq="ME")
    holdings = pd.DataFrame(
        [
            {
                "manager": "stopped",
                "period_date": pd.Timestamp("2019-12-31"),
                "filing_date": pd.Timestamp("2020-01-15"),
                "accession_number": "old",
                "submission_type": "13F-HR",
                "ticker": "OLD",
                "value": 100.0,
                "sec_type": "SH",
            },
            {
                "manager": "fresh",
                "period_date": pd.Timestamp("2020-12-31"),
                "filing_date": pd.Timestamp("2021-02-15"),
                "accession_number": "new",
                "submission_type": "13F-HR",
                "ticker": "NEW",
                "value": 100.0,
                "sec_type": "SH",
            },
        ]
    )
    prices = pd.DataFrame(0.0, index=months, columns=["OLD", "NEW"])
    cfg = BacktestConfig(
        universe=UniverseConfig(
            min_history_quarters=1,
            max_stale_filing_months=6,
            max_stale_period_months=6,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(idea_signal="active_weight", max_name_weight=1.0),
    )
    chars = en.manager_characteristics(holdings)
    visible_cache = en.build_visible_versions_cache(chars, prices.index)

    active_cache = en.build_active_benchmark_weights_cache(
        holdings,
        prices,
        chars=chars,
        visible_versions_cache=visible_cache,
        cfg=cfg,
    )

    assert active_cache[pd.Timestamp("2021-02-28")].index.tolist() == ["NEW"]


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
    assert summary["max_weight"] == pytest.approx(1.0)
    assert summary["top5_weight"] == pytest.approx(1.0)
    assert summary["top10_weight"] == pytest.approx(1.0)
    assert summary["effective_number"] == pytest.approx(1.0)
    assert summary["traded_names"] == 1
    assert summary["buy_names"] == 1
    assert summary["sell_names"] == 0
    assert summary["increased_names"] == 1
    assert summary["decreased_names"] == 0
    assert trace["holdings"]["ticker"].tolist() == ["A"]
    assert set(trace["managers"]["manager"]) == {"m1", "m2"}


def test_rebalance_trace_reports_issuer_groups_and_multi_class_exposure():
    holdings = pd.DataFrame(
        [
            {
                "manager": "m1",
                "period_date": pd.Timestamp("2020-03-31"),
                "filing_date": pd.Timestamp("2020-01-15"),
                "accession_number": "a1",
                "submission_type": "13F-HR",
                "ticker": ticker,
                "value": value,
                "sec_type": "SH",
            }
            for ticker, value in [
                ("GOOG", 40.0),
                ("GOOGL", 30.0),
                ("MSFT", 20.0),
                ("AMZN", 10.0),
            ]
        ]
    )
    prices = pd.DataFrame(
        {ticker: [0.0] for ticker in ["GOOG", "GOOGL", "MSFT", "AMZN"]},
        index=pd.to_datetime(["2020-01-31"]),
    )
    groups = pd.Series({"GOOG": "ALPHABET", "GOOGL": "ALPHABET"})
    cfg = BacktestConfig(
        universe=UniverseConfig(
            min_history_quarters=1,
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(top_n_ideas=4, max_name_weight=0.50, max_issuer_weight=0.40),
    )

    trace = rebalance_trace(holdings, prices, cfg, security_groups=groups)
    summary = trace["summary"].iloc[0]
    held = trace["holdings"].set_index("ticker")

    assert summary["max_issuer_weight"] <= 0.40 + 1e-12
    assert summary["issuer_groups"] == 3
    assert bool(summary["name_cap_feasible"]) is True
    assert bool(summary["issuer_cap_feasible"]) is True
    assert "ALPHABET" in summary["multi_class_exposures"]
    assert held.loc["GOOG", "issuer_group"] == "ALPHABET"
    assert held.loc["GOOGL", "issuer_group"] == "ALPHABET"
    assert held.loc[["GOOG", "GOOGL"], "weight"].sum() <= 0.40 + 1e-12


def test_load_security_groups_uses_override_file(tmp_path):
    path = tmp_path / "security_overrides.csv"
    path.write_text(
        "ticker,issuer_group,asset_type,note\n"
        "GOOG,ALPHABET,common_stock,Alphabet Class C\n"
        "GOOGL,ALPHABET,common_stock,Alphabet Class A\n",
        encoding="utf-8",
    )

    groups = load_security_groups(["GOOG", "GOOGL", "MSFT"], path)

    assert groups.loc["GOOG"] == "ALPHABET"
    assert groups.loc["GOOGL"] == "ALPHABET"
    assert groups.loc["MSFT"] == "MSFT"


def test_rebalance_summary_stats_reports_portfolio_and_turnover_summary():
    summary = pd.DataFrame(
        [
            {
                "rebalance_month": "2020-01-31",
                "effective_names": 10,
                "target_names": 8,
                "carried_names": 2,
                "turnover_one_way": 0.25,
                "cost_bps": 3.75,
                "max_weight": 0.10,
                "issuer_groups": 10,
                "top5_weight": 0.45,
                "top10_weight": 0.80,
                "effective_number": 9.5,
                "traded_names": 6,
                "buy_names": 4,
                "sell_names": 2,
                "name_cap_feasible": True,
                "issuer_cap_feasible": True,
                "top_holdings": "A:10.00%",
            },
            {
                "rebalance_month": "2020-02-29",
                "effective_names": 12,
                "target_names": 10,
                "carried_names": 1,
                "turnover_one_way": 0.40,
                "cost_bps": 6.0,
                "max_weight": 0.08,
                "issuer_groups": 12,
                "top5_weight": 0.40,
                "top10_weight": 0.75,
                "effective_number": 11.0,
                "traded_names": 5,
                "buy_names": 3,
                "sell_names": 1,
                "name_cap_feasible": True,
                "issuer_cap_feasible": False,
                "top_holdings": "B:8.00%",
            },
        ]
    )

    stats = _rebalance_summary_stats(summary)

    assert stats["rebalance_months"] == 2
    assert stats["avg_effective_names"] == pytest.approx(11)
    assert stats["max_turnover_one_way"] == pytest.approx(0.40)
    assert stats["avg_max_weight"] == pytest.approx(0.09)
    assert stats["avg_issuer_groups"] == pytest.approx(11)
    assert stats["name_cap_feasible_ratio"] == pytest.approx(1.0)
    assert stats["issuer_cap_feasible_ratio"] == pytest.approx(0.5)
    assert stats["last_rebalance_month"] == "2020-02-29"
    assert stats["last_top_holdings"] == "B:8.00%"


def test_value_unit_continuity_diagnostics_flags_large_cutoff_jump():
    chars = pd.DataFrame(
        [
            {
                "manager": "BERKSHIRE",
                "period_date": pd.Timestamp("2022-09-30"),
                "filing_date": pd.Timestamp("2022-11-14"),
                "accession_number": "a1",
                "aum": 100.0,
            },
            {
                "manager": "BERKSHIRE",
                "period_date": pd.Timestamp("2022-12-31"),
                "filing_date": pd.Timestamp("2023-02-14"),
                "accession_number": "a2",
                "aum": 100000.0,
            },
        ]
    )

    out = value_unit_continuity_diagnostics(chars)

    assert len(out) == 1
    assert out.loc[0, "manager"] == "BERKSHIRE"
    assert bool(out.loc[0, "suspicious_unit_jump"]) is True


def test_mapping_diagnostics_reports_unmapped_value():
    holdings = pd.DataFrame(
        {
            "cusip": ["111111111", "222222222"],
            "value": [75.0, 25.0],
            "sec_type": ["SH", "SH"],
            "share_amount_type": ["SH", "SH"],
        }
    )
    cmap = {"111111111": "AAA"}

    diag = mapping_diagnostics(holdings, cmap)
    mapped = map_holdings_to_tickers(holdings, cmap, strict=False)

    assert diag["cusips_mapped"] == 1
    assert diag["cusips_unmapped"] == 1
    assert diag["value_coverage"] == 0.75
    assert diag["price_candidate_value_coverage"] == 0.75
    assert diag["by_sec_type"]["SH"]["value_coverage"] == 0.75
    assert diag["top_unmapped_by_value"][0]["cusip"] == "222222222"
    assert mapped["ticker"].tolist() == ["AAA"]


def test_openfigi_selector_normalizes_share_class_ticker_for_yfinance():
    data = [
        {
            "ticker": "BRK/B",
            "marketSector": "Equity",
            "exchCode": "US",
            "securityType2": "Common Stock",
        }
    ]

    assert _select_openfigi_ticker(data) == "BRK-B"
    assert _is_yfinance_ticker("BRK/B")


def test_openfigi_id_type_uses_cins_for_foreign_cusip_like_ids():
    assert _openfigi_id_type("G5960L103") == "ID_CINS"
    assert _openfigi_id_type("N6596X109") == "ID_CINS"
    assert _openfigi_id_type("084670702") == "ID_CUSIP"


def test_openfigi_cache_invalidates_legacy_negative_rows(tmp_path):
    cache_path = tmp_path / "openfigi.parquet"
    pd.DataFrame(
        [
            {"cusip": "084670702", "ticker": None},
            {"cusip": "02079K305", "ticker": "GOOGL"},
        ]
    ).to_parquet(cache_path, index=False)

    loaded = _load_openfigi_cache(cache_path)

    assert loaded == {"02079K305": "GOOGL"}
    _write_openfigi_cache(cache_path, {"084670702": None})
    loaded = _load_openfigi_cache(cache_path)
    assert loaded == {"084670702": None}


def test_openfigi_metadata_cache_drives_equity_only_filter(tmp_path):
    cache_path = tmp_path / "openfigi.parquet"
    _write_openfigi_cache(
        cache_path,
        {
            "037833100": {
                "ticker": "AAPL",
                "name": "APPLE INC",
                "marketSector": "Equity",
                "exchCode": "US",
                "securityType2": "Common Stock",
            },
            "464287200": {
                "ticker": "IVV",
                "name": "ISHARES CORE S&P 500 ETF",
                "marketSector": "Equity",
                "exchCode": "US",
                "securityType2": "ETF",
            },
        },
    )
    cmap = _load_openfigi_cache(cache_path)
    metadata = load_openfigi_metadata(cache_path)
    holdings = pd.DataFrame(
        {
            "cusip": ["037833100", "464287200"],
            "issuer": ["APPLE INC", "BLACKROCK"],
            "sec_type": ["SH", "SH"],
            "value": [100.0, 50.0],
        }
    )

    mapped = map_holdings_to_tickers(holdings, cmap, openfigi_metadata=metadata)
    filtered = priceable_holdings(mapped, exclude_fund_like=True)

    assert cmap == {"037833100": "AAPL", "464287200": "IVV"}
    assert bool(metadata.set_index("cusip").loc["464287200", "is_fund_like"]) is True
    assert filtered["ticker"].tolist() == ["AAPL"]
    assert filtered.attrs["price_filter_diagnostics"]["rows_fund_like_dropped"] == 1


def test_cusip_to_ticker_uses_cins_id_type_and_normalizes_ticker(monkeypatch, tmp_path):
    requests_seen = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [
                {
                    "data": [
                        {
                            "ticker": "BRK/B",
                            "marketSector": "Equity",
                            "exchCode": "US",
                            "securityType2": "Common Stock",
                        }
                    ]
                },
                {
                    "data": [
                        {
                            "ticker": "MDT",
                            "marketSector": "Equity",
                            "exchCode": "US",
                            "securityType2": "Common Stock",
                        }
                    ]
                },
            ]

    def fake_post(url, json, headers, timeout):
        requests_seen.append(json)
        return FakeResponse()

    monkeypatch.setattr(da.requests, "post", fake_post)
    monkeypatch.setattr(da.time, "sleep", lambda *_: None)

    cache_path = tmp_path / "openfigi.parquet"
    out = cusip_to_ticker(["G5960L103", "084670702"], cache_path=cache_path)

    jobs = requests_seen[0]
    metadata = load_openfigi_metadata(cache_path).set_index("cusip")
    assert [job["idType"] for job in jobs] == ["ID_CUSIP", "ID_CINS"]
    assert out == {"084670702": "BRK-B", "G5960L103": "MDT"}
    assert metadata.loc["084670702", "ticker"] == "BRK-B"
    assert metadata.loc["084670702", "securityType2"] == "Common Stock"
    assert pd.notna(metadata.loc["084670702", "metadata_version"])


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


def test_priceable_holdings_can_exclude_fund_like_rows(tmp_path):
    exclusion_path = tmp_path / "funds.csv"
    pd.DataFrame({"ticker": ["CUSTOM"], "reason": ["test_fund"]}).to_csv(exclusion_path, index=False)
    holdings = pd.DataFrame(
        {
            "ticker": ["AAPL", "SPY", "IVV", "VEU", "CUSTOM", "MSFT"],
            "issuer": [
                "APPLE INC",
                "SPDR S&P 500 ETF TR",
                "ISHARES TR",
                "VANGUARD INTL EQUITY INDEX FDS",
                "CUSTOM TRUST",
                "MICROSOFT CORP",
            ],
            "sec_type": ["SH", "SH", "SH", "SH", "SH", "SH"],
            "value": [100.0, 20.0, 30.0, 35.0, 40.0, 50.0],
        }
    )

    filtered = priceable_holdings(
        holdings,
        exclude_fund_like=True,
        fund_ticker_exclusions_path=exclusion_path,
    )

    assert filtered["ticker"].tolist() == ["AAPL", "MSFT"]
    diag = filtered.attrs["price_filter_diagnostics"]
    assert diag["exclude_fund_like"] is True
    assert diag["rows_fund_like_dropped"] == 4
    assert diag["value_fund_like_dropped"] == 125.0
    assert {"CUSTOM", "IVV", "SPY", "VEU"}.issubset(set(diag["tickers_fund_like_dropped_sample"]))


def test_load_benchmark_weight_table_normalizes_tickers_and_percent_weights(tmp_path):
    path = tmp_path / "spy_weights.csv"
    pd.DataFrame(
        {
            "month_end": ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"],
            "ticker": ["BRK.B", "AAPL", "BRK/B", "AAPL"],
            "weight": ["2.5%", "7.5%", "3.0%", "7.0%"],
        }
    ).to_csv(path, index=False)

    table = da.load_benchmark_weight_table(path)
    weights = da.benchmark_weights_by_month(
        table,
        pd.to_datetime(["2020-01-31", "2020-02-29"]),
        max_stale_days=45,
    )

    assert set(table["ticker"]) == {"AAPL", "BRK-B"}
    assert weights[pd.Timestamp("2020-01-31")].sum() == pytest.approx(1.0)
    assert weights[pd.Timestamp("2020-01-31")].loc["BRK-B"] == pytest.approx(0.25)


def test_benchmark_weights_by_month_rejects_uncovered_history(tmp_path):
    path = tmp_path / "spy_weights.csv"
    pd.DataFrame(
        {
            "month_end": ["2020-02-29"],
            "ticker": ["AAPL"],
            "weight": [1.0],
        }
    ).to_csv(path, index=False)
    table = da.load_benchmark_weight_table(path)

    with pytest.raises(ValueError, match="does not cover requested months"):
        da.benchmark_weights_by_month(table, pd.to_datetime(["2020-01-31", "2020-02-29"]))


def test_benchmark_weights_by_month_allows_recent_prior_month_by_default(tmp_path):
    path = tmp_path / "spy_weights.csv"
    pd.DataFrame(
        {
            "month_end": ["2020-01-31", "2020-03-31"],
            "ticker": ["AAPL", "AAPL"],
            "weight": [1.0, 1.0],
        }
    ).to_csv(path, index=False)
    table = da.load_benchmark_weight_table(path)

    weights = da.benchmark_weights_by_month(
        table,
        pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
        max_stale_days=45,
    )

    assert weights[pd.Timestamp("2020-02-29")].loc["AAPL"] == pytest.approx(1.0)


def test_benchmark_weights_by_month_rejects_old_prior_snapshot(tmp_path):
    path = tmp_path / "spy_weights.csv"
    pd.DataFrame(
        {
            "month_end": ["2020-01-31"],
            "ticker": ["AAPL"],
            "weight": [1.0],
        }
    ).to_csv(path, index=False)
    table = da.load_benchmark_weight_table(path)

    with pytest.raises(ValueError, match="2020-03-31"):
        da.benchmark_weights_by_month(
            table,
            pd.to_datetime(["2020-03-31"]),
            max_stale_days=45,
        )


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


def test_attribution_drops_nan_factor_rows_before_ols():
    idx = pd.date_range("2020-01-31", periods=14, freq="ME")
    ret = pd.Series([0.01] * 14, index=idx)
    factors = pd.DataFrame(
        {
            "RF": [0.001] * 14,
            "MKT": [0.01] * 14,
            "SMB": [0.0] * 14,
            "HML": [0.0] * 14,
            "RMW": [0.0] * 14,
            "CMA": [0.0] * 14,
            "MOM": [0.0] * 14,
        },
        index=idx,
    )
    factors.loc[idx[2], "MOM"] = np.nan
    factors.loc[idx[4], "SMB"] = np.inf

    out = attribution(ret, factors)

    assert out["n_months"] == 14
    assert out["factor_months_used"] == 12
    assert "ann_alpha" in out


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


def test_price_cache_overwrites_stale_ticker_history(tmp_path):
    cache_path = tmp_path / "prices.parquet"
    stale = pd.DataFrame(
        {"SPY": [600.0, 610.0]},
        index=pd.to_datetime(["2025-01-31", "2025-02-28"]),
    )
    fresh = pd.DataFrame(
        {"SPY": [200.0, 220.0, 660.0]},
        index=pd.to_datetime(["2015-01-31", "2015-02-28", "2026-03-31"]),
    )

    _write_price_cache(cache_path, stale)
    _write_price_cache(cache_path, fresh)
    loaded = _load_price_cache(cache_path, "2015-01-01", "2026-03-31")

    assert loaded["SPY"].dropna().index.min() == pd.Timestamp("2015-01-31")
    assert loaded["SPY"].dropna().index.max() == pd.Timestamp("2026-03-31")
    assert loaded.loc[pd.Timestamp("2015-01-31"), "SPY"] == 200.0


def test_fetch_prices_refetches_cached_ticker_when_cache_does_not_cover_window(monkeypatch, tmp_path):
    cache_path = tmp_path / "prices.parquet"
    _write_price_cache(
        cache_path,
        pd.DataFrame({"AAPL": [100.0, 101.0]}, index=pd.to_datetime(["2025-01-31", "2025-02-28"])),
    )
    requested_dates = pd.to_datetime(["2015-01-31", "2015-02-28", "2026-03-31"])
    calls = []

    def fake_yf_download_close(yf, batch, start, end):
        calls.append((tuple(batch), start, end))
        return pd.DataFrame({"AAPL": [50.0, 55.0, 110.0]}, index=requested_dates)

    monkeypatch.setattr(da, "_yf_download_close", fake_yf_download_close)

    returns = da.fetch_prices(
        ["AAPL"],
        "2015-01-01",
        "2026-03-31",
        batch_size=1,
        max_retries=0,
        cache_path=cache_path,
        use_chart_fallback=False,
    )

    assert calls == [(("AAPL",), "2015-01-01", "2026-03-31")]
    assert returns.attrs["price_diagnostics"]["tickers_from_cache"] == 0
    assert returns.attrs["price_diagnostics"]["tickers_refetched_due_to_incomplete_cache"] == 1
    assert "AAPL" in returns.columns


def test_fetch_prices_full_window_refetches_short_cache_even_with_coverage_metadata(monkeypatch, tmp_path):
    cache_path = tmp_path / "prices.parquet"
    _write_price_cache(
        cache_path,
        pd.DataFrame({"SPY": [600.0, 610.0]}, index=pd.to_datetime(["2025-01-31", "2025-02-28"])),
    )
    da._write_price_coverage(cache_path, ["SPY"], "2015-01-01", "2026-03-31", "fetched")
    requested_dates = pd.to_datetime(["2015-01-31", "2015-02-28", "2026-03-31"])
    calls = []

    def fake_yf_download_close(yf, batch, start, end):
        calls.append((tuple(batch), start, end))
        return pd.DataFrame({"SPY": [200.0, 220.0, 660.0]}, index=requested_dates)

    monkeypatch.setattr(da, "_yf_download_close", fake_yf_download_close)

    returns = da.fetch_prices(
        ["SPY"],
        "2015-01-01",
        "2026-03-31",
        batch_size=1,
        max_retries=0,
        cache_path=cache_path,
        require_full_window=True,
    )
    loaded = _load_price_cache(cache_path, "2015-01-01", "2026-03-31")

    assert calls == [(("SPY",), "2015-01-01", "2026-03-31")]
    assert returns.attrs["price_diagnostics"]["tickers_from_cache"] == 0
    assert returns.attrs["price_diagnostics"]["tickers_refetched_due_to_incomplete_cache"] == 1
    assert loaded["SPY"].dropna().index.min() == pd.Timestamp("2015-01-31")
    assert loaded.loc[pd.Timestamp("2015-01-31"), "SPY"] == 200.0


def test_fetch_prices_refetches_short_cache_even_with_coverage_metadata_by_default(monkeypatch, tmp_path):
    cache_path = tmp_path / "prices.parquet"
    _write_price_cache(
        cache_path,
        pd.DataFrame({"AAPL": [600.0, 610.0]}, index=pd.to_datetime(["2025-01-31", "2025-02-28"])),
    )
    da._write_price_coverage(cache_path, ["AAPL"], "2015-01-01", "2026-03-31", "fetched")
    requested_dates = pd.to_datetime(["2015-01-31", "2015-02-28", "2026-03-31"])
    calls = []

    def fake_yf_download_close(yf, batch, start, end):
        calls.append((tuple(batch), start, end))
        return pd.DataFrame({"AAPL": [20.0, 22.0, 220.0]}, index=requested_dates)

    monkeypatch.setattr(da, "_yf_download_close", fake_yf_download_close)

    returns = da.fetch_prices(
        ["AAPL"],
        "2015-01-01",
        "2026-03-31",
        batch_size=1,
        max_retries=0,
        cache_path=cache_path,
    )
    loaded = _load_price_cache(cache_path, "2015-01-01", "2026-03-31")

    assert calls == [(("AAPL",), "2015-01-01", "2026-03-31")]
    assert returns.attrs["price_diagnostics"]["tickers_from_cache"] == 0
    assert returns.attrs["price_diagnostics"]["tickers_refetched_due_to_incomplete_cache"] == 1
    assert returns.attrs["price_diagnostics"]["tickers_refetched_due_to_false_coverage"] == 1
    assert loaded["AAPL"].dropna().index.min() == pd.Timestamp("2015-01-31")
    assert loaded.loc[pd.Timestamp("2015-01-31"), "AAPL"] == 20.0


def test_fetch_prices_reuses_trusted_partial_history_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "prices.parquet"
    partial = pd.DataFrame(
        {"AAAU": [10.0, 11.0, 12.0]},
        index=pd.to_datetime(["2018-08-31", "2018-09-30", "2026-03-31"]),
    )
    _write_price_cache(cache_path, partial)
    da._write_price_coverage(cache_path, ["AAAU"], "2015-01-01", "2026-03-31", "fetched", partial)
    calls = []

    def fake_yf_download_close(yf, batch, start, end):
        calls.append((tuple(batch), start, end))
        return pd.DataFrame()

    monkeypatch.setattr(da, "_yf_download_close", fake_yf_download_close)

    returns = da.fetch_prices(
        ["AAAU"],
        "2015-01-01",
        "2026-03-31",
        batch_size=1,
        max_retries=0,
        cache_path=cache_path,
        use_chart_fallback=False,
    )

    assert calls == []
    assert "AAAU" in returns.columns
    assert returns.attrs["price_diagnostics"]["tickers_from_cache"] == 1
    assert returns.attrs["price_diagnostics"]["tickers_from_trusted_partial_cache"] == 1
    assert returns.attrs["price_diagnostics"]["tickers_refetched_due_to_incomplete_cache"] == 0


def test_fetch_prices_does_not_use_partial_cache_when_full_window_required(monkeypatch, tmp_path):
    cache_path = tmp_path / "prices.parquet"
    partial = pd.DataFrame(
        {"SPY": [600.0, 610.0]},
        index=pd.to_datetime(["2025-01-31", "2026-03-31"]),
    )
    _write_price_cache(cache_path, partial)
    da._write_price_coverage(cache_path, ["SPY"], "2015-01-01", "2026-03-31", "fetched", partial)
    requested_dates = pd.to_datetime(["2015-01-31", "2015-02-28", "2026-03-31"])
    calls = []

    def fake_yf_download_close(yf, batch, start, end):
        calls.append((tuple(batch), start, end))
        return pd.DataFrame({"SPY": [200.0, 220.0, 660.0]}, index=requested_dates)

    monkeypatch.setattr(da, "_yf_download_close", fake_yf_download_close)

    returns = da.fetch_prices(
        ["SPY"],
        "2015-01-01",
        "2026-03-31",
        batch_size=1,
        max_retries=0,
        cache_path=cache_path,
        require_full_window=True,
    )

    assert calls == [(("SPY",), "2015-01-01", "2026-03-31")]
    assert returns.attrs["price_diagnostics"]["tickers_from_trusted_partial_cache"] == 0
    assert returns.attrs["price_diagnostics"]["tickers_refetched_due_to_incomplete_cache"] == 1


def test_fetch_prices_refetches_stale_partial_history_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "prices.parquet"
    stale_partial = pd.DataFrame(
        {"AAAU": [10.0, 11.0]},
        index=pd.to_datetime(["2018-08-31", "2025-12-31"]),
    )
    _write_price_cache(cache_path, stale_partial)
    da._write_price_coverage(cache_path, ["AAAU"], "2015-01-01", "2026-03-31", "fetched", stale_partial)
    calls = []

    def fake_yf_download_close(yf, batch, start, end):
        calls.append((tuple(batch), start, end))
        return pd.DataFrame(
            {"AAAU": [10.0, 11.0, 12.0]},
            index=pd.to_datetime(["2018-08-31", "2026-02-28", "2026-03-31"]),
        )

    monkeypatch.setattr(da, "_yf_download_close", fake_yf_download_close)

    returns = da.fetch_prices(
        ["AAAU"],
        "2015-01-01",
        "2026-03-31",
        batch_size=1,
        max_retries=0,
        cache_path=cache_path,
        use_chart_fallback=False,
    )

    assert calls == [(("AAAU",), "2015-01-01", "2026-03-31")]
    assert returns.attrs["price_diagnostics"]["tickers_from_trusted_partial_cache"] == 0
    assert returns.attrs["price_diagnostics"]["tickers_refetched_due_to_incomplete_cache"] == 1


def test_fetch_prices_patches_partial_yfinance_history_with_chart(monkeypatch, tmp_path):
    cache_path = tmp_path / "prices.parquet"
    yf_dates = pd.to_datetime(["2025-01-31", "2025-02-28"])
    chart_dates = pd.to_datetime(["2015-01-31", "2015-02-28", "2026-03-31"])
    chart_calls = []

    def fake_yf_download_close(yf, batch, start, end):
        return pd.DataFrame({"AAPL": [600.0, 610.0]}, index=yf_dates)

    def fake_chart_guarded(batch, start, end, *, max_workers, timeout_seconds):
        chart_calls.append((tuple(batch), start, end))
        return pd.DataFrame({"AAPL": [20.0, 22.0, 220.0]}, index=chart_dates)

    monkeypatch.setattr(da, "_yf_download_close", fake_yf_download_close)
    monkeypatch.setattr(da, "_yahoo_chart_download_close_guarded", fake_chart_guarded)

    returns = da.fetch_prices(
        ["AAPL"],
        "2015-01-01",
        "2026-03-31",
        batch_size=1,
        max_retries=0,
        cache_path=cache_path,
    )
    loaded = _load_price_cache(cache_path, "2015-01-01", "2026-03-31")

    assert chart_calls == [(("AAPL",), "2015-01-01", "2026-03-31")]
    assert returns.attrs["price_diagnostics"]["used_chart_fallback"] is True
    assert returns.attrs["price_diagnostics"]["partial_history_patched_tickers"] == ["AAPL"]
    assert loaded["AAPL"].dropna().index.min() == pd.Timestamp("2015-01-31")
    assert loaded.loc[pd.Timestamp("2015-01-31"), "AAPL"] == 20.0


def test_price_cache_audit_flags_false_full_coverage(tmp_path):
    cache_path = tmp_path / "prices.parquet"
    _write_price_cache(
        cache_path,
        pd.DataFrame({"AAPL": [600.0, 610.0]}, index=pd.to_datetime(["2025-01-31", "2025-02-28"])),
    )
    da._write_price_coverage(cache_path, ["AAPL"], "2015-01-01", "2026-03-31", "fetched")

    audit = da.audit_price_cache_coverage(cache_path, "2015-01-01", "2026-03-31", ["AAPL"])

    row = audit.iloc[0]
    assert row["ticker"] == "AAPL"
    assert row["actual_first_close"] == pd.Timestamp("2025-01-31")
    assert row["actual_spans_requested_window"] == False
    assert row["coverage_status"] == "fetched"
    assert row["false_full_coverage"] == True


def test_fetch_prices_uses_chart_fallback_when_yfinance_probe_is_empty(monkeypatch, tmp_path):
    dates = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])

    def fake_yf_download_close(yf, batch, start, end):
        return pd.DataFrame()

    def fake_yfinance_probe(yf, start, end, timeout_seconds=None):
        return "probe_empty_close"

    def fake_chart_probe(start, end, timeout_seconds=None):
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


def test_fetch_prices_switches_to_chart_fallback_after_yfinance_timeout(monkeypatch, tmp_path):
    dates = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])
    chart_batches = []

    def fake_yf_download_close_guarded(yf, batch, start, end, timeout_seconds):
        raise TimeoutError(f"yfinance batch timed out after {timeout_seconds}s: {batch[0]}..{batch[-1]}")

    def fake_chart_probe(start, end, timeout_seconds=None):
        return "chart_probe_ok=SPY,IWD,AAPL"

    def fake_chart_download_close(batch, start, end, *, max_workers=8):
        chart_batches.append(tuple(batch))
        data = {ticker: [100.0, 110.0, 121.0] for ticker in batch}
        return pd.DataFrame(data, index=dates)

    monkeypatch.setattr(da, "_yf_download_close_guarded", fake_yf_download_close_guarded)
    monkeypatch.setattr(da, "_yahoo_chart_probe", fake_chart_probe)
    monkeypatch.setattr(da, "_yahoo_chart_download_close", fake_chart_download_close)

    returns = da.fetch_prices(
        ["AAPL", "MSFT", "NVDA"],
        "2025-01-01",
        "2025-03-31",
        batch_size=2,
        max_retries=0,
        cache_path=tmp_path / "prices.parquet",
        yfinance_batch_timeout_seconds=1,
    )

    assert chart_batches == [("AAPL", "MSFT"), ("NVDA",)]
    assert returns.columns.tolist() == ["AAPL", "MSFT", "NVDA"]
    assert returns.attrs["price_diagnostics"]["used_chart_fallback"] is True
    assert returns.attrs["price_diagnostics"]["yfinance_batch_timeout_seconds"] == 1
    assert "timed out" in returns.attrs["price_diagnostics"]["failed_batches"][0]


def test_yahoo_chart_fallback_splits_timed_out_batches(monkeypatch, tmp_path):
    dates = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])
    calls = []

    def fake_yf_download_close_guarded(yf, batch, start, end, timeout_seconds):
        return pd.DataFrame()

    def fake_yfinance_probe(yf, start, end, timeout_seconds=None):
        return "probe_empty_close"

    def fake_chart_probe(start, end, timeout_seconds=None):
        return "chart_probe_ok=SPY,IWD,AAPL"

    def fake_chart_guarded(batch, start, end, *, max_workers, timeout_seconds):
        calls.append(tuple(batch))
        if len(batch) > 1:
            raise TimeoutError(f"Yahoo Chart batch timed out after {timeout_seconds}s: {batch[0]}..{batch[-1]}")
        return pd.DataFrame({batch[0]: [100.0, 110.0, 121.0]}, index=dates)

    monkeypatch.setattr(da, "_yf_download_close_guarded", fake_yf_download_close_guarded)
    monkeypatch.setattr(da, "_yfinance_probe", fake_yfinance_probe)
    monkeypatch.setattr(da, "_yahoo_chart_probe", fake_chart_probe)
    monkeypatch.setattr(da, "_yahoo_chart_download_close_guarded", fake_chart_guarded)

    returns = da.fetch_prices(
        ["AAPL", "MSFT"],
        "2025-01-01",
        "2025-03-31",
        batch_size=2,
        max_retries=0,
        max_consecutive_empty_batches=1,
        cache_path=tmp_path / "prices.parquet",
        chart_fallback_batch_timeout_seconds=1,
    )

    assert calls == [("AAPL", "MSFT"), ("AAPL",), ("MSFT",)]
    assert returns.columns.tolist() == ["AAPL", "MSFT"]
    assert returns.attrs["price_diagnostics"]["used_chart_fallback"] is True
    assert returns.attrs["price_diagnostics"]["chart_fallback_batch_timeout_seconds"] == 1


def test_fetch_prices_keeps_partial_monthly_return_columns(monkeypatch):
    dates = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30"])

    def fake_yf_download_close(yf, batch, start, end):
        return pd.DataFrame(
            {
                "AAPL": [100.0, 110.0, 121.0, 133.1],
                "MSFT": [100.0, 110.0, np.nan, 121.0],
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

    assert returns.columns.tolist() == ["AAPL", "MSFT"]
    assert returns["MSFT"].isna().any()
    assert returns.attrs["price_diagnostics"]["tickers_with_close"] == 2
    assert returns.attrs["price_diagnostics"]["tickers_with_complete_returns"] == 1
    assert returns.attrs["price_diagnostics"]["tickers_with_partial_returns"] == 1
    assert returns.attrs["price_diagnostics"]["tickers_no_returns_dropped"] == 0


def test_deflated_sharpe_reports_trials_when_oos_is_insufficient():
    out = deflated_sharpe(pd.Series([0.01] * 6), n_trials=16)

    assert out["note"] == "insufficient OOS"
    assert out["T"] == 6
    assert out["n_trials"] == 16


def test_interactive_results_writes_self_contained_html(tmp_path):
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    grid = pd.DataFrame(
        [
            {
                "config_id": "aum_band=0.5-5B|idea_signal=active_weight",
                "aum_band": "0.5-5B",
                "idea_signal": "active_weight",
                "top_n_ideas": 3,
                "min_consensus_funds": 2,
                "holding_horizon_q": 0,
                "min_portfolio_names": 10,
                "max_portfolio_names": 25,
                "valid_config": True,
                "invested_month_frac": 0.95,
                "avg_effective_names": 24.0,
                "avg_max_weight": 0.05,
                "name_cap_feasible_ratio": 1.0,
                "ann_return": 0.12,
                "ann_vol": 0.20,
                "max_drawdown": -0.05,
                "active_sharpe": 0.8,
            }
        ]
    )
    returns = {"aum_band=0.5-5B|idea_signal=active_weight": pd.Series([0.01, -0.02, 0.03], index=idx)}
    benchmark = pd.Series([0.0, 0.01, 0.0], index=idx, name="SPY")
    path = tmp_path / "interactive.html"

    out = rp.interactive_results(grid, returns, benchmark, path=str(path))

    assert out == str(path)
    text = path.read_text(encoding="utf-8")
    assert "13F Parameter Sweep Results" in text
    assert "active_weight" in text
    assert "growth" in text
    assert "splitter" in text
    assert "valid_config" in text
    assert "invested_month_frac" in text
