"""
Run a 13F-clone research example.

Default mode is synthetic and requires no network. Live mode builds a rule-based
SEC 13F universe, maps CUSIPs to tickers, downloads prices/factors, and then runs
the same engine/sweep/report stack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from engine import (
    BacktestConfig,
    PortfolioConfig,
    UniverseConfig,
    _needs_active_benchmark_weights,
    _needs_idiosyncratic_vol,
    attribution,
    build_idiosyncratic_vol_cache,
    build_visible_versions_cache,
    manager_characteristics,
    marginal_ir,
    raw_filing_put_weights,
    rebalance_trace,
    run_backtest,
)
from manager_classifier import (
    ManagerClassifierConfig,
    build_manager_classification,
    classification_summary,
    config_hash as manager_classifier_config_hash,
    load_manager_overrides,
    override_file_hash as manager_override_file_hash,
)
from market_cap import fetch_market_cap_history, load_market_cap_table, market_caps_by_month
from report import dashboard, interactive_results, single_config_result_grid
from run_diagnostics import (
    manager_characteristics_audit,
    print_rebalance_summary as _print_rebalance_summary,
    rebalance_summary_stats as _rebalance_summary_stats,
    trace_core_diagnostics as _trace_core_diagnostics,
    value_unit_continuity_diagnostics,
    write_manager_filter_acceptance,
)
from runtime_support import (
    config_hash as _config_hash,
    frame_shallow_mb as _frame_shallow_mb,
    git_sha as _git_sha,
    json_default as _json_default,
    load_local_env as _load_local_env,
    process_rss_mb as _process_rss_mb,
    progress_printer as _progress_printer,
    write_manifest,
)
from sweep import active_return_stream, deflated_sharpe, grid_eval, walk_forward


LIVE_CONFIG = {
    "identity": "YourName you@firm.com",
    "openfigi_key": None,
    "sec_history_start": "2013-10-01",
    "price_history_start": "2014-01-01",
    "start": "2015-01-01",
    "end": "2026-05-31",
    # Broad-market total-return proxy. Use QQQ for a tighter growth-style proxy.
    "benchmark_ticker": "SPY",
    # Raw SEC ingest must cover the thesis band plus the 15-30B comparison band.
    "min_aum": 0.1e9,
    "max_aum": 30e9,
    "max_holdings": 40,
    "max_put_weight": 0.10,
    "require_factors": False,
    "openfigi_cache_path": "openfigi_cache.parquet",
    "price_cache_path": "yfinance_close_cache.parquet",
    "price_source": "chart",
    "security_overrides_path": "data/security_overrides.csv",
    "exclude_fund_like_holdings": True,
    "fund_ticker_exclusions_path": "data/fund_ticker_exclusions.csv",
    "manager_overrides_path": "data/manager_overrides.csv",
    "manager_classification_cache_dir": "data/processed",
    "refresh_openfigi_metadata": False,
    "force_refresh_openfigi": False,
    "active_benchmark_source": "manager_held_mcap",
    "active_benchmark_weights_path": "data/processed/benchmark_weights_spy.parquet",
    "active_benchmark_max_stale_days": 45,
    "market_cap_cache_path": "data/processed/market_cap_history.parquet",
    "market_cap_auto_download": True,
    "market_cap_max_stale_days": 45,
    "market_cap_shares_max_stale_days": 550,
    "market_cap_batch_size": 25,
    "market_cap_workers": 6,
    "market_cap_request_timeout": 20,
    "idio_vol_cache_dir": "data/processed",
    "idio_vol_window_months": 24,
    "idio_vol_min_obs": 12,
    "idio_vol_floor": 0.10,
    "idio_vol_cap": 0.80,
    "idio_vol_winsor_lower": 0.05,
    "idio_vol_winsor_upper": 0.95,
}


def load_security_groups(tickers, path: str | pathlib.Path | None = "data/security_overrides.csv") -> pd.Series:
    clean = pd.Index([str(t).strip().upper() for t in tickers if pd.notna(t)])
    groups = pd.Series(clean, index=clean, dtype="string")
    if path is None:
        return groups.astype(str)
    p = pathlib.Path(path)
    if not p.exists():
        print(f"  [warn] security overrides not found: {p}; issuer groups default to ticker")
        return groups.astype(str)
    overrides = pd.read_csv(p)
    required = {"ticker", "issuer_group"}
    missing = required.difference(overrides.columns)
    if missing:
        raise ValueError(f"security overrides missing columns: {sorted(missing)}")
    overrides = overrides.dropna(subset=["ticker", "issuer_group"]).copy()
    overrides["ticker"] = overrides["ticker"].astype(str).str.strip().str.upper()
    overrides["issuer_group"] = overrides["issuer_group"].astype(str).str.strip().str.upper()
    mapped = overrides.set_index("ticker")["issuer_group"]
    common = groups.index.intersection(mapped.index)
    groups.loc[common] = mapped.loc[common].astype(str)
    multi_groups = mapped[mapped.isin(mapped[mapped.duplicated(keep=False)])].nunique()
    print(
        "  security overrides: "
        f"{len(common)}/{len(groups)} active tickers mapped to issuer groups "
        f"({multi_groups} multi-ticker groups in override file)"
    )
    return groups.astype(str)


def build_synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    print("[1/6] Building synthetic data (no network)")
    rng = np.random.default_rng(7)
    periods = pd.date_range("2014-03-31", "2023-12-31", freq="QE")
    months = pd.date_range("2014-01-31", "2024-06-30", freq="ME")
    n_names = 250
    tickers = [f"S{i:03d}" for i in range(n_names)]

    factors = pd.DataFrame(index=months)
    for col, (mu, sd) in {
        "MKT": (0.006, 0.04),
        "SMB": (0.001, 0.02),
        "HML": (0.001, 0.02),
        "RMW": (0.001, 0.015),
        "CMA": (0.0, 0.015),
        "MOM": (0.002, 0.03),
    }.items():
        factors[col] = rng.normal(mu, sd, len(months))
    factors["RF"] = 0.001

    betas = pd.DataFrame(
        {
            "MKT": rng.normal(1.0, 0.2, n_names),
            "HML": rng.normal(0.0, 0.5, n_names),
            "SMB": rng.normal(0.0, 0.4, n_names),
        },
        index=tickers,
    )
    value_rank = pd.Series(betas["HML"].rank(pct=True), index=tickers)

    prices = pd.DataFrame(
        (factors[["MKT", "SMB", "HML"]].values @ betas[["MKT", "SMB", "HML"]].T.values)
        + rng.normal(0, 0.05, (len(months), n_names)),
        index=months,
        columns=tickers,
    )

    mktcap = pd.Series(rng.pareto(1.5, n_names) + 1, index=tickers)
    bench_w = mktcap / mktcap.sum()
    bench_ret = factors["RF"] + 0.6 * factors["HML"] + 0.9 * factors["MKT"]
    bench_ret.name = "synthetic_value_benchmark"
    value_scores = pd.DataFrame({p: value_rank for p in periods}).T

    def make_manager(name: str, good: bool) -> list[dict]:
        rows: list[dict] = []
        held = set(
            (
                rng.choice(np.where(value_rank > 0.55)[0], rng.integers(8, 15), replace=False)
                if good
                else rng.choice(n_names, rng.integers(60, 90), replace=False)
            ).tolist()
        )
        for p in periods:
            if good:
                if rng.random() < 0.5 and len(held) > 6:
                    held.discard(rng.choice(list(held)))
                    held.add(int(rng.choice(np.where(value_rank > 0.55)[0])))
                aum = rng.uniform(2e9, 20e9)
            else:
                drop = set(rng.choice(list(held), max(1, len(held) // 3), replace=False))
                held -= drop
                held |= set(rng.choice(n_names, len(drop), replace=False).tolist())
                aum = rng.uniform(1e9, 25e9)
            vals = rng.pareto(2.0, len(held)) + 0.2
            accession = f"{name}-{p:%Y%m%d}"
            for ti, v in zip(sorted(held), vals):
                rows.append(
                    {
                        "manager": name,
                        "period_date": p,
                        "filing_date": p + pd.Timedelta(days=44),
                        "accession_number": accession,
                        "submission_type": "13F-HR",
                        "ticker": f"S{ti:03d}",
                        "value": float(v) * aum / vals.sum(),
                        "sec_type": "SH",
                    }
                )
        return rows

    holdings = pd.DataFrame(
        sum([make_manager(f"GOOD{i:02d}", True) for i in range(8)], [])
        + sum([make_manager(f"BAD{i:02d}", False) for i in range(8)], [])
    )
    holdings.attrs["raw_mapped_holdings"] = holdings.copy()
    holdings.attrs["raw_filing_put_weights"] = raw_filing_put_weights(holdings)

    good_top = (
        holdings[holdings.manager.str.startswith("GOOD")]
        .sort_values("value", ascending=False)
        .groupby(["manager", "period_date"])
        .head(8)
    )
    for r in good_top.itertuples():
        fwd = months[(months > r.filing_date) & (months <= r.filing_date + pd.Timedelta(days=185))]
        if len(fwd) and r.ticker in prices.columns:
            prices.loc[fwd, r.ticker] += 0.004

    print(
        f"    managers: {holdings.manager.nunique()}, "
        f"filings: {holdings.groupby(['manager', 'period_date']).ngroups}, "
        f"months: {len(months)}, tickers: {n_names}"
    )
    return holdings, prices, factors, value_scores, bench_w, bench_ret, prices.copy()


def _top_cusips_by_value(holdings: pd.DataFrame, limit: int | None) -> pd.Index:
    if limit is None or limit <= 0:
        return pd.Index(holdings["cusip"].dropna().unique())
    return holdings.groupby("cusip")["value"].sum().nlargest(limit).index


def build_live_data(
    cfg: dict,
    *,
    cusip_limit: int | None = None,
    price_ticker_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, None, None, pd.Series | None, pd.DataFrame]:
    print("[1/6] Building rule-based universe from SEC 13F datasets")
    import build_universe as bu
    import data_adapters as da

    cfg = dict(cfg)
    cfg["openfigi_key"] = cfg.get("openfigi_key") or os.environ.get("OPENFIGI_API_KEY")

    holdings_start = cfg.get("sec_history_start") or cfg["start"]
    if holdings_start != cfg["start"]:
        print(
            "    SEC holdings history start: "
            f"{holdings_start} (strategy price window starts {cfg['start']})"
        )
    h_cusip = bu.build_holdings_universe(
        holdings_start,
        cfg["end"],
        cfg["identity"],
        cache_dir="13f_cache",
        min_aum=cfg["min_aum"],
        max_aum=cfg["max_aum"],
        max_holdings=cfg["max_holdings"],
        max_put_weight=cfg["max_put_weight"],
    )
    filing_put_weights = raw_filing_put_weights(h_cusip)
    print(
        f"    rule-based pool: {h_cusip['cik'].nunique()} filers, "
        f"{h_cusip.groupby(['cik', 'period_date']).ngroups} filer-periods"
    )
    print("[1/6] Mapping CUSIPs through OpenFIGI")
    target_cusips = _top_cusips_by_value(h_cusip, cusip_limit)
    if cusip_limit:
        h_cusip = h_cusip[h_cusip["cusip"].isin(target_cusips)].copy()
        print(f"    smoke CUSIP subset: top {len(target_cusips)} CUSIPs by disclosed value")
    require_openfigi_metadata = bool(
        cfg.get("refresh_openfigi_metadata", False)
        or cfg.get("exclude_fund_like_holdings", False)
    )
    if require_openfigi_metadata and cfg.get("exclude_fund_like_holdings", False):
        print("    equity-only mode: requiring OpenFIGI metadata for ETF/fund classification")
    cmap = da.cusip_to_ticker(
        target_cusips,
        api_key=cfg["openfigi_key"],
        cache_path=cfg.get("openfigi_cache_path"),
        require_metadata=require_openfigi_metadata,
        force_refresh=bool(cfg.get("force_refresh_openfigi", False)),
    )
    openfigi_metadata = da.load_openfigi_metadata(cfg.get("openfigi_cache_path"))
    if not openfigi_metadata.empty:
        metadata_rows = int(openfigi_metadata["metadata_version"].notna().sum())
        print(
            "  OpenFIGI metadata cache: "
            f"{metadata_rows}/{len(openfigi_metadata)} cached rows have metadata"
        )
    holdings = da.map_holdings_to_tickers(
        h_cusip,
        cmap,
        openfigi_metadata=openfigi_metadata,
    )
    raw_mapped_holdings = holdings.copy()
    holdings = da.priceable_holdings(
        holdings,
        exclude_fund_like=bool(cfg.get("exclude_fund_like_holdings", False)),
        fund_ticker_exclusions_path=cfg.get("fund_ticker_exclusions_path"),
    )
    print("[1/6] Downloading monthly prices from yfinance")
    price_holdings = holdings
    if price_ticker_limit:
        top_tickers = holdings.groupby("ticker")["value"].sum().nlargest(price_ticker_limit).index
        price_holdings = holdings[holdings["ticker"].isin(top_tickers)].copy()
        print(f"    smoke ticker subset: top {len(top_tickers)} tickers by disclosed value")
    price_history_start = cfg.get("price_history_start") or cfg["start"]
    if price_history_start != cfg["start"]:
        print(f"    price/factor warm-up: {price_history_start} -> {cfg['start']} (not included in backtest returns)")
    signal_prices = da.fetch_prices(
        price_holdings.ticker.unique(),
        price_history_start,
        cfg["end"],
        cache_path=cfg.get("price_cache_path"),
        price_source=cfg.get("price_source", "auto"),
    )
    holdings = da.align_holdings_to_prices(price_holdings, signal_prices)
    prices = signal_prices.loc[pd.Timestamp(cfg["start"]):pd.Timestamp(cfg["end"])].copy()
    holdings.attrs["raw_mapped_holdings"] = raw_mapped_holdings
    holdings.attrs["raw_filing_put_weights"] = filing_put_weights
    print("[1/6] Downloading Fama-French factors")
    try:
        factors = da.fetch_factors(price_history_start, cfg["end"])
        print(f"    factors: {len(factors)} monthly rows")
    except Exception as exc:
        if cfg.get("require_factors"):
            raise
        print(f"    [warn] {exc}")
        print("    [warn] continuing without factor regression; verify factor source/dependency for FF attribution")
        factors = pd.DataFrame(index=prices.index)
        factors.attrs["factor_diagnostics"] = {"available": False, "reason": str(exc)}
    print(f"[1/6] Downloading benchmark prices: {cfg['benchmark_ticker']}")
    try:
        bench_ret = da.fetch_prices(
            [cfg["benchmark_ticker"]],
            cfg["start"],
            cfg["end"],
            cache_path=cfg.get("price_cache_path"),
            price_source=cfg.get("price_source", "auto"),
            require_full_window=True,
        ).iloc[:, 0]
        bench_ret.name = cfg["benchmark_ticker"]
    except Exception as exc:
        print(f"    [warn] benchmark fetch failed; benchmark disabled: {exc}")
        bench_ret = None
    print(f"    managers: {holdings.manager.nunique()}, tickers: {holdings.ticker.nunique()}")
    return holdings, prices, factors, None, None, bench_ret, signal_prices


def run_live_smoke(
    output_root: pathlib.Path,
    *,
    cusip_limit: int,
    ticker_limit: int,
    cfg: dict | None = None,
) -> pathlib.Path:
    cfg = dict(LIVE_CONFIG if cfg is None else cfg)
    holdings, prices, factors, _, _, bench_ret, _ = build_live_data(
        cfg,
        cusip_limit=cusip_limit,
        price_ticker_limit=ticker_limit,
    )
    payload = {
        "mode": "live-smoke",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "config": {
            "start": cfg["start"],
            "end": cfg["end"],
            "cusip_limit": cusip_limit,
            "ticker_limit": ticker_limit,
            "benchmark_ticker": cfg["benchmark_ticker"],
            "exclude_fund_like_holdings": bool(cfg.get("exclude_fund_like_holdings", False)),
            "fund_ticker_exclusions_path": cfg.get("fund_ticker_exclusions_path"),
            "refresh_openfigi_metadata": bool(cfg.get("refresh_openfigi_metadata", False)),
        },
        "input_summary": {
            "holdings_rows": int(len(holdings)),
            "manager_count": int(holdings["manager"].nunique()),
            "ticker_count": int(holdings["ticker"].nunique()),
            "price_months": int(len(prices)),
            "price_columns": int(len(prices.columns)),
            "factor_months": int(len(factors)),
            "benchmark_available": bench_ret is not None,
            "mapping_diagnostics": holdings.attrs.get("mapping_diagnostics"),
            "price_filter_diagnostics": holdings.attrs.get("price_filter_diagnostics"),
            "price_alignment_diagnostics": holdings.attrs.get("price_alignment_diagnostics"),
            "price_diagnostics": prices.attrs.get("price_diagnostics"),
            "factor_diagnostics": factors.attrs.get("factor_diagnostics"),
        },
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"live_smoke_{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    print(f"  Saved live smoke diagnostics: {path}")
    return path


def _load_active_benchmark_weights_by_month(
    *,
    live_config: dict,
    months,
    tickers,
    cfg: BacktestConfig,
    mode: str,
) -> tuple[dict[pd.Timestamp, pd.Series] | None, dict[str, Any]]:
    if not _needs_active_benchmark_weights(cfg.portfolio.idea_signal):
        return None, {"required": False}
    if cfg.active_benchmark_source in {"visible_13f_aggregate", "13f_aggregate"}:
        return None, {"required": True, "source": cfg.active_benchmark_source, "external_table": False}
    if cfg.active_benchmark_source == "manager_held_mcap":
        if mode == "synthetic":
            ordered = sorted(map(str, tickers))
            synthetic_caps = pd.Series(
                {ticker: float(len(ordered) - i) for i, ticker in enumerate(ordered)},
                dtype=float,
            )
            print("    active benchmark: manager_held_mcap synthetic fixture")
            weights = {pd.Timestamp(month): synthetic_caps.copy() for month in months}
            return weights, {
                "required": True,
                "source": "manager_held_mcap",
                "market_cap_source": "synthetic_fixture",
                "strict_pit_row_fraction": 1.0,
            }
        path = pathlib.Path(live_config.get("market_cap_cache_path", "data/processed/market_cap_history.parquet"))
        if bool(live_config.get("market_cap_auto_download", True)):
            table = fetch_market_cap_history(
                tickers,
                pd.Timestamp(min(months)).to_period("M").start_time.date().isoformat(),
                pd.Timestamp(max(months)).date().isoformat(),
                cache_path=path,
                batch_size=int(live_config.get("market_cap_batch_size", 25)),
                max_workers=int(live_config.get("market_cap_workers", 6)),
                request_timeout=int(live_config.get("market_cap_request_timeout", 20)),
                max_shares_stale_days=int(live_config.get("market_cap_shares_max_stale_days", 550)),
            )
        else:
            table = load_market_cap_table(path)
        caps = market_caps_by_month(
            table,
            months,
            max_stale_days=int(live_config.get("market_cap_max_stale_days", 45)),
        )
        coverage = pd.Series({month: len(values) for month, values in caps.items()}, dtype=float)
        strict_frac = float(table.get("strict_pit", pd.Series(False, index=table.index)).mean())
        print(
            "    active benchmark: manager_held_mcap "
            f"months={len(caps)}, median_tickers={coverage.median():.0f}, "
            f"strict_pit_rows={strict_frac:.1%}, source={','.join(sorted(table['source'].astype(str).unique()))}, "
            f"cache={path}"
        )
        return caps, {
            "required": True,
            "source": "manager_held_mcap",
            "market_cap_path": str(path),
            "market_cap_sources": sorted(table["source"].astype(str).unique().tolist()),
            "strict_pit_row_fraction": strict_frac,
            "months": int(len(caps)),
            "zero_coverage_months": int((coverage == 0).sum()),
            "min_tickers": int(coverage.min()) if len(coverage) else 0,
            "median_tickers": float(coverage.median()) if len(coverage) else 0.0,
            "max_tickers": int(coverage.max()) if len(coverage) else 0,
        }
    import data_adapters as da

    path = pathlib.Path(live_config.get("active_benchmark_weights_path", ""))
    max_stale_days = int(live_config.get("active_benchmark_max_stale_days", 0))
    table = da.load_benchmark_weight_table(path)
    weights = da.benchmark_weights_by_month(table, months, max_stale_days=max_stale_days)
    print(
        f"    active benchmark: {cfg.active_benchmark_source} "
        f"{len(table['month_end'].drop_duplicates())} snapshots, "
        f"{len(weights)} monthly weights loaded from {path}"
    )
    return weights, {
        "required": True,
        "source": cfg.active_benchmark_source,
        "weights_path": str(path),
        "snapshots": int(table["month_end"].nunique()),
        "months": int(len(weights)),
    }


def _preflight_active_benchmark_inputs(*, mode: str, live_config: dict, cfg: BacktestConfig | None) -> None:
    if mode != "live" or cfg is None:
        return
    if not _needs_active_benchmark_weights(cfg.portfolio.idea_signal):
        return
    if cfg.active_benchmark_source in {"visible_13f_aggregate", "13f_aggregate", "manager_held_mcap"}:
        return
    path = pathlib.Path(live_config.get("active_benchmark_weights_path", ""))
    if path.exists():
        return
    raise FileNotFoundError(
        "active benchmark weight file is required before live data processing starts: "
        f"{path}\n"
        "The repository cannot safely auto-generate historical SPY constituent weights; "
        "using a current snapshot for historical months would introduce look-ahead bias.\n"
        "Provide a PIT monthly file with columns month_end,ticker,weight, or run explicitly with "
        "--active-benchmark-source visible_13f_aggregate for the old 13F-aggregate proxy."
    )


def _strategy_rule_summary(cfg: BacktestConfig, *, value_scores, benchmark_weights, security_groups=None) -> dict:
    return {
        "rebalance_rule": (
            "At each month-end on or after a visible SEC filing date, select each "
            "manager's latest non-stale filing version available by filing_date and rebalance."
        ),
        "point_in_time_rule": (
            "Only filings with filing_date <= rebalance_month are visible; latest visible "
            "manager books older than max_stale_filing_months or max_stale_period_months are excluded."
        ),
        "execution_timing": (
            "Existing holdings earn the current month's return first; new target "
            "weights are set at month-end for subsequent months."
        ),
        "transaction_cost": {
            "model": "one_way_turnover * bps_per_side",
            "bps_per_side": cfg.cost.bps_per_side,
        },
        "universe": asdict(cfg.universe),
        "manager_filter_mode": cfg.manager_filter_mode,
        "portfolio": asdict(cfg.portfolio),
        "portfolio_construction": {
            "idea_selection": "rank within each manager; signal magnitude is not an allocation weight",
            "manager_equal": (
                "equal budget per contributing manager; allocate each manager budget across selected "
                "Top-K names in proportion to reported book weights"
            ),
            "max_portfolio_names_note": (
                "The thesis Top-30 aggregate cutoff is an operational convenience and a documented "
                "deviation from strict paper-style replication."
                if cfg.portfolio.max_portfolio_names == 30
                else None
            ),
        },
        "security_grouping": {
            "enabled": security_groups is not None,
            "max_issuer_weight": cfg.portfolio.max_issuer_weight,
            "unmapped_ticker_policy": "issuer_group defaults to ticker",
        },
        "missing_price_policy": cfg.missing_price_policy,
        "active_filter_status": {
            "concentration_configured": bool(cfg.universe.use_concentration),
            "concentration_active": bool(cfg.universe.use_concentration),
            "value_tilt_configured": bool(cfg.universe.use_value_tilt),
            "value_tilt_active": bool(cfg.universe.use_value_tilt and value_scores is not None),
            "active_share_configured": bool(cfg.universe.use_active_share),
            "active_share_active": bool(cfg.universe.use_active_share and benchmark_weights is not None),
            "hedge_filter_configured": bool(cfg.universe.use_hedge_filter),
            "hedge_put_weight_source": "raw SEC filing before CUSIP mapping and equity filtering",
            "active_weight_signal": _needs_active_benchmark_weights(cfg.portfolio.idea_signal),
            "active_weight_benchmark": (
                cfg.active_benchmark_source
                if _needs_active_benchmark_weights(cfg.portfolio.idea_signal)
                else None
            ),
            "cps_ir_signal": _needs_idiosyncratic_vol(cfg.portfolio.idea_signal),
            "cps_ir_method_note": (
                "positive active overweight times PIT trailing CAPM residual vol; "
                "24m/default floor/cap/winsor settings are heuristic guardrails"
            ),
        },
    }


def _dashboard_parameter_summary(
    *,
    mode: str,
    cfg: BacktestConfig,
    prices: pd.DataFrame,
    holdings: pd.DataFrame,
    benchmark,
    axes: dict,
    train_m: int,
    test_m: int,
    security_groups: pd.Series,
) -> dict[str, str]:
    u = cfg.universe
    p = cfg.portfolio
    c = cfg.cost
    benchmark_name = getattr(benchmark, "name", None) if benchmark is not None else None
    sweep_axis_text = "; ".join(
        f"{scope}.{field}={list(values)}"
        for (scope, field), values in axes.items()
    )
    return {
        "Data": (
            f"mode={mode}, returns={prices.index.min().date()}..{prices.index.max().date()}, "
            f"months={len(prices)}, managers={holdings['manager'].nunique()}, "
            f"tickers={len(prices.columns)}, benchmark={benchmark_name or 'disabled'}"
        ),
        "Universe": (
            f"min_aum={u.min_aum:.0f}, max_aum={u.max_aum:.0f}, "
            f"manager_filter={cfg.manager_filter_mode}, "
            f"use_concentration={u.use_concentration}, max_holdings={u.max_holdings}, "
            f"min_top{u.top_n_concentration}_weight={u.min_top_n_weight:.0%}, "
            f"turnover_q={u.turnover_quantile}, min_history_q={u.min_history_quarters}, "
            f"hedge_put_max={u.hedge_put_max_weight:.0%}, value_tilt_min={u.value_tilt_min_pctl:.0%}"
        ),
        "Portfolio": (
            f"idea_signal={p.idea_signal}, top_n_ideas={p.top_n_ideas}, "
            f"idea_aggregation={p.idea_aggregation or ('score' if p.consensus_weight else 'manager_count')}, "
            f"min_consensus_funds={p.min_consensus_funds}, holding_horizon_q={p.holding_horizon_q}, "
            f"max_portfolio_names={p.max_portfolio_names}, "
            f"max_name={p.max_name_weight:.1%}, max_issuer={p.max_issuer_weight:.1%}, "
            f"min_portfolio_names={p.min_portfolio_names}, "
            f"min_active_weight_holdings={p.min_active_weight_holdings}, "
            f"active_benchmark={cfg.active_benchmark_source}, "
            f"cps_idio_vol=CAPM 24m min_obs=12 heuristic floor/cap/winsor"
        ),
        "Execution/cost": (
            f"rebalance=month-end after filing_date visibility, missing_price_policy={cfg.missing_price_policy}, "
            f"cost={c.bps_per_side:.1f}bps per one-way turnover"
        ),
        "Validation": (
            f"walk_forward=train {train_m}m/test {test_m}m, select_on=active_sharpe, "
            f"sweep_trials={int(np.prod([len(v) for v in axes.values()]))}"
        ),
        "Sweep axes": sweep_axis_text,
        "Security grouping": (
            f"issuer_groups={security_groups.nunique()}, override_unmapped_policy=ticker_as_issuer_group"
        ),
    }


def write_rebalance_outputs(
    out_dir: pathlib.Path,
    label: str,
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    *,
    value_scores=None,
    benchmark_weights=None,
    chars=None,
    visible_versions_cache=None,
    security_groups=None,
    active_benchmark_weights_by_month=None,
    idiosyncratic_vol_by_month=None,
    manager_classification=None,
    manager_overrides=None,
) -> dict[str, Any]:
    trace = rebalance_trace(
        holdings,
        prices,
        cfg,
        value_scores=value_scores,
        benchmark_weights=benchmark_weights,
        chars=chars,
        visible_versions_cache=visible_versions_cache,
        security_groups=security_groups,
        active_benchmark_weights_by_month=active_benchmark_weights_by_month,
        manager_classification=manager_classification,
        manager_overrides=manager_overrides,
        idiosyncratic_vol_by_month=idiosyncratic_vol_by_month,
    )
    outputs: dict[str, str] = {}
    for name, df in trace.items():
        filename = (
            "rebalance_manager_candidates_audit.csv"
            if name == "manager_candidates_audit" and label == "thesis"
            else f"rebalance_{name}_{label}.csv"
        )
        path = out_dir / filename
        df.to_csv(path, index=False)
        outputs[name] = str(path)

    rules_path = out_dir / f"rebalance_rules_{label}.json"
    rules_path.write_text(
        json.dumps(
            _strategy_rule_summary(
                cfg,
                value_scores=value_scores,
                benchmark_weights=benchmark_weights,
                security_groups=security_groups,
            ),
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    outputs["rules"] = str(rules_path)
    outputs["summary_stats"] = _rebalance_summary_stats(trace["summary"])
    return outputs


def write_sweep_outputs(
    out_dir: pathlib.Path,
    grid: pd.DataFrame,
    returns_by_config_id: dict[str, pd.Series] | None,
    benchmark: pd.Series | None,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    grid_path = out_dir / "sweep_grid.csv"
    grid.to_csv(grid_path, index=False)
    outputs["grid"] = str(grid_path)

    returns_by_config_id = returns_by_config_id or {}
    rows = []
    for config_id, ret in returns_by_config_id.items():
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
        if ret.empty:
            continue
        bench = benchmark.reindex(ret.index).replace([np.inf, -np.inf], np.nan) if benchmark is not None else None
        active = ret - bench if bench is not None else ret
        growth = (1 + ret.fillna(0.0)).cumprod()
        bench_growth = (1 + bench.fillna(0.0)).cumprod() if bench is not None else pd.Series(index=ret.index, dtype=float)
        active_growth = (1 + active.fillna(0.0)).cumprod()
        dd = growth / growth.cummax() - 1
        for date, value in ret.items():
            rows.append({
                "config_id": config_id,
                "date": pd.Timestamp(date).date().isoformat(),
                "return": float(value),
                "benchmark_return": float(bench.loc[date]) if bench is not None and pd.notna(bench.loc[date]) else np.nan,
                "active_return": float(active.loc[date]) if pd.notna(active.loc[date]) else np.nan,
                "growth_of_one": float(growth.loc[date]),
                "benchmark_growth_of_one": float(bench_growth.loc[date]) if bench is not None and pd.notna(bench_growth.loc[date]) else np.nan,
                "active_growth_of_one": float(active_growth.loc[date]),
                "drawdown": float(dd.loc[date]),
            })
    returns_path = out_dir / "sweep_returns.csv"
    pd.DataFrame(rows).to_csv(returns_path, index=False)
    outputs["returns"] = str(returns_path)

    html_path = out_dir / "interactive_results.html"
    outputs["interactive_html"] = interactive_results(
        grid,
        returns_by_config_id,
        benchmark=benchmark,
        path=str(html_path),
    )
    return outputs


def _sweep_config_id_for_cfg(grid: pd.DataFrame, axes: dict, cfg: BacktestConfig) -> str | None:
    label: dict[str, Any] = {}
    for (scope, field), values in axes.items():
        if scope == "universe" and field == "aum_band":
            match = next(
                (
                    value
                    for value in values
                    if float(value[1]) == float(cfg.universe.min_aum)
                    and float(value[2]) == float(cfg.universe.max_aum)
                ),
                None,
            )
            if match is None:
                return None
            label[field] = match[0]
            continue
        owner = cfg.universe if scope == "universe" else cfg.portfolio if scope == "portfolio" else cfg
        label[field] = getattr(owner, field)
    config_id = "|".join(f"{key}={value}" for key, value in sorted(label.items()))
    if "config_id" not in grid or not grid["config_id"].astype(str).eq(config_id).any():
        return None
    return config_id


def _file_hash(path: str | pathlib.Path) -> str | None:
    p = pathlib.Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _hash_frame_for_cache(df: pd.DataFrame, *, columns: list[str] | None = None) -> str:
    if df is None or df.empty:
        return hashlib.sha256(b"empty").hexdigest()[:16]
    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        work = df.loc[:, keep]
    else:
        work = df
    schema_payload = json.dumps(
        [(str(col), str(work[col].dtype)) for col in work.columns],
        separators=(",", ":"),
    ).encode("utf-8")
    # Sorting compact uint64 row hashes is substantially cheaper than sorting a
    # 500k-row object frame, while keeping the cache key row-order independent.
    row_hashes = pd.util.hash_pandas_object(work, index=False, categorize=True).to_numpy(dtype="uint64")
    row_hashes.sort()
    return hashlib.sha256(schema_payload + row_hashes.tobytes()).hexdigest()[:16]


def _hash_matrix_for_cache(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return hashlib.sha256(b"empty").hexdigest()[:16]
    work = df.copy()
    work.index = pd.to_datetime(work.index)
    work.columns = [str(c) for c in work.columns]
    work = work.sort_index()
    work = work.reindex(sorted(work.columns), axis=1)
    labels = json.dumps(
        {
            "index": [pd.Timestamp(x).isoformat() for x in work.index],
            "columns": list(map(str, work.columns)),
        },
        sort_keys=True,
    ).encode("utf-8")
    row_hash = pd.util.hash_pandas_object(work, index=True).values.tobytes()
    return hashlib.sha256(labels + row_hash).hexdigest()[:16]


def _manager_classification_cache_key(
    *,
    raw_holdings: pd.DataFrame,
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    cfg: ManagerClassifierConfig,
    progress=None,
) -> tuple[str, dict[str, Any]]:
    holding_cols = [
        "manager",
        "manager_name",
        "period_date",
        "filing_date",
        "accession_number",
        "ticker",
        "issuer",
        "value",
        "sec_type",
        "is_fund_like",
    ]
    if progress is not None:
        progress("classification cache key: hashing raw holdings")
    raw_holdings_hash = _hash_frame_for_cache(raw_holdings, columns=holding_cols)
    if progress is not None:
        progress("classification cache key: hashing filtered holdings")
    filtered_holdings_hash = _hash_frame_for_cache(holdings, columns=holding_cols)
    if progress is not None:
        progress("classification cache key: hashing monthly returns")
    prices_hash = _hash_matrix_for_cache(prices)
    if progress is not None:
        progress("classification cache key: hashing factors")
    factors_hash = _hash_matrix_for_cache(factors)
    payload = {
        "schema_version": 2,
        "classification_config_hash": manager_classifier_config_hash(cfg),
        "override_file_hash": manager_override_file_hash(cfg.override_path),
        "manager_classifier_py": _file_hash("manager_classifier.py"),
        "engine_py": _file_hash("engine.py"),
        "raw_holdings_hash": raw_holdings_hash,
        "filtered_holdings_hash": filtered_holdings_hash,
        "prices_hash": prices_hash,
        "factors_hash": factors_hash,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")).hexdigest()[:16]
    return key, payload


def _write_manager_classification_cache(
    classification: pd.DataFrame,
    *,
    cache_path: pathlib.Path,
    meta_path: pathlib.Path,
    cache_key_payload: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    classification.to_parquet(cache_path, index=False)
    meta = {
        "cache_key_payload": cache_key_payload,
        "attrs": classification.attrs,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _read_manager_classification_cache(cache_path: pathlib.Path, meta_path: pathlib.Path) -> pd.DataFrame | None:
    if not cache_path.exists() or not meta_path.exists():
        return None
    classification = pd.read_parquet(cache_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    classification.attrs.update(meta.get("attrs", {}))
    return classification


def _signals_need_idiosyncratic_vol(cfgs: list[BacktestConfig], axes: dict | None = None) -> bool:
    for cfg in cfgs:
        if cfg is not None and _needs_idiosyncratic_vol(cfg.portfolio.idea_signal):
            return True
    if axes:
        for (scope, field), values in axes.items():
            if scope == "portfolio" and field == "idea_signal":
                if any(_needs_idiosyncratic_vol(str(v)) for v in values):
                    return True
    return False


def _idio_vol_cache_key(
    *,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    months,
    live_config: dict,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema_version": 1,
        "engine_py": _file_hash("engine.py"),
        "prices_hash": _hash_matrix_for_cache(prices),
        "factors_hash": _hash_matrix_for_cache(factors),
        "months_hash": _hash_matrix_for_cache(pd.DataFrame(index=pd.Index(pd.to_datetime(months)), data={"x": 1})),
        "window_months": int(live_config.get("idio_vol_window_months", 24)),
        "min_obs": int(live_config.get("idio_vol_min_obs", 24)),
        "floor": float(live_config.get("idio_vol_floor", 0.10)),
        "cap": float(live_config.get("idio_vol_cap", 0.80)),
        "winsor_lower": float(live_config.get("idio_vol_winsor_lower", 0.05)),
        "winsor_upper": float(live_config.get("idio_vol_winsor_upper", 0.95)),
        "model": "capm",
        "pit_rule": "uses returns strictly before asof month",
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")).hexdigest()[:16]
    return key, payload


def _idio_vol_cache_to_frame(cache: dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
    rows = []
    for month, vol in sorted(cache.items()):
        s = pd.Series(vol, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        for ticker, value in s.items():
            rows.append({
                "month_end": pd.Timestamp(month).to_period("M").to_timestamp("M"),
                "ticker": str(ticker).upper(),
                "idio_vol": float(value),
            })
    return pd.DataFrame(rows, columns=["month_end", "ticker", "idio_vol"])


def _idio_vol_frame_to_cache(frame: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    if frame is None or frame.empty:
        return {}
    work = frame.copy()
    work["month_end"] = pd.to_datetime(work["month_end"]).dt.to_period("M").dt.to_timestamp("M")
    work["ticker"] = work["ticker"].astype(str).str.upper()
    work["idio_vol"] = pd.to_numeric(work["idio_vol"], errors="coerce")
    work = work.dropna(subset=["month_end", "ticker", "idio_vol"])
    out: dict[pd.Timestamp, pd.Series] = {}
    for month, sub in work.groupby("month_end", sort=True):
        s = sub.groupby("ticker")["idio_vol"].last().astype(float)
        out[pd.Timestamp(month)] = s[s > 0].sort_index()
    return out


def _complete_idio_vol_cache_months(
    cache: dict[pd.Timestamp, pd.Series],
    months,
) -> dict[pd.Timestamp, pd.Series]:
    """Restore requested empty months omitted by long-form Parquet storage."""
    normalized = {
        pd.Timestamp(month).to_period("M").to_timestamp("M"): pd.Series(values, dtype=float)
        for month, values in cache.items()
    }
    for month in pd.Index(pd.to_datetime(months)).to_period("M").to_timestamp("M"):
        normalized.setdefault(pd.Timestamp(month), pd.Series(dtype=float))
    return dict(sorted(normalized.items()))


def _idio_vol_summary(
    cache: dict[pd.Timestamp, pd.Series],
    *,
    prices: pd.DataFrame,
    cache_hit: bool,
    cache_path: pathlib.Path,
    meta_path: pathlib.Path,
    cache_key: str,
    cache_key_payload: dict[str, Any],
    elapsed_sec: float,
) -> dict[str, Any]:
    month_counts = pd.Series({pd.Timestamp(k): len(v) for k, v in cache.items()}, dtype=float)
    requested = int(len(prices.columns))
    avg_coverage = float((month_counts / requested).mean()) if requested and len(month_counts) else 0.0
    return {
        "model": "capm",
        "window_months": int(cache_key_payload["window_months"]),
        "min_obs": int(cache_key_payload["min_obs"]),
        "floor": float(cache_key_payload["floor"]),
        "cap": float(cache_key_payload["cap"]),
        "winsor_lower": float(cache_key_payload["winsor_lower"]),
        "winsor_upper": float(cache_key_payload["winsor_upper"]),
        "method_note": "24m window and vol floor/cap/winsorization are pragmatic guardrails, not academic calibration.",
        "cache_key": cache_key,
        "cache_hit": bool(cache_hit),
        "cache_path": str(cache_path),
        "cache_meta_path": str(meta_path),
        "cache_key_payload": cache_key_payload,
        "elapsed_sec": float(elapsed_sec),
        "months": int(len(cache)),
        "avg_ticker_coverage_frac": avg_coverage,
        "min_tickers": int(month_counts.min()) if len(month_counts) else 0,
        "median_tickers": float(month_counts.median()) if len(month_counts) else 0.0,
        "max_tickers": int(month_counts.max()) if len(month_counts) else 0,
    }


def _idiosyncratic_vol_artifacts(
    *,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    months,
    live_config: dict,
    required: bool,
    progress=None,
) -> tuple[dict[pd.Timestamp, pd.Series] | None, dict[str, Any]]:
    if not required:
        return None, {"required": False, "note": "no CPS-IR signal requested"}
    t0 = time.perf_counter()
    progress = progress or _progress_printer(t0)
    progress(f"idio-vol setup: returns={prices.shape[0]}x{prices.shape[1]}, factors={factors.shape}")
    cache_key, cache_key_payload = _idio_vol_cache_key(
        prices=prices,
        factors=factors,
        months=months,
        live_config=live_config,
    )
    cache_dir = pathlib.Path(live_config.get("idio_vol_cache_dir", "data/processed"))
    cache_path = cache_dir / f"idiosyncratic_vol.{cache_key}.parquet"
    meta_path = cache_dir / f"idiosyncratic_vol.{cache_key}.meta.json"
    cache_hit = cache_path.exists() and meta_path.exists()
    if cache_hit:
        print(f"  idio-vol cache hit: {cache_path}")
        frame = pd.read_parquet(cache_path)
        cache = _idio_vol_frame_to_cache(frame)
    else:
        print(f"  idio-vol cache miss: {cache_path}")
        cache = build_idiosyncratic_vol_cache(
            prices,
            factors,
            months,
            window_months=int(live_config.get("idio_vol_window_months", 24)),
            min_obs=int(live_config.get("idio_vol_min_obs", 24)),
            floor=float(live_config.get("idio_vol_floor", 0.10)),
            cap=float(live_config.get("idio_vol_cap", 0.80)),
            winsor_lower=float(live_config.get("idio_vol_winsor_lower", 0.05)),
            winsor_upper=float(live_config.get("idio_vol_winsor_upper", 0.95)),
            progress=progress,
        )
        progress("idio-vol serializing cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        _idio_vol_cache_to_frame(cache).to_parquet(cache_path, index=False)
        meta_path.write_text(
            json.dumps({"cache_key_payload": cache_key_payload}, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
    cache = _complete_idio_vol_cache_months(cache, months)
    summary = _idio_vol_summary(
        cache,
        prices=prices,
        cache_hit=cache_hit,
        cache_path=cache_path,
        meta_path=meta_path,
        cache_key=cache_key,
        cache_key_payload=cache_key_payload,
        elapsed_sec=time.perf_counter() - t0,
    )
    print(
        "  idio-vol cache: "
        f"months={summary['months']} median_tickers={summary['median_tickers']:.0f} "
        f"avg_coverage={summary['avg_ticker_coverage_frac']:.1%} "
        f"({'hit' if cache_hit else 'built'}) in {summary['elapsed_sec']:.1f}s"
    )
    print("  [info] idio-vol method: 24m CAPM residual vol; floor/cap/winsor settings are heuristic guardrails")
    return cache, summary


def _manager_filter_artifacts(
    *,
    out_dir: pathlib.Path,
    raw_holdings: pd.DataFrame,
    holdings: pd.DataFrame,
    chars: pd.DataFrame,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    visible_versions_cache: dict[pd.Timestamp, pd.DataFrame],
    live_config: dict,
    progress=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cfg = ManagerClassifierConfig(
        override_path=str(live_config.get("manager_overrides_path", "data/manager_overrides.csv"))
    )
    overrides = load_manager_overrides(cfg.override_path)
    t0 = time.perf_counter()
    progress = progress or _progress_printer(t0)
    progress(
        f"classification setup: raw_rows={len(raw_holdings):,}, filtered_rows={len(holdings):,}, "
        f"chars={len(chars):,}, months={len(prices.index)}"
    )
    cache_key, cache_key_payload = _manager_classification_cache_key(
        raw_holdings=raw_holdings,
        holdings=holdings,
        prices=prices,
        factors=factors,
        cfg=cfg,
        progress=progress,
    )
    cache_dir = pathlib.Path(live_config.get("manager_classification_cache_dir", "data/processed"))
    cache_path = cache_dir / f"manager_classification.{cache_key}.parquet"
    meta_path = cache_dir / f"manager_classification.{cache_key}.meta.json"
    classification = _read_manager_classification_cache(cache_path, meta_path)
    cache_hit = classification is not None
    if cache_hit:
        print(f"  manager classification cache hit: {cache_path}")
    else:
        print(f"  manager classification cache miss: {cache_path}")
        classification = build_manager_classification(
            raw_holdings,
            holdings,
            chars,
            prices.index,
            prices,
            factors,
            visible_versions_cache=visible_versions_cache,
            config=cfg,
            progress=progress,
        )
        progress("manager classification serializing parquet cache")
        _write_manager_classification_cache(
            classification,
            cache_path=cache_path,
            meta_path=meta_path,
            cache_key_payload=cache_key_payload,
        )
    path = out_dir / "manager_classification.csv"
    progress(f"manager classification writing audit CSV: {path}")
    classification.to_csv(path, index=False)
    summary = classification_summary(classification)
    summary.update({
        "path": str(path),
        "override_path": cfg.override_path,
        "override_rows": int(len(overrides)),
        "override_file_hash": manager_override_file_hash(cfg.override_path),
        "classification_config": cfg.__dict__,
        "classification_config_hash": manager_classifier_config_hash(cfg),
        "cache_key": cache_key,
        "cache_hit": bool(cache_hit),
        "cache_path": str(cache_path),
        "cache_meta_path": str(meta_path),
        "cache_key_payload": cache_key_payload,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
    })
    print(
        "  manager classification: "
        f"{summary.get('rows', 0)} rows, latest managers={summary.get('latest_managers', 0)}, "
        f"hash={summary.get('classification_hash')} "
        f"({'cache hit' if cache_hit else 'built'}) in {summary['elapsed_sec']:.1f}s"
    )
    print(f"    style counts latest: {summary.get('style_counts_latest', {})}")
    print(f"    dirty reasons latest: {summary.get('dirty_reason_counts_latest', {})}")
    print(f"    classification coverage latest: {summary.get('source_counts_latest', {})}")
    if overrides.empty:
        print(f"    manager overrides: none ({cfg.override_path})")
    else:
        print(f"    manager overrides: {len(overrides)} rows ({cfg.override_path})")
    return classification, overrides, summary




def _default_run_configs() -> tuple[BacktestConfig, BacktestConfig, dict, int, int]:
    cfg_a = BacktestConfig(
        universe=UniverseConfig(
            min_aum=0.1e9,
            max_aum=10e9,
            use_concentration=True,
            min_top_n_weight=0.50,
            max_holdings=LIVE_CONFIG["max_holdings"],
            turnover_quantile=0.34,
            hedge_put_max_weight=0.05,
            value_tilt_min_pctl=0.50,
            min_history_quarters=4,
        ),
        portfolio=PortfolioConfig(
            idea_signal="cps_ir",
            top_n_ideas=3,
            idea_aggregation="score",
            min_consensus_funds=2,
            min_portfolio_names=10,
            max_portfolio_names=30,
            holding_horizon_q=0,
            max_name_weight=0.10,
            max_issuer_weight=0.15,
            min_active_weight_holdings=10,
        ),
        manager_filter_mode="dedicated_like",
        active_benchmark_source="manager_held_mcap",
    )
    cfg_b = BacktestConfig(
        universe=UniverseConfig(
            use_size_band=False,
            use_concentration=False,
            use_low_turnover=False,
            use_hedge_filter=False,
            use_value_tilt=False,
        ),
        portfolio=PortfolioConfig(idea_signal="level", min_consensus_funds=1, holding_horizon_q=0),
        manager_filter_mode="all",
    )
    axes = {
        ("backtest", "manager_filter_mode"): ["dedicated_like"],
        ("universe", "aum_band"): [
            ("0.1-10B", 0.1e9, 10e9),
        ],
        ("portfolio", "idea_signal"): [
            "cps_ir",
            "cps_ir_change",
            "cps_ir_initiation",
        ],
        ("portfolio", "top_n_ideas"): [1, 3, 5],
        ("portfolio", "idea_aggregation"): ["manager_equal", "score"],
        ("portfolio", "min_consensus_funds"): [1, 2],
        ("portfolio", "holding_horizon_q"): [0, 1],
        ("portfolio", "min_portfolio_names"): [10],
        ("portfolio", "max_portfolio_names"): [30],
        ("portfolio", "min_active_weight_holdings"): [10],
        ("universe", "use_concentration"): [True],
        # Keep these fixed in the default Cartesian sweep to avoid exploding
        # runtime; marginal-IR still evaluates their isolated impact.
        ("universe", "use_low_turnover"): [True],
        ("universe", "use_value_tilt"): [True],
    }
    train_m, test_m = 48, 12
    return cfg_a, cfg_b, axes, train_m, test_m


def _print_startup_parameters(
    *,
    mode: str,
    output_root: pathlib.Path,
    cfg_a: BacktestConfig | None = None,
    cfg_b: BacktestConfig | None = None,
    axes: dict | None = None,
    train_m: int | None = None,
    test_m: int | None = None,
    smoke_cusips: int | None = None,
    smoke_tickers: int | None = None,
    skip_marginal: bool = False,
    skip_sweep: bool = False,
    equity_only: bool = False,
    refresh_openfigi_metadata: bool = False,
    price_source: str | None = None,
    live_config: dict | None = None,
) -> None:
    live_cfg = live_config or LIVE_CONFIG
    print("\nRun Parameters")
    print(f"  mode                  {mode}")
    print(f"  output_root           {output_root}")
    print(f"  equity-only filter    {equity_only}")
    if mode in {"live", "live-smoke"}:
        print(f"  SEC history start     {live_cfg.get('sec_history_start')}")
        print(f"  price window          {live_cfg.get('start')} -> {live_cfg.get('end')}")
        print(f"  benchmark             {live_cfg.get('benchmark_ticker')}")
        print(f"  SEC identity          {live_cfg.get('identity')}")
        print(f"  OpenFIGI key          {'env/config present' if live_cfg.get('openfigi_key') or os.environ.get('OPENFIGI_API_KEY') else 'missing'}")
        print(f"  OpenFIGI cache        {live_cfg.get('openfigi_cache_path')}")
        print(f"  price cache           {live_cfg.get('price_cache_path')}")
        print(f"  price source          {price_source or live_cfg.get('price_source')}")
        print(f"  security overrides    {live_cfg.get('security_overrides_path')}")
        print(f"  exclude fund-like     {live_cfg.get('exclude_fund_like_holdings')}")
        print(f"  active benchmark      {live_cfg.get('active_benchmark_source')}")
        if live_cfg.get("active_benchmark_source") == "manager_held_mcap":
            print(f"  market-cap cache      {live_cfg.get('market_cap_cache_path')}")
            print(f"  market-cap stale d    {live_cfg.get('market_cap_max_stale_days')}")
            print(f"  market-cap auto fetch {live_cfg.get('market_cap_auto_download')}")
        elif live_cfg.get("active_benchmark_source") not in {"visible_13f_aggregate", "13f_aggregate"}:
            print(f"  active bench weights  {live_cfg.get('active_benchmark_weights_path')}")
            print(f"  active bench stale d  {live_cfg.get('active_benchmark_max_stale_days')}")
        if equity_only:
            print(f"  fund exclusions       {live_cfg.get('fund_ticker_exclusions_path')}")
        print(f"  refresh FIGI metadata {refresh_openfigi_metadata}")
        print(
            "  live universe         "
            f"min_aum={live_cfg.get('min_aum'):.0f}, "
            f"max_aum={live_cfg.get('max_aum'):.0f}, "
            f"max_holdings={live_cfg.get('max_holdings')}, "
            f"max_put_weight={live_cfg.get('max_put_weight'):.0%}"
        )
    if mode == "live-smoke":
        print(f"  smoke CUSIPs          {smoke_cusips}")
        print(f"  smoke tickers         {smoke_tickers}")
    if cfg_a is not None:
        u = cfg_a.universe
        p = cfg_a.portfolio
        c = cfg_a.cost
        print(
            "  thesis universe       "
            f"manager_filter={cfg_a.manager_filter_mode}, "
            f"min_aum={u.min_aum:.0f}, max_aum={u.max_aum:.0f}, "
            f"use_concentration={u.use_concentration}, max_holdings={u.max_holdings}, "
            f"min_top{u.top_n_concentration}_weight={u.min_top_n_weight:.0%}, "
            f"turnover_q={u.turnover_quantile}, min_history_q={u.min_history_quarters}, "
            f"max_stale_filing_m={u.max_stale_filing_months}, max_stale_period_m={u.max_stale_period_months}, "
            f"hedge_put_max={u.hedge_put_max_weight:.0%}, value_tilt_min={u.value_tilt_min_pctl:.0%}"
        )
        print(
            "  thesis portfolio      "
            f"idea_signal={p.idea_signal}, top_n_ideas={p.top_n_ideas}, "
            f"idea_aggregation={p.idea_aggregation or ('score' if p.consensus_weight else 'manager_count')}, "
            f"min_consensus={p.min_consensus_funds}, holding_horizon_q={p.holding_horizon_q}, "
            f"min_portfolio_names={p.min_portfolio_names}, max_portfolio_names={p.max_portfolio_names}, "
            f"max_name={p.max_name_weight:.1%}, max_issuer={p.max_issuer_weight:.1%}, "
            f"min_active_weight_holdings={p.min_active_weight_holdings}, "
            f"missing_price_policy={cfg_a.missing_price_policy}, cost={c.bps_per_side:.1f}bps"
        )
        print(f"  thesis active bench   {cfg_a.active_benchmark_source}")
        print(
            "  CPS idio vol          "
            f"model=capm, window={live_cfg.get('idio_vol_window_months')}m, "
            f"min_obs={live_cfg.get('idio_vol_min_obs')}, "
            f"floor={live_cfg.get('idio_vol_floor'):.0%}, cap={live_cfg.get('idio_vol_cap'):.0%}, "
            f"winsor={live_cfg.get('idio_vol_winsor_lower'):.0%}/{live_cfg.get('idio_vol_winsor_upper'):.0%} "
            "(heuristic)"
        )
    if cfg_b is not None:
        print(
            "  placebo portfolio     "
            f"manager_filter={cfg_b.manager_filter_mode}, "
            f"idea_signal={cfg_b.portfolio.idea_signal}, "
            f"min_consensus={cfg_b.portfolio.min_consensus_funds}, "
            f"holding_horizon_q={cfg_b.portfolio.holding_horizon_q}"
        )
    if axes is not None:
        axis_text = "; ".join(f"{scope}.{field}={list(values)}" for (scope, field), values in axes.items())
        print(f"  sweep axes            {axis_text}")
        print(f"  sweep trials          {int(np.prod([len(v) for v in axes.values()]))}")
    if train_m is not None and test_m is not None:
        print(f"  walk-forward          train={train_m}m, test={test_m}m, select_on=active_sharpe")
    print(f"  skip_marginal         {skip_marginal}")
    print(f"  skip_sweep            {skip_sweep}")
    print("")


def run(
    mode: str,
    output_root: pathlib.Path,
    *,
    smoke_cusips: int = 300,
    smoke_tickers: int = 200,
    skip_marginal: bool = False,
    skip_sweep: bool = False,
    sweep_checkpoint_every: int = 5,
    equity_only: bool = False,
    refresh_openfigi_metadata: bool = False,
    price_source: str | None = None,
    active_benchmark_source: str | None = None,
    active_benchmark_weights_path: str | None = None,
    active_benchmark_max_stale_days: int | None = None,
    manager_filter_mode: str | None = None,
) -> pathlib.Path:
    cfg_a, cfg_b, axes, train_m, test_m = _default_run_configs()
    live_config = dict(LIVE_CONFIG)
    if equity_only:
        live_config["exclude_fund_like_holdings"] = True
    if refresh_openfigi_metadata:
        live_config["refresh_openfigi_metadata"] = True
    if price_source is not None:
        live_config["price_source"] = price_source
    if active_benchmark_source is not None:
        live_config["active_benchmark_source"] = active_benchmark_source
    if active_benchmark_weights_path is not None:
        live_config["active_benchmark_weights_path"] = active_benchmark_weights_path
    if active_benchmark_max_stale_days is not None:
        live_config["active_benchmark_max_stale_days"] = active_benchmark_max_stale_days
    if manager_filter_mode is not None:
        cfg_a = replace(cfg_a, manager_filter_mode=manager_filter_mode)
    cfg_a = replace(
        cfg_a,
        active_benchmark_source=(
            live_config.get("active_benchmark_source", "manager_held_mcap")
            if mode == "live"
            else "manager_held_mcap"
        ),
    )
    _preflight_active_benchmark_inputs(mode=mode, live_config=live_config, cfg=cfg_a)
    _print_startup_parameters(
        mode=mode,
        output_root=output_root,
        cfg_a=None if mode == "live-smoke" else cfg_a,
        cfg_b=None if mode == "live-smoke" else cfg_b,
        axes=None if mode == "live-smoke" else axes,
        train_m=None if mode == "live-smoke" else train_m,
        test_m=None if mode == "live-smoke" else test_m,
        smoke_cusips=smoke_cusips,
        smoke_tickers=smoke_tickers,
        skip_marginal=skip_marginal,
        skip_sweep=skip_sweep,
        equity_only=equity_only,
        refresh_openfigi_metadata=refresh_openfigi_metadata,
        price_source=live_config.get("price_source"),
        live_config=live_config,
    )
    if mode == "synthetic":
        holdings, prices, factors, value_scores, bench_w, bench_ret, signal_prices = build_synthetic_data()
    elif mode == "live-smoke":
        return run_live_smoke(
            output_root,
            cusip_limit=smoke_cusips,
            ticker_limit=smoke_tickers,
            cfg=live_config,
        )
    else:
        holdings, prices, factors, value_scores, bench_w, bench_ret, signal_prices = build_live_data(live_config)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"[output] Writing incremental reports under {out_dir}")

    print("[2/6] Building manager research artifacts", flush=True)
    stage2_started = time.perf_counter()
    stage2_log_path = out_dir / "stage2_progress.log"
    progress = _progress_printer(stage2_started, stage2_log_path)
    progress(f"stage 2 log initialized: {stage2_log_path}")
    # A DataFrame stored inside attrs may be deep-copied by ordinary pandas
    # operations. Detach the raw classification book before characteristics to
    # prevent repeated copies of hundreds of thousands of holding rows.
    filing_put_weights = holdings.attrs.pop("raw_filing_put_weights", None)
    if mode == "live" and filing_put_weights is None:
        raise ValueError(
            "live run is missing raw SEC filing PUT weights; refusing to use the equity-only "
            "book for the hedge filter"
        )
    raw_holdings_for_classification = holdings.attrs.pop("raw_mapped_holdings", None)
    if raw_holdings_for_classification is None:
        raw_holdings_for_classification = holdings
        progress("raw mapped holdings not attached; classification will use filtered holdings")
    else:
        progress(
            f"detached raw mapped holdings rows={len(raw_holdings_for_classification):,} "
            f"from filtered rows={len(holdings):,}"
        )
    progress(
        f"2a/5 characteristics starting: holdings_shallow={_frame_shallow_mb(holdings):,.1f}MB, "
        f"raw_shallow={_frame_shallow_mb(raw_holdings_for_classification):,.1f}MB"
    )
    t_step = time.perf_counter()
    try:
        chars = manager_characteristics(
            holdings,
            bench_w,
            filing_put_weights=filing_put_weights,
            progress=progress,
        )
    except Exception as exc:
        progress(f"2a/5 characteristics FAILED: {type(exc).__name__}: {exc}")
        raise
    progress(
        f"2a/5 characteristics done: rows={len(chars):,}, shallow={_frame_shallow_mb(chars):,.1f}MB, "
        f"step={time.perf_counter() - t_step:.1f}s"
    )
    progress("2a/5 raw-book characteristics audit starting")
    raw_chars = manager_characteristics(
        raw_holdings_for_classification,
        bench_w,
        filing_put_weights=filing_put_weights,
        progress=progress,
    )
    characteristics_audit = manager_characteristics_audit(raw_chars, chars)
    characteristics_audit_path = out_dir / "manager_characteristics_raw_investable.csv"
    characteristics_audit.to_csv(characteristics_audit_path, index=False)
    progress(
        f"2a/5 raw/investable characteristics saved: {characteristics_audit_path} "
        f"rows={len(characteristics_audit):,}"
    )
    del raw_chars
    progress("2b/5 visible-version snapshots starting")
    t_step = time.perf_counter()
    try:
        visible_cache = build_visible_versions_cache(chars, prices.index, progress=progress)
    except Exception as exc:
        progress(f"2b/5 visible snapshots FAILED: {type(exc).__name__}: {exc}")
        raise
    progress(
        f"2b/5 visible snapshots done: months={len(visible_cache)}, "
        f"rows={sum(len(x) for x in visible_cache.values()):,}, step={time.perf_counter() - t_step:.1f}s"
    )
    progress("2c/5 manager classification/cache starting")
    manager_classification, manager_overrides, manager_classification_summary = _manager_filter_artifacts(
        out_dir=out_dir,
        raw_holdings=raw_holdings_for_classification,
        holdings=holdings,
        chars=chars,
        prices=signal_prices,
        factors=factors,
        visible_versions_cache=visible_cache,
        live_config=live_config,
        progress=progress,
    )
    progress("2c/5 manager classification/cache done")
    del raw_holdings_for_classification
    progress("2d/5 security issuer groups starting")
    security_groups = load_security_groups(
        prices.columns,
        live_config.get("security_overrides_path", "data/security_overrides.csv"),
    )
    progress(f"2d/5 security issuer groups done: groups={security_groups.nunique():,}")
    progress("2e/5 active benchmark and idio-vol caches starting")
    active_benchmark_weights_by_month, active_benchmark_summary = _load_active_benchmark_weights_by_month(
        live_config=live_config,
        months=prices.index,
        tickers=prices.columns,
        cfg=cfg_a,
        mode=mode,
    )
    if active_benchmark_weights_by_month is not None:
        coverage_path = out_dir / "active_benchmark_coverage_by_month.csv"
        pd.DataFrame(
            [
                {"month_end": month, "covered_tickers": len(values)}
                for month, values in sorted(active_benchmark_weights_by_month.items())
            ]
        ).to_csv(coverage_path, index=False)
        active_benchmark_summary["coverage_output"] = str(coverage_path)
        progress(f"active benchmark coverage saved: {coverage_path}")
    idiosyncratic_vol_by_month, idio_vol_summary = _idiosyncratic_vol_artifacts(
        prices=signal_prices,
        factors=factors,
        months=prices.index,
        live_config=live_config,
        required=_signals_need_idiosyncratic_vol([cfg_a, cfg_b], None if skip_sweep else axes),
        progress=progress,
    )
    progress(f"stage 2 complete in {time.perf_counter() - stage2_started:.1f}s")

    print("[3/6] Running thesis and placebo backtests")
    t_step = time.perf_counter()
    print("    thesis backtest running")
    ret_a = run_backtest(
        holdings,
        prices,
        cfg_a,
        value_scores,
        bench_w,
        chars,
        visible_cache,
        security_groups,
        active_benchmark_weights_by_month,
        manager_classification,
        manager_overrides,
        capture_rebalance=True,
        idiosyncratic_vol_by_month=idiosyncratic_vol_by_month,
    )
    print(f"    thesis backtest done in {time.perf_counter() - t_step:.1f}s")
    t_placebo = time.perf_counter()
    print("    placebo backtest running")
    ret_b = run_backtest(
        holdings,
        prices,
        cfg_b,
        value_scores,
        bench_w,
        chars,
        visible_cache,
        security_groups,
        active_benchmark_weights_by_month,
        manager_classification,
        manager_overrides,
        idiosyncratic_vol_by_month=idiosyncratic_vol_by_month,
    )
    print(f"    placebo backtest done in {time.perf_counter() - t_placebo:.1f}s")
    print("    attribution running")
    att_a = attribution(ret_a, factors, bench_ret)
    att_b = attribution(ret_b, factors, bench_ret)
    print(f"    thesis/placebo + attribution done in {time.perf_counter() - t_step:.1f}s")
    for label, att in (("Thesis", att_a), ("Placebo", att_b)):
        print(f"\n  {label}")
        for key in ["ann_return", "ann_vol", "sharpe", "ann_alpha", "alpha_t", "ir_vs_benchmark"]:
            val = att.get(key)
            print(f"  {key:<22} {round(val, 3) if isinstance(val, float) else val}")

    dashboard_params = _dashboard_parameter_summary(
        mode=mode,
        cfg=cfg_a,
        prices=prices,
        holdings=holdings,
        benchmark=bench_ret,
        axes=axes,
        train_m=train_m,
        test_m=test_m,
        security_groups=security_groups,
    )
    thesis_summary = ret_a.attrs.get("rebalance_summary", pd.DataFrame())
    thesis_summary_path = out_dir / "rebalance_summary_thesis_partial.csv"
    thesis_summary.to_csv(thesis_summary_path, index=False)
    stage3_stats = _rebalance_summary_stats(thesis_summary)
    print(f"  Saved thesis rebalance summary: {thesis_summary_path}")
    _print_rebalance_summary(stage3_stats)
    stage3_payload = {
        "stage": "after_thesis_placebo",
        "mode": mode,
        "cfg_thesis": asdict(cfg_a),
        "cfg_placebo": asdict(cfg_b),
        "active_benchmark": active_benchmark_summary,
        "sweep_axes": {f"{scope}.{field}": values for (scope, field), values in axes.items()},
        "metrics": {"thesis": att_a, "placebo": att_b},
        "rebalance_summary_stats": stage3_stats,
        "outputs": {"rebalance_summary_thesis_partial": str(thesis_summary_path)},
    }
    write_manifest(out_dir / "manifest_stage3.json", stage3_payload)
    quick_dashboard_path = dashboard(
        ret_a,
        ret_b,
        bench_ret,
        factors,
        pd.DataFrame({"filter": ["(skipped)"], "metric": [np.nan], "delta": [np.nan]}),
        pd.DataFrame(),
        heat_x="max_portfolio_names",
        heat_y="aum_band",
        dsr_info={
            "note": "sweep not run yet",
            "metric": "active_return_vs_benchmark",
            "benchmark": getattr(bench_ret, "name", None) if bench_ret is not None else None,
            "n_trials": int(np.prod([len(v) for v in axes.values()])),
            "T": 0,
        },
        parameter_summary=dashboard_params,
        title=f"13F-clone quick dashboard [{mode.upper()} DATA]",
        path=str(out_dir / "strategy_dashboard_stage3.png"),
    )
    print(f"  Saved quick dashboard: {quick_dashboard_path}")
    print("  Manager-filter acceptance diagnostics running")
    manager_filter_acceptance = write_manager_filter_acceptance(
        out_dir,
        holdings=holdings,
        prices=prices,
        base_cfg=cfg_a,
        factors=factors,
        benchmark=bench_ret,
        value_scores=value_scores,
        benchmark_weights=bench_w,
        chars=chars,
        visible_versions_cache=visible_cache,
        security_groups=security_groups,
        active_benchmark_weights_by_month=active_benchmark_weights_by_month,
        idiosyncratic_vol_by_month=idiosyncratic_vol_by_month,
        manager_classification=manager_classification,
        manager_overrides=manager_overrides,
    )

    if skip_marginal:
        print("\n[4/6] Marginal-IR ablation skipped")
        ablation = pd.DataFrame({"filter": ["(skipped)"], "metric": [np.nan], "delta": [np.nan]})
    else:
        print("\n[4/6] Marginal-IR ablation")
        t_step = time.perf_counter()
        ablation = marginal_ir(
            holdings,
            prices,
            factors,
            cfg_a,
            bench_ret,
            value_scores,
            bench_w,
            chars=chars,
            visible_versions_cache=visible_cache,
            security_groups=security_groups,
            active_benchmark_weights_by_month=active_benchmark_weights_by_month,
            idiosyncratic_vol_by_month=idiosyncratic_vol_by_month,
            manager_classification=manager_classification,
            manager_overrides=manager_overrides,
            verbose=True,
        )
        print(f"  marginal-ir total time {time.perf_counter() - t_step:.1f}s")
        print(ablation.to_string(index=False))

    grid_returns = {}
    required_m = train_m + test_m
    if skip_sweep:
        print("\n[5/6] Grid eval and walk-forward sweep skipped")
        grid = pd.DataFrame()
        oos_ret = pd.Series(dtype=float)
        wf_log = pd.DataFrame()
        n_trials = int(np.prod([len(v) for v in axes.values()]))
        dsr = {
            "note": "skipped by --skip-sweep",
            "metric": "active_return_vs_benchmark",
            "benchmark": getattr(bench_ret, "name", None) if bench_ret is not None else None,
            "n_trials": int(n_trials),
            "price_months": int(len(prices)),
            "required_months": int(required_m),
            "T": 0,
        }
    else:
        print("\n[5/6] Grid eval and walk-forward sweep")
        t_step = time.perf_counter()
        grid = grid_eval(
            holdings,
            prices,
            factors,
            cfg_a,
            axes,
            bench_ret,
            value_scores,
            bench_w,
            metric="active_sharpe",
            chars=chars,
            visible_versions_cache=visible_cache,
            verbose=True,
            include_returns=True,
            security_groups=security_groups,
            active_benchmark_weights_by_month=active_benchmark_weights_by_month,
            idiosyncratic_vol_by_month=idiosyncratic_vol_by_month,
            manager_classification=manager_classification,
            manager_overrides=manager_overrides,
            checkpoint_dir=out_dir,
            checkpoint_every=sweep_checkpoint_every,
        )
        grid_returns = grid.attrs.get("returns_by_config")
        print(f"  grid eval total time {time.perf_counter() - t_step:.1f}s")
    if not skip_sweep and len(prices) >= required_m:
        oos_ret, wf_log, n_trials = walk_forward(
            holdings,
            prices,
            factors,
            cfg_a,
            axes,
            bench_ret,
            value_scores,
            bench_w,
            train_m=train_m,
            test_m=test_m,
            select_on="active_sharpe",
            chars=chars,
            visible_versions_cache=visible_cache,
            verbose=True,
            precomputed_returns=grid_returns,
            security_groups=security_groups,
            active_benchmark_weights_by_month=active_benchmark_weights_by_month,
            idiosyncratic_vol_by_month=idiosyncratic_vol_by_month,
            manager_classification=manager_classification,
            manager_overrides=manager_overrides,
        )
        oos_dsr_stream = active_return_stream(oos_ret, bench_ret)
        dsr = deflated_sharpe(
            oos_dsr_stream,
            n_trials,
            sr_variance=float(np.nanvar(grid["active_sharpe"] / np.sqrt(12))),
        )
        dsr["metric"] = "active_return_vs_benchmark"
        dsr["benchmark"] = getattr(bench_ret, "name", None) if bench_ret is not None else None
        dsr["raw_oos_months"] = int(len(oos_ret.dropna()))
    elif not skip_sweep:
        n_trials = int(np.prod([len(v) for v in axes.values()]))
        oos_ret = pd.Series(dtype=float)
        wf_log = pd.DataFrame()
        dsr = {
            "note": "insufficient price history for walk-forward",
            "metric": "active_return_vs_benchmark",
            "benchmark": getattr(bench_ret, "name", None) if bench_ret is not None else None,
            "price_months": int(len(prices)),
            "required_months": int(required_m),
            "train_m": int(train_m),
            "test_m": int(test_m),
            "n_trials": int(n_trials),
            "T": 0,
        }
    if "note" in dsr:
        print(f"  OOS active Sharpe      skipped ({dsr['note']})")
        print(f"  price months           {dsr.get('price_months', len(oos_ret))}/{dsr.get('required_months', required_m)}")
        print(f"  n_trials               {dsr.get('n_trials')}")
    else:
        print(f"  OOS active Sharpe      {dsr.get('ann_SR', float('nan')):.2f}")
        print(f"  n_trials               {dsr.get('n_trials')}")
        print(f"  Deflated Sharpe (DSR)  {dsr.get('DSR', float('nan')):.2f}")

    if skip_sweep:
        interactive_config_id = "thesis"
        interactive_grid = single_config_result_grid(
            interactive_config_id,
            ret_a,
            config={
                "manager_filter_mode": cfg_a.manager_filter_mode,
                "aum_band": f"{cfg_a.universe.min_aum / 1e9:g}-{cfg_a.universe.max_aum / 1e9:g}B",
                "use_concentration": cfg_a.universe.use_concentration,
                "use_low_turnover": cfg_a.universe.use_low_turnover,
                "use_value_tilt": cfg_a.universe.use_value_tilt,
                "idea_signal": cfg_a.portfolio.idea_signal,
                "top_n_ideas": cfg_a.portfolio.top_n_ideas,
                "min_consensus_funds": cfg_a.portfolio.min_consensus_funds,
                "holding_horizon_q": cfg_a.portfolio.holding_horizon_q,
                "min_portfolio_names": cfg_a.portfolio.min_portfolio_names,
                "max_portfolio_names": cfg_a.portfolio.max_portfolio_names,
                "min_active_weight_holdings": cfg_a.portfolio.min_active_weight_holdings,
            },
            metrics=att_a,
            rebalance_stats=stage3_stats,
        )
        interactive_returns = {interactive_config_id: ret_a}
        sweep_outputs = {
            "interactive_html": str(out_dir / "interactive_results.html"),
            "status": "sweep_skipped_single_thesis_config",
        }
    else:
        print("\n[6/6] Writing sweep result files")
        sweep_outputs = write_sweep_outputs(
            out_dir,
            grid,
            grid.attrs.get("returns_by_config_id"),
            bench_ret,
        )
        print(f"  Saved sweep grid:       {sweep_outputs['grid']}")
        print(f"  Saved sweep returns:    {sweep_outputs['returns']}")
        interactive_grid = grid
        interactive_returns = grid.attrs.get("returns_by_config_id") or {}

    print("\n[6/6] Rendering strategy dashboard")
    dashboard_path = dashboard(
        ret_a,
        ret_b,
        bench_ret,
        factors,
        ablation,
        grid,
        heat_x="max_portfolio_names",
        heat_y="aum_band",
        dsr_info=dsr,
        oos_log=wf_log,
        parameter_summary=dashboard_params,
        title=f"13F-clone strategy dashboard [{mode.upper()} DATA]",
        path=str(out_dir / "strategy_dashboard.png"),
    )
    print("[6/6] Writing rebalance audit files")
    rebalance_outputs = write_rebalance_outputs(
        out_dir,
        "thesis",
        holdings,
        prices,
        cfg_a,
        value_scores=value_scores,
        benchmark_weights=bench_w,
        chars=chars,
        visible_versions_cache=visible_cache,
        security_groups=security_groups,
        active_benchmark_weights_by_month=active_benchmark_weights_by_month,
        idiosyncratic_vol_by_month=idiosyncratic_vol_by_month,
        manager_classification=manager_classification,
        manager_overrides=manager_overrides,
    )
    print(f"  Saved rebalance summary:  {rebalance_outputs['summary']}")
    print(f"  Saved rebalance holdings: {rebalance_outputs['holdings']}")
    print(f"  Saved rebalance managers: {rebalance_outputs['managers']}")
    print(f"  Saved rebalance rules:    {rebalance_outputs['rules']}")
    _print_rebalance_summary(rebalance_outputs.get("summary_stats", {}))
    print("[6/6] Writing value-unit diagnostics")
    value_diag = value_unit_continuity_diagnostics(chars)
    value_diag_path = out_dir / "value_unit_diagnostics.csv"
    value_diag.to_csv(value_diag_path, index=False)
    suspicious_value_unit_jumps = int(value_diag["suspicious_unit_jump"].sum()) if not value_diag.empty else 0
    print(
        f"  Saved value-unit diagnostics: {value_diag_path} "
        f"({suspicious_value_unit_jumps} suspicious cutoff jumps)"
    )
    manifest_payload = {
        "mode": mode,
        "live_config": live_config if mode == "live" else None,
        "cfg_thesis": asdict(cfg_a),
        "cfg_placebo": asdict(cfg_b),
        "active_benchmark": active_benchmark_summary,
        "sweep_axes": {f"{scope}.{field}": values for (scope, field), values in axes.items()},
        "dashboard_parameter_summary": dashboard_params,
        "input_summary": {
            "holdings_rows": int(len(holdings)),
            "manager_count": int(holdings["manager"].nunique()),
            "ticker_count": int(holdings["ticker"].nunique()),
            "price_months": int(len(prices)),
            "price_columns": int(len(prices.columns)),
            "factor_months": int(len(factors)),
            "mapping_diagnostics": holdings.attrs.get("mapping_diagnostics"),
            "price_filter_diagnostics": holdings.attrs.get("price_filter_diagnostics"),
            "price_alignment_diagnostics": holdings.attrs.get("price_alignment_diagnostics"),
            "price_diagnostics": prices.attrs.get("price_diagnostics"),
            "security_overrides_path": live_config.get("security_overrides_path", "data/security_overrides.csv"),
            "active_benchmark_source": cfg_a.active_benchmark_source,
            "active_benchmark_weights_path": live_config.get("active_benchmark_weights_path"),
            "active_benchmark_max_stale_days": live_config.get("active_benchmark_max_stale_days"),
            "issuer_group_count": int(security_groups.nunique()),
            "value_unit_diagnostics": {
                "path": str(value_diag_path),
                "rows": int(len(value_diag)),
                "suspicious_unit_jumps": suspicious_value_unit_jumps,
            },
            "manager_classification": manager_classification_summary,
            "manager_filter_acceptance": manager_filter_acceptance,
            "idiosyncratic_vol": idio_vol_summary,
        },
        "metrics": {"thesis": att_a, "placebo": att_b, "dsr": dsr},
        "rebalance_summary_stats": rebalance_outputs.get("summary_stats"),
        "outputs": {
            "dashboard": dashboard_path,
            "manager_characteristics_raw_investable": str(characteristics_audit_path),
            "rebalance_thesis": rebalance_outputs,
            "sweep": sweep_outputs,
            "stage2_progress_log": str(stage2_log_path),
        },
    }
    manifest_path = out_dir / "manifest.json"
    write_manifest(manifest_path, manifest_payload)
    manifest_for_html = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_summary_for_html = dict(manifest_for_html.get("input_summary") or {})
    input_summary_for_html["mapping"] = input_summary_for_html.get("mapping_diagnostics") or {}
    rules_for_html = json.loads(pathlib.Path(rebalance_outputs["rules"]).read_text(encoding="utf-8"))
    portfolio_config_id = (
        interactive_config_id
        if skip_sweep
        else _sweep_config_id_for_cfg(grid, axes, cfg_a)
    )
    interactive_results(
        interactive_grid,
        interactive_returns,
        benchmark=bench_ret,
        path=sweep_outputs["interactive_html"],
        portfolio_holdings=pd.read_csv(rebalance_outputs["holdings"]),
        rebalance_summary=pd.read_csv(rebalance_outputs["summary"]),
        portfolio_config_id=portfolio_config_id,
        meta_payload={
            "benchmarkName": getattr(bench_ret, "name", None),
            "thesisConfigId": portfolio_config_id,
            "configHash": manifest_for_html.get("config_hash"),
            "gitSha": manifest_for_html.get("git_sha"),
            "runTimestampUtc": manifest_for_html.get("run_timestamp_utc"),
            "dashboardParameterSummary": dashboard_params,
            "metrics": manifest_for_html.get("metrics") or {},
            "rebalanceSummaryStats": manifest_for_html.get("rebalance_summary_stats") or {},
            "inputSummary": input_summary_for_html,
            "rules": rules_for_html,
        },
    )
    print(f"  Saved interactive HTML: {sweep_outputs['interactive_html']}")
    print(f"  Saved dashboard: {dashboard_path}")
    print(f"  Saved manifest:  {manifest_path}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 13F-clone research example.")
    parser.add_argument(
        "--mode",
        choices=["synthetic", "live", "live-smoke"],
        default="synthetic",
        help="synthetic is offline; live-smoke tests the live data chain; live runs the full research stack.",
    )
    parser.add_argument("--output-root", default="reports", help="Directory for run outputs.")
    parser.add_argument("--smoke-cusips", type=int, default=300, help="Top CUSIPs by value to map in live-smoke mode.")
    parser.add_argument("--smoke-tickers", type=int, default=200, help="Top tickers by value to price in live-smoke mode.")
    parser.add_argument("--skip-marginal", action="store_true", help="Skip marginal-IR ablation after thesis/placebo.")
    parser.add_argument("--skip-sweep", action="store_true", help="Skip parameter grid and walk-forward sweep.")
    parser.add_argument(
        "--equity-only",
        action="store_true",
        help="Exclude ETF/ETN/fund-like 13F rows before pricing and idea generation.",
    )
    parser.add_argument(
        "--refresh-openfigi-metadata",
        action="store_true",
        help="Re-query cached OpenFIGI CUSIPs that lack metadata fields needed for security classification.",
    )
    parser.add_argument(
        "--price-source",
        choices=["chart", "auto", "yfinance"],
        default=None,
        help="Price download source. Default live config uses chart to avoid yfinance hangs on restricted networks.",
    )
    parser.add_argument(
        "--active-benchmark-source",
        choices=["manager_held_mcap", "visible_13f_aggregate", "spy_holdings"],
        default=None,
        help=(
            "Benchmark used for active_weight/CPS-IR signals. manager_held_mcap is the default; "
            "visible_13f_aggregate is only a diagnostic peer-13F proxy."
        ),
    )
    parser.add_argument(
        "--active-benchmark-weights",
        default=None,
        help="CSV/Parquet/XLSX long table with month_end,ticker,weight for non-13F active benchmark sources.",
    )
    parser.add_argument(
        "--active-benchmark-max-stale-days",
        type=int,
        default=None,
        help="Maximum age of benchmark-weight snapshot allowed for a rebalance month.",
    )
    parser.add_argument(
        "--sweep-checkpoint-every",
        type=int,
        default=5,
        help="Write sweep_grid_partial.csv every N grid configs; 0 disables checkpointing.",
    )
    parser.add_argument(
        "--manager-filter-mode",
        choices=["all", "exclude_dirty", "dedicated_like"],
        default=None,
        help="Manager-type filter for the thesis run. all is the untouched baseline.",
    )
    return parser.parse_args()


def main() -> None:
    _load_local_env()
    args = parse_args()
    run(
        args.mode,
        pathlib.Path(args.output_root),
        smoke_cusips=args.smoke_cusips,
        smoke_tickers=args.smoke_tickers,
        skip_marginal=args.skip_marginal,
        skip_sweep=args.skip_sweep,
        sweep_checkpoint_every=args.sweep_checkpoint_every,
        equity_only=args.equity_only,
        refresh_openfigi_metadata=args.refresh_openfigi_metadata,
        price_source=args.price_source,
        active_benchmark_source=args.active_benchmark_source,
        active_benchmark_weights_path=args.active_benchmark_weights,
        active_benchmark_max_stale_days=args.active_benchmark_max_stale_days,
        manager_filter_mode=args.manager_filter_mode,
    )


if __name__ == "__main__":
    main()
