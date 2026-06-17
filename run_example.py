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
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from engine import (
    BacktestConfig,
    PortfolioConfig,
    UniverseConfig,
    attribution,
    build_visible_versions_cache,
    manager_characteristics,
    marginal_ir,
    rebalance_trace,
    run_backtest,
)
from report import dashboard
from sweep import active_return_stream, deflated_sharpe, grid_eval, walk_forward


LIVE_CONFIG = {
    "identity": "YourName you@firm.com",
    "openfigi_key": None,
    "sec_history_start": "2013-10-01",
    "start": "2015-01-01",
    "end": "2026-05-31",
    # Broad-market total-return proxy. Use QQQ for a tighter growth-style proxy.
    "benchmark_ticker": "SPY",
    "min_aum": 1e9,
    "max_aum": 30e9,
    "max_holdings": 60,
    "max_put_weight": 0.10,
    "require_factors": False,
    "openfigi_cache_path": "openfigi_cache.parquet",
    "price_cache_path": "yfinance_close_cache.parquet",
}


def _load_local_env(path: pathlib.Path = pathlib.Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _json_default(x: Any) -> Any:
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, np.generic):
        return x.item()
    return str(x)


def _config_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def build_synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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
    return holdings, prices, factors, value_scores, bench_w, bench_ret


def _top_cusips_by_value(holdings: pd.DataFrame, limit: int | None) -> pd.Index:
    if limit is None or limit <= 0:
        return pd.Index(holdings["cusip"].dropna().unique())
    return holdings.groupby("cusip")["value"].sum().nlargest(limit).index


def build_live_data(
    cfg: dict,
    *,
    cusip_limit: int | None = None,
    price_ticker_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, None, None, pd.Series | None]:
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
    print(
        f"    rule-based pool: {h_cusip['cik'].nunique()} filers, "
        f"{h_cusip.groupby(['cik', 'period_date']).ngroups} filer-periods"
    )
    print("[1/6] Mapping CUSIPs through OpenFIGI")
    target_cusips = _top_cusips_by_value(h_cusip, cusip_limit)
    if cusip_limit:
        h_cusip = h_cusip[h_cusip["cusip"].isin(target_cusips)].copy()
        print(f"    smoke CUSIP subset: top {len(target_cusips)} CUSIPs by disclosed value")
    cmap = da.cusip_to_ticker(
        target_cusips,
        api_key=cfg["openfigi_key"],
        cache_path=cfg.get("openfigi_cache_path"),
    )
    holdings = da.map_holdings_to_tickers(h_cusip, cmap)
    holdings = da.priceable_holdings(holdings)
    print("[1/6] Downloading monthly prices from yfinance")
    price_holdings = holdings
    if price_ticker_limit:
        top_tickers = holdings.groupby("ticker")["value"].sum().nlargest(price_ticker_limit).index
        price_holdings = holdings[holdings["ticker"].isin(top_tickers)].copy()
        print(f"    smoke ticker subset: top {len(top_tickers)} tickers by disclosed value")
    prices = da.fetch_prices(
        price_holdings.ticker.unique(),
        cfg["start"],
        cfg["end"],
        cache_path=cfg.get("price_cache_path"),
    )
    holdings = da.align_holdings_to_prices(price_holdings, prices)
    print("[1/6] Downloading Fama-French factors")
    try:
        factors = da.fetch_factors(cfg["start"], cfg["end"])
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
            require_full_window=True,
        ).iloc[:, 0]
        bench_ret.name = cfg["benchmark_ticker"]
    except Exception as exc:
        print(f"    [warn] benchmark fetch failed; benchmark disabled: {exc}")
        bench_ret = None
    print(f"    managers: {holdings.manager.nunique()}, tickers: {holdings.ticker.nunique()}")
    return holdings, prices, factors, None, None, bench_ret


def run_live_smoke(output_root: pathlib.Path, *, cusip_limit: int, ticker_limit: int) -> pathlib.Path:
    cfg = dict(LIVE_CONFIG)
    holdings, prices, factors, _, _, bench_ret = build_live_data(
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


def write_manifest(path: pathlib.Path, payload: dict) -> None:
    versions = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "config_hash": _config_hash(payload),
        "library_versions": versions,
        **payload,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _strategy_rule_summary(cfg: BacktestConfig, *, value_scores, benchmark_weights) -> dict:
    return {
        "rebalance_rule": (
            "At each month-end on or after a visible SEC filing date, select each "
            "manager's latest filing version available by filing_date and rebalance."
        ),
        "point_in_time_rule": "Only filings with filing_date <= rebalance_month are visible.",
        "execution_timing": (
            "Existing holdings earn the current month's return first; new target "
            "weights are set at month-end for subsequent months."
        ),
        "transaction_cost": {
            "model": "one_way_turnover * bps_per_side",
            "bps_per_side": cfg.cost.bps_per_side,
        },
        "universe": asdict(cfg.universe),
        "portfolio": asdict(cfg.portfolio),
        "missing_price_policy": cfg.missing_price_policy,
        "active_filter_status": {
            "value_tilt_configured": bool(cfg.universe.use_value_tilt),
            "value_tilt_active": bool(cfg.universe.use_value_tilt and value_scores is not None),
            "active_share_configured": bool(cfg.universe.use_active_share),
            "active_share_active": bool(cfg.universe.use_active_share and benchmark_weights is not None),
            "active_weight_signal": cfg.portfolio.idea_signal == "active_weight",
            "active_weight_benchmark": (
                "PIT equal-manager aggregate of all visible latest 13F books at each rebalance"
                if cfg.portfolio.idea_signal == "active_weight"
                else None
            ),
        },
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
) -> dict[str, str]:
    trace = rebalance_trace(
        holdings,
        prices,
        cfg,
        value_scores=value_scores,
        benchmark_weights=benchmark_weights,
        chars=chars,
        visible_versions_cache=visible_versions_cache,
    )
    outputs: dict[str, str] = {}
    for name, df in trace.items():
        path = out_dir / f"rebalance_{name}_{label}.csv"
        df.to_csv(path, index=False)
        outputs[name] = str(path)

    rules_path = out_dir / f"rebalance_rules_{label}.json"
    rules_path.write_text(
        json.dumps(
            _strategy_rule_summary(cfg, value_scores=value_scores, benchmark_weights=benchmark_weights),
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    outputs["rules"] = str(rules_path)
    return outputs


def run(mode: str, output_root: pathlib.Path, *, smoke_cusips: int = 300, smoke_tickers: int = 200) -> pathlib.Path:
    if mode == "synthetic":
        holdings, prices, factors, value_scores, bench_w, bench_ret = build_synthetic_data()
    elif mode == "live-smoke":
        return run_live_smoke(output_root, cusip_limit=smoke_cusips, ticker_limit=smoke_tickers)
    else:
        holdings, prices, factors, value_scores, bench_w, bench_ret = build_live_data(LIVE_CONFIG)

    print("[2/6] Computing per-manager characteristics")
    t_step = time.perf_counter()
    chars = manager_characteristics(holdings, bench_w)
    print(f"    {len(chars)} manager-filing-version rows in {time.perf_counter() - t_step:.1f}s")
    t_step = time.perf_counter()
    visible_cache = build_visible_versions_cache(chars, prices.index)
    print(f"    {len(visible_cache)} month-end visible-version snapshots in {time.perf_counter() - t_step:.1f}s")

    cfg_a = BacktestConfig(
        universe=UniverseConfig(
            min_aum=1e9,
            max_aum=30e9,
            min_top_n_weight=0.50,
            max_holdings=40,
            turnover_quantile=0.34,
            hedge_put_max_weight=0.05,
            value_tilt_min_pctl=0.50,
            min_history_quarters=4,
        ),
        portfolio=PortfolioConfig(
            idea_signal="change",
            top_n_ideas=8,
            min_consensus_funds=2,
            holding_horizon_q=2,
            max_name_weight=0.05,
        ),
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
    )

    print("[3/6] Running thesis and placebo backtests")
    t_step = time.perf_counter()
    print("    thesis backtest running")
    ret_a = run_backtest(holdings, prices, cfg_a, value_scores, bench_w, chars, visible_cache)
    print(f"    thesis backtest done in {time.perf_counter() - t_step:.1f}s")
    t_placebo = time.perf_counter()
    print("    placebo backtest running")
    ret_b = run_backtest(holdings, prices, cfg_b, value_scores, bench_w, chars, visible_cache)
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
        verbose=True,
    )
    print(f"  marginal-ir total time {time.perf_counter() - t_step:.1f}s")
    print(ablation.to_string(index=False))

    print("\n[5/6] Grid eval and walk-forward sweep")
    axes = {
        ("portfolio", "idea_signal"): ["level", "change", "initiation", "active_weight"],
        ("portfolio", "min_consensus_funds"): [1, 2],
        ("portfolio", "top_n_ideas"): [5, 8],
        ("universe", "turnover_quantile"): [0.34, 0.50],
    }
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
    )
    grid_returns = grid.attrs.get("returns_by_config")
    print(f"  grid eval total time {time.perf_counter() - t_step:.1f}s")
    train_m, test_m = 48, 12
    required_m = train_m + test_m
    if len(prices) >= required_m:
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
    else:
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

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    print("\n[6/6] Rendering strategy dashboard")
    dashboard_path = dashboard(
        ret_a,
        ret_b,
        bench_ret,
        factors,
        ablation,
        grid,
        heat_x="top_n_ideas",
        heat_y="turnover_quantile",
        dsr_info=dsr,
        oos_log=wf_log,
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
    )
    print(f"  Saved rebalance summary:  {rebalance_outputs['summary']}")
    print(f"  Saved rebalance holdings: {rebalance_outputs['holdings']}")
    print(f"  Saved rebalance managers: {rebalance_outputs['managers']}")
    print(f"  Saved rebalance rules:    {rebalance_outputs['rules']}")
    manifest_payload = {
        "mode": mode,
        "live_config": LIVE_CONFIG if mode == "live" else None,
        "cfg_thesis": asdict(cfg_a),
        "cfg_placebo": asdict(cfg_b),
        "sweep_axes": {f"{scope}.{field}": values for (scope, field), values in axes.items()},
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
        },
        "metrics": {"thesis": att_a, "placebo": att_b, "dsr": dsr},
        "outputs": {"dashboard": dashboard_path, "rebalance_thesis": rebalance_outputs},
    }
    write_manifest(out_dir / "manifest.json", manifest_payload)
    print(f"  Saved dashboard: {dashboard_path}")
    print(f"  Saved manifest:  {out_dir / 'manifest.json'}")
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
    return parser.parse_args()


def main() -> None:
    _load_local_env()
    args = parse_args()
    run(
        args.mode,
        pathlib.Path(args.output_root),
        smoke_cusips=args.smoke_cusips,
        smoke_tickers=args.smoke_tickers,
    )


if __name__ == "__main__":
    main()
