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
    attribution,
    build_visible_versions_cache,
    manager_characteristics,
    marginal_ir,
    rebalance_trace,
    run_backtest,
)
from report import dashboard, interactive_results
from sweep import active_return_stream, deflated_sharpe, grid_eval, walk_forward


LIVE_CONFIG = {
    "identity": "YourName you@firm.com",
    "openfigi_key": None,
    "sec_history_start": "2013-10-01",
    "start": "2015-01-01",
    "end": "2026-05-31",
    # Broad-market total-return proxy. Use QQQ for a tighter growth-style proxy.
    "benchmark_ticker": "SPY",
    "min_aum": 0.5e9,
    "max_aum": 30e9,
    "max_holdings": 60,
    "max_put_weight": 0.10,
    "require_factors": False,
    "openfigi_cache_path": "openfigi_cache.parquet",
    "price_cache_path": "yfinance_close_cache.parquet",
    "price_source": "chart",
    "security_overrides_path": "data/security_overrides.csv",
    "exclude_fund_like_holdings": False,
    "fund_ticker_exclusions_path": "data/fund_ticker_exclusions.csv",
    "refresh_openfigi_metadata": False,
    "active_benchmark_source": "spy_holdings",
    "active_benchmark_weights_path": "data/processed/benchmark_weights_spy.parquet",
    "active_benchmark_max_stale_days": 45,
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
        require_metadata=bool(cfg.get("refresh_openfigi_metadata", False)),
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
    prices = da.fetch_prices(
        price_holdings.ticker.unique(),
        cfg["start"],
        cfg["end"],
        cache_path=cfg.get("price_cache_path"),
        price_source=cfg.get("price_source", "auto"),
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
            price_source=cfg.get("price_source", "auto"),
            require_full_window=True,
        ).iloc[:, 0]
        bench_ret.name = cfg["benchmark_ticker"]
    except Exception as exc:
        print(f"    [warn] benchmark fetch failed; benchmark disabled: {exc}")
        bench_ret = None
    print(f"    managers: {holdings.manager.nunique()}, tickers: {holdings.ticker.nunique()}")
    return holdings, prices, factors, None, None, bench_ret


def run_live_smoke(
    output_root: pathlib.Path,
    *,
    cusip_limit: int,
    ticker_limit: int,
    cfg: dict | None = None,
) -> pathlib.Path:
    cfg = dict(LIVE_CONFIG if cfg is None else cfg)
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


def _load_active_benchmark_weights_by_month(
    *,
    live_config: dict,
    months,
    cfg: BacktestConfig,
) -> dict[pd.Timestamp, pd.Series] | None:
    if not _needs_active_benchmark_weights(cfg.portfolio.idea_signal):
        return None
    if cfg.active_benchmark_source in {"visible_13f_aggregate", "13f_aggregate"}:
        return None
    import data_adapters as da

    path = pathlib.Path(live_config.get("active_benchmark_weights_path", ""))
    max_stale_days = int(live_config.get("active_benchmark_max_stale_days", 45))
    table = da.load_benchmark_weight_table(path)
    weights = da.benchmark_weights_by_month(table, months, max_stale_days=max_stale_days)
    print(
        f"    active benchmark: {cfg.active_benchmark_source} "
        f"{len(table['month_end'].drop_duplicates())} snapshots, "
        f"{len(weights)} monthly weights loaded from {path}"
    )
    return weights


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
        "portfolio": asdict(cfg.portfolio),
        "security_grouping": {
            "enabled": security_groups is not None,
            "max_issuer_weight": cfg.portfolio.max_issuer_weight,
            "unmapped_ticker_policy": "issuer_group defaults to ticker",
        },
        "missing_price_policy": cfg.missing_price_policy,
        "active_filter_status": {
            "value_tilt_configured": bool(cfg.universe.use_value_tilt),
            "value_tilt_active": bool(cfg.universe.use_value_tilt and value_scores is not None),
            "active_share_configured": bool(cfg.universe.use_active_share),
            "active_share_active": bool(cfg.universe.use_active_share and benchmark_weights is not None),
            "active_weight_signal": _needs_active_benchmark_weights(cfg.portfolio.idea_signal),
            "active_weight_benchmark": (
                cfg.active_benchmark_source
                if _needs_active_benchmark_weights(cfg.portfolio.idea_signal)
                else None
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
            f"max_holdings={u.max_holdings}, min_top{u.top_n_concentration}_weight={u.min_top_n_weight:.0%}, "
            f"turnover_q={u.turnover_quantile}, min_history_q={u.min_history_quarters}, "
            f"hedge_put_max={u.hedge_put_max_weight:.0%}, value_tilt_min={u.value_tilt_min_pctl:.0%}"
        ),
        "Portfolio": (
            f"idea_signal={p.idea_signal}, top_n_ideas={p.top_n_ideas}, "
            f"min_consensus_funds={p.min_consensus_funds}, holding_horizon_q={p.holding_horizon_q}, "
            f"max_portfolio_names={p.max_portfolio_names}, "
            f"max_name={p.max_name_weight:.1%}, max_issuer={p.max_issuer_weight:.1%}, "
            f"min_portfolio_names={p.min_portfolio_names}, "
            f"min_active_weight_holdings={p.min_active_weight_holdings}, "
            f"active_benchmark={cfg.active_benchmark_source}"
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
    )
    outputs: dict[str, str] = {}
    for name, df in trace.items():
        path = out_dir / f"rebalance_{name}_{label}.csv"
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


def value_unit_continuity_diagnostics(
    chars: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp = "2023-01-01",
) -> pd.DataFrame:
    """Check for suspicious AUM jumps around the historical SEC value-unit cutoff."""
    columns = [
        "manager",
        "prev_period_date",
        "period_date",
        "prev_filing_date",
        "filing_date",
        "prev_aum",
        "aum",
        "aum_ratio",
        "abs_log10_ratio",
        "suspicious_unit_jump",
    ]
    if chars is None or chars.empty:
        return pd.DataFrame(columns=columns)
    cutoff = pd.Timestamp(cutoff)
    latest = (
        chars.sort_values(["manager", "period_date", "filing_date", "accession_number"])
        .groupby(["manager", "period_date"], as_index=False)
        .tail(1)
        .sort_values(["manager", "period_date"])
    )
    rows: list[dict[str, Any]] = []
    for manager, g in latest.groupby("manager", sort=False):
        g = g.sort_values("period_date")
        prev = None
        for row in g.itertuples(index=False):
            if prev is not None and pd.Timestamp(prev.filing_date) < cutoff <= pd.Timestamp(row.filing_date):
                prev_aum = float(prev.aum)
                aum = float(row.aum)
                ratio = aum / prev_aum if prev_aum > 0 else np.nan
                abs_log10_ratio = abs(float(np.log10(ratio))) if ratio and np.isfinite(ratio) and ratio > 0 else np.nan
                rows.append({
                    "manager": manager,
                    "prev_period_date": pd.Timestamp(prev.period_date).date().isoformat(),
                    "period_date": pd.Timestamp(row.period_date).date().isoformat(),
                    "prev_filing_date": pd.Timestamp(prev.filing_date).date().isoformat(),
                    "filing_date": pd.Timestamp(row.filing_date).date().isoformat(),
                    "prev_aum": prev_aum,
                    "aum": aum,
                    "aum_ratio": ratio,
                    "abs_log10_ratio": abs_log10_ratio,
                    "suspicious_unit_jump": bool(pd.notna(ratio) and (ratio >= 50.0 or ratio <= 0.02)),
                })
            prev = row
    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        out = out.sort_values("abs_log10_ratio", ascending=False, na_position="last")
    return out


def _pct(value: float) -> str:
    return f"{value:.1%}" if pd.notna(value) else "n/a"


def _rebalance_summary_stats(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {"rebalance_months": 0}
    out: dict[str, Any] = {"rebalance_months": int(len(summary))}
    numeric_cols = [
        "selected_managers",
        "visible_managers",
        "stale_managers_dropped",
        "stale_filing_managers",
        "stale_period_managers",
        "active_eligible_managers",
        "zero_contributor_managers",
        "raw_idea_rows",
        "raw_idea_names",
        "consensus_idea_names",
        "effective_names",
        "target_names",
        "target_names_before_caps",
        "carried_names",
        "turnover_one_way",
        "cost_bps",
        "max_weight",
        "issuer_groups",
        "max_issuer_weight",
        "top5_weight",
        "top10_weight",
        "effective_number",
        "traded_names",
        "buy_names",
        "sell_names",
    ]
    for col in numeric_cols:
        if col not in summary:
            continue
        s = pd.to_numeric(summary[col], errors="coerce").dropna()
        if s.empty:
            continue
        out[f"avg_{col}"] = float(s.mean())
        out[f"max_{col}"] = float(s.max())
    last = summary.iloc[-1]
    for col in ["name_cap_feasible", "issuer_cap_feasible"]:
        if col in summary:
            s = summary[col].astype(bool)
            out[f"{col}_months"] = int(s.sum())
            out[f"{col}_ratio"] = float(s.mean()) if len(s) else float("nan")
    if "valid_rebalance" in summary:
        s = summary["valid_rebalance"].astype(bool)
        out["valid_rebalance_months"] = int(s.sum())
        out["valid_rebalance_ratio"] = float(s.mean()) if len(s) else float("nan")
        out["invalid_rebalance_months"] = int((~s).sum())
    if "effective_names" in summary:
        invested = pd.to_numeric(summary["effective_names"], errors="coerce").fillna(0).gt(0)
        out["invested_month_frac"] = float(invested.mean()) if len(invested) else float("nan")
    if {"zero_contributor_managers", "selected_managers"}.issubset(summary.columns):
        zero = pd.to_numeric(summary["zero_contributor_managers"], errors="coerce").fillna(0).sum()
        selected = pd.to_numeric(summary["selected_managers"], errors="coerce").fillna(0).sum()
        out["zero_contributor_manager_frac"] = float(zero / selected) if selected > 0 else float("nan")
    out["last_rebalance_month"] = str(last.get("rebalance_month", ""))
    out["last_effective_names"] = int(last.get("effective_names", 0) or 0)
    out["last_turnover_one_way"] = float(last.get("turnover_one_way", 0.0) or 0.0)
    out["last_max_weight"] = float(last.get("max_weight", 0.0) or 0.0)
    out["last_max_issuer_weight"] = float(last.get("max_issuer_weight", 0.0) or 0.0)
    out["last_top_holdings"] = str(last.get("top_holdings", ""))
    out["last_top_issuer_exposures"] = str(last.get("top_issuer_exposures", ""))
    out["last_multi_class_exposures"] = str(last.get("multi_class_exposures", ""))
    return out


def _print_rebalance_summary(stats: dict[str, Any]) -> None:
    if not stats or stats.get("rebalance_months", 0) == 0:
        print("  Rebalance summary: no rebalance months")
        return
    print("\n  Rebalance Summary")
    print(f"  months                 {stats['rebalance_months']}")
    print(
        "  holdings avg/max       "
        f"{stats.get('avg_effective_names', float('nan')):.1f}/"
        f"{stats.get('max_effective_names', float('nan')):.0f}"
    )
    if "valid_rebalance_ratio" in stats:
        print(
            "  valid/invested months  "
            f"{_pct(stats.get('valid_rebalance_ratio', float('nan')))}/"
            f"{_pct(stats.get('invested_month_frac', float('nan')))}"
        )
    if "zero_contributor_manager_frac" in stats:
        print(
            "  zero contributor mgrs  "
            f"{_pct(stats.get('zero_contributor_manager_frac', float('nan')))}"
        )
    if "avg_stale_managers_dropped" in stats:
        print(
            "  stale mgrs dropped avg/max "
            f"{stats.get('avg_stale_managers_dropped', float('nan')):.1f}/"
            f"{stats.get('max_stale_managers_dropped', float('nan')):.0f}"
        )
    print(
        "  traded names avg/max   "
        f"{stats.get('avg_traded_names', float('nan')):.1f}/"
        f"{stats.get('max_traded_names', float('nan')):.0f}"
    )
    print(
        "  one-way turnover avg/max "
        f"{_pct(stats.get('avg_turnover_one_way', float('nan')))}/"
        f"{_pct(stats.get('max_turnover_one_way', float('nan')))}"
    )
    print(
        "  max weight avg/max     "
        f"{_pct(stats.get('avg_max_weight', float('nan')))}/"
        f"{_pct(stats.get('max_max_weight', float('nan')))}"
    )
    print(
        "  max issuer avg/max     "
        f"{_pct(stats.get('avg_max_issuer_weight', float('nan')))}/"
        f"{_pct(stats.get('max_max_issuer_weight', float('nan')))}"
    )
    print(
        "  issuer groups avg/max  "
        f"{stats.get('avg_issuer_groups', float('nan')):.1f}/"
        f"{stats.get('max_issuer_groups', float('nan')):.0f}"
    )
    if "name_cap_feasible_ratio" in stats or "issuer_cap_feasible_ratio" in stats:
        print(
            "  cap feasible months    "
            f"name={_pct(stats.get('name_cap_feasible_ratio', float('nan')))} | "
            f"issuer={_pct(stats.get('issuer_cap_feasible_ratio', float('nan')))}"
        )
    print(
        "  top10 weight avg/max   "
        f"{_pct(stats.get('avg_top10_weight', float('nan')))}/"
        f"{_pct(stats.get('max_top10_weight', float('nan')))}"
    )
    print(
        "  cost bps avg/max       "
        f"{stats.get('avg_cost_bps', float('nan')):.2f}/"
        f"{stats.get('max_cost_bps', float('nan')):.2f}"
    )
    print(
        "  latest rebalance       "
        f"{stats.get('last_rebalance_month')} | "
        f"names={stats.get('last_effective_names')} | "
        f"turnover={_pct(stats.get('last_turnover_one_way', float('nan')))} | "
        f"max_weight={_pct(stats.get('last_max_weight', float('nan')))}"
    )
    top = stats.get("last_top_holdings")
    if top:
        print(f"  latest top holdings    {top}")
    issuers = stats.get("last_top_issuer_exposures")
    if issuers:
        print(f"  latest top issuers     {issuers}")
    multi = stats.get("last_multi_class_exposures")
    if multi:
        print(f"  latest multi-class     {multi}")


def _default_run_configs() -> tuple[BacktestConfig, BacktestConfig, dict, int, int]:
    cfg_a = BacktestConfig(
        universe=UniverseConfig(
            min_aum=0.5e9,
            max_aum=5e9,
            use_concentration=False,
            min_top_n_weight=0.50,
            max_holdings=40,
            turnover_quantile=0.34,
            hedge_put_max_weight=0.05,
            value_tilt_min_pctl=0.50,
            min_history_quarters=4,
        ),
        portfolio=PortfolioConfig(
            idea_signal="active_weight",
            top_n_ideas=5,
            min_consensus_funds=2,
            min_portfolio_names=10,
            max_portfolio_names=30,
            holding_horizon_q=1,
            max_name_weight=0.10,
            max_issuer_weight=0.15,
            min_active_weight_holdings=10,
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
    axes = {
        ("universe", "aum_band"): [
            ("0.5-5B", 0.5e9, 5e9),
            ("15-30B", 15e9, 30e9),
        ],
        ("portfolio", "idea_signal"): [
            "active_weight",
            "active_weight_change",
            "active_weight_initiation",
        ],
        ("portfolio", "top_n_ideas"): [5, 10],
        ("portfolio", "min_consensus_funds"): [2, 5],
        ("portfolio", "holding_horizon_q"): [0, 1],
        ("portfolio", "min_portfolio_names"): [10],
        ("portfolio", "max_portfolio_names"): [30],
        ("portfolio", "min_active_weight_holdings"): [10],
        ("universe", "use_concentration"): [False, True],
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
        print(f"  active benchmark      {live_cfg.get('active_benchmark_source')}")
        if live_cfg.get("active_benchmark_source") not in {"visible_13f_aggregate", "13f_aggregate"}:
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
            f"min_aum={u.min_aum:.0f}, max_aum={u.max_aum:.0f}, "
            f"max_holdings={u.max_holdings}, min_top{u.top_n_concentration}_weight={u.min_top_n_weight:.0%}, "
            f"turnover_q={u.turnover_quantile}, min_history_q={u.min_history_quarters}, "
            f"max_stale_filing_m={u.max_stale_filing_months}, max_stale_period_m={u.max_stale_period_months}, "
            f"hedge_put_max={u.hedge_put_max_weight:.0%}, value_tilt_min={u.value_tilt_min_pctl:.0%}"
        )
        print(
            "  thesis portfolio      "
            f"idea_signal={p.idea_signal}, top_n_ideas={p.top_n_ideas}, "
            f"min_consensus={p.min_consensus_funds}, holding_horizon_q={p.holding_horizon_q}, "
            f"min_portfolio_names={p.min_portfolio_names}, max_portfolio_names={p.max_portfolio_names}, "
            f"max_name={p.max_name_weight:.1%}, max_issuer={p.max_issuer_weight:.1%}, "
            f"min_active_weight_holdings={p.min_active_weight_holdings}, "
            f"missing_price_policy={cfg_a.missing_price_policy}, cost={c.bps_per_side:.1f}bps"
        )
        print(f"  thesis active bench   {cfg_a.active_benchmark_source}")
    if cfg_b is not None:
        print(
            "  placebo portfolio     "
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
    cfg_a = replace(
        cfg_a,
        active_benchmark_source=(
            live_config.get("active_benchmark_source", "visible_13f_aggregate")
            if mode == "live"
            else "visible_13f_aggregate"
        ),
    )
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
        holdings, prices, factors, value_scores, bench_w, bench_ret = build_synthetic_data()
    elif mode == "live-smoke":
        return run_live_smoke(
            output_root,
            cusip_limit=smoke_cusips,
            ticker_limit=smoke_tickers,
            cfg=live_config,
        )
    else:
        holdings, prices, factors, value_scores, bench_w, bench_ret = build_live_data(live_config)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"[output] Writing incremental reports under {out_dir}")

    print("[2/6] Computing per-manager characteristics")
    t_step = time.perf_counter()
    chars = manager_characteristics(holdings, bench_w)
    print(f"    {len(chars)} manager-filing-version rows in {time.perf_counter() - t_step:.1f}s")
    t_step = time.perf_counter()
    visible_cache = build_visible_versions_cache(chars, prices.index)
    print(f"    {len(visible_cache)} month-end visible-version snapshots in {time.perf_counter() - t_step:.1f}s")
    security_groups = load_security_groups(
        prices.columns,
        live_config.get("security_overrides_path", "data/security_overrides.csv"),
    )
    active_benchmark_weights_by_month = _load_active_benchmark_weights_by_month(
        live_config=live_config,
        months=prices.index,
        cfg=cfg_a,
    )

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
        capture_rebalance=True,
    )
    print(f"    thesis backtest done in {time.perf_counter() - t_step:.1f}s")
    t_placebo = time.perf_counter()
    print("    placebo backtest running")
    ret_b = run_backtest(holdings, prices, cfg_b, value_scores, bench_w, chars, visible_cache, security_groups)
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
        sweep_outputs = {}
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
        print(f"  Saved interactive HTML: {sweep_outputs['interactive_html']}")

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
        },
        "metrics": {"thesis": att_a, "placebo": att_b, "dsr": dsr},
        "rebalance_summary_stats": rebalance_outputs.get("summary_stats"),
        "outputs": {"dashboard": dashboard_path, "rebalance_thesis": rebalance_outputs, "sweep": sweep_outputs},
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
        choices=["visible_13f_aggregate", "spy_holdings"],
        default=None,
        help="Benchmark used for active_weight signals. Live default is spy_holdings.",
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
    )


if __name__ == "__main__":
    main()
