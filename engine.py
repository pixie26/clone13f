"""
13F-clone backtest engine v2 — pure-pandas core (no network, fully testable).

v2 adds the portfolio-layer signals that v1 was missing, each as a config axis so
the SWEEP can treat them as hypotheses under test (see sweep.py):
  - idea_signal : "level" | "change" | "initiation"   (trade Δposition, not just level)
  - min_consensus_funds : gate names by independent cross-fund agreement
  - true active share   : 0.5*Σ|w_fund - w_bench|  (vs a passed benchmark, not proxied)
  - holding_horizon_q   : carry a name up to N quarters after it leaves the target

Standardized inputs unchanged (see data_adapters.py): holdings, prices, factors,
optional value_scores, optional benchmark_weights (Series ticker->weight).

Note: the `prices` argument is a monthly returns matrix, not price levels.
The name is kept for call-site stability.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm

from manager_classifier import apply_manager_overrides, filter_selected_versions


# --------------------------------------------------------------------------- #
@dataclass
class UniverseConfig:
    min_aum: float = 1e9
    max_aum: float = 30e9
    top_n_concentration: int = 10
    min_top_n_weight: float = 0.50
    max_holdings: int = 40  # hard cap; does not become optional when Top-N concentration passes
    turnover_quantile: float = 0.34
    min_history_quarters: int = 4
    max_stale_filing_months: int | None = 6
    max_stale_period_months: int | None = 6
    hedge_put_max_weight: float = 0.05
    value_tilt_min_pctl: float = 0.50
    min_active_share: float = 0.60          # used only if use_active_share & benchmark given
    # ablation toggles
    use_size_band: bool = True
    use_concentration: bool = True
    use_low_turnover: bool = True
    use_hedge_filter: bool = True
    use_value_tilt: bool = True
    use_active_share: bool = False


@dataclass
class PortfolioConfig:
    top_n_ideas: int = 8
    idea_signal: str = "level"              # "level" | "change" | "initiation" | active-weight variants
    consensus_weight: bool = True
    idea_aggregation: str | None = None      # "manager_equal" | "score" | "manager_count" | "equal_name"
    min_consensus_funds: int = 1            # drop names held by < this many in-universe funds
    min_portfolio_names: int = 0            # mark a rebalance invalid/cash if fewer target names survive
    max_portfolio_names: int | None = None  # cap final aggregate target before weight caps
    max_name_weight: float = 0.05
    max_issuer_weight: float = 0.075
    holding_horizon_q: int = 0              # 0 = full rebalance; N = carry N extra quarters
    min_active_weight_holdings: int = 10    # active_weight needs enough book breadth to be meaningful


@dataclass
class CostConfig:
    bps_per_side: float = 15.0


@dataclass
class BacktestConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    manager_filter_mode: str = "all"        # "all" | "exclude_dirty" | "dedicated_like"
    active_benchmark_source: str = "visible_13f_aggregate"
    # "exit" liquidates a held name when its monthly return is missing, then
    # redeploys into priced survivors. This reduces complete-window survivorship
    # bias from partial yfinance histories, but is still a CRSP/WRDS placeholder.
    missing_price_policy: str = "exit"      # "exit" | "zero" | "raise"


# --------------------------------------------------------------------------- #
def _book_weights(df: pd.DataFrame) -> pd.Series:
    longs = df[df.get("sec_type", "SH").fillna("SH") == "SH"] if "sec_type" in df else df
    v = longs.groupby("ticker")["value"].sum()
    tot = v.sum()
    return v / tot if tot > 0 else v


def _active_share(w: pd.Series, wb: pd.Series | None) -> float:
    if wb is None or len(wb) == 0:
        return np.nan
    alln = w.index.union(wb.index)
    a = w.reindex(alln).fillna(0.0)
    b = wb.reindex(alln).fillna(0.0)
    b = b / b.sum() if b.sum() > 0 else b
    return float(0.5 * (a - b).abs().sum())


def _aggregate_book_weights(books) -> pd.Series:
    """Equal-manager aggregate of visible 13F books, used as a PIT benchmark proxy."""
    total = pd.Series(dtype=float)
    n = 0
    for book in books:
        if book is None or len(book) == 0:
            continue
        w = pd.Series(book, dtype=float)
        w = w[w > 0]
        if w.empty:
            continue
        total = total.add(w / w.sum(), fill_value=0.0)
        n += 1
    if n == 0:
        return total
    total = total / n
    return total / total.sum() if total.sum() > 0 else total


_ACTIVE_WEIGHT_SIGNALS = {
    "active_weight",
    "active_weight_change",
    "active_weight_initiation",
}
_CPS_IR_SIGNALS = {
    "cps_ir",
    "cps_ir_change",
    "cps_ir_initiation",
}


def _needs_active_benchmark_weights(idea_signal: str) -> bool:
    return idea_signal in _ACTIVE_WEIGHT_SIGNALS or idea_signal in _CPS_IR_SIGNALS


def _needs_idiosyncratic_vol(idea_signal: str) -> bool:
    return idea_signal in _CPS_IR_SIGNALS


def _active_benchmark_source(cfg: BacktestConfig | str | None) -> str:
    if isinstance(cfg, BacktestConfig):
        return str(cfg.active_benchmark_source or "visible_13f_aggregate")
    if cfg is None:
        return "visible_13f_aggregate"
    return str(cfg)


def _uses_visible_13f_active_benchmark(cfg: BacktestConfig | str | None) -> bool:
    return _active_benchmark_source(cfg) in {"visible_13f_aggregate", "13f_aggregate"}


def _uses_manager_held_mcap_benchmark(cfg: BacktestConfig | str | None) -> bool:
    return _active_benchmark_source(cfg) == "manager_held_mcap"


def _active_benchmark_for_month(
    cfg: BacktestConfig,
    month,
    latest_versions: pd.DataFrame | None = None,
    active_benchmark_weights_by_month: dict[pd.Timestamp, pd.Series] | None = None,
) -> pd.Series | None:
    if not _needs_active_benchmark_weights(cfg.portfolio.idea_signal):
        return None
    source = _active_benchmark_source(cfg)
    if _uses_visible_13f_active_benchmark(source):
        if latest_versions is None:
            raise ValueError("visible_13f_aggregate active benchmark requires latest_versions")
        return _aggregate_book_weights(latest_versions["bw"]) if not latest_versions.empty else pd.Series(dtype=float)
    if active_benchmark_weights_by_month is None:
        raise ValueError(
            f"{source} active benchmark requires active_benchmark_weights_by_month; "
            "provide a point-in-time monthly benchmark weight table."
        )
    weights = active_benchmark_weights_by_month.get(pd.Timestamp(month))
    if weights is None or len(weights) == 0:
        raise ValueError(f"{source} active benchmark has no weights for {pd.Timestamp(month).date()}")
    return weights


def _require_active_benchmark_weights(
    cfg: PortfolioConfig,
    active_benchmark_weights: pd.Series | None,
) -> pd.Series | None:
    if not _needs_active_benchmark_weights(cfg.idea_signal):
        return None
    if active_benchmark_weights is None or len(active_benchmark_weights) == 0:
        raise ValueError(
            f"{cfg.idea_signal} requires PIT active_benchmark_weights; "
            "do not fall back to selected-manager aggregate weights."
        )
    bench = pd.Series(active_benchmark_weights, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    bench = bench[bench > 0]
    if bench.empty or bench.sum() <= 0:
        raise ValueError(f"{cfg.idea_signal} requires positive PIT active_benchmark_weights")
    return bench / bench.sum()


def _require_idiosyncratic_vol(
    cfg: PortfolioConfig,
    idiosyncratic_vol: pd.Series | None,
) -> pd.Series | None:
    if not _needs_idiosyncratic_vol(cfg.idea_signal):
        return None
    if idiosyncratic_vol is None:
        raise ValueError(
            f"{cfg.idea_signal} requires PIT idiosyncratic_vol_by_month; "
            "do not fill missing residual-vol coverage with zero."
        )
    if len(idiosyncratic_vol) == 0:
        return pd.Series(dtype=float)
    vol = pd.Series(idiosyncratic_vol, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    vol = vol[vol > 0]
    if vol.empty:
        raise ValueError(f"{cfg.idea_signal} requires positive PIT idiosyncratic volatility")
    return vol


def _manager_held_mcap_weights(
    manager_weights: pd.Series,
    market_caps: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Align a manager book to covered names and market-cap weight that subset."""
    manager = pd.Series(manager_weights, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    caps = pd.Series(market_caps, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    manager = manager[manager > 0]
    caps = caps[caps > 0]
    covered = manager.index.intersection(caps.index)
    if len(covered) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    manager = manager.reindex(covered)
    manager = manager / manager.sum()
    benchmark = caps.reindex(covered)
    benchmark = benchmark / benchmark.sum()
    return manager.sort_index(), benchmark.sort_index()


def _idiosyncratic_vol_for_month(
    cfg: BacktestConfig,
    month,
    idiosyncratic_vol_by_month: dict[pd.Timestamp, pd.Series] | None = None,
) -> pd.Series | None:
    if not _needs_idiosyncratic_vol(cfg.portfolio.idea_signal):
        return None
    if idiosyncratic_vol_by_month is None:
        raise ValueError(
            f"{cfg.portfolio.idea_signal} requires idiosyncratic_vol_by_month; "
            "build a PIT residual-vol cache before running CPS-IR signals."
        )
    vol = idiosyncratic_vol_by_month.get(pd.Timestamp(month))
    if vol is None:
        raise ValueError(f"{cfg.portfolio.idea_signal} has no idiosyncratic vol for {pd.Timestamp(month).date()}")
    return vol


def _capm_residual_vol_window(
    returns_window: pd.DataFrame,
    factors_window: pd.DataFrame,
    *,
    min_obs: int,
) -> pd.Series:
    """Vectorized CAPM residual volatility for one trailing window.

    The input returns are monthly returns. The output is annualized residual
    volatility. Missing ticker-month returns are left missing and each ticker
    must independently pass min_obs.
    """
    if returns_window.empty or "MKT" not in factors_window:
        return pd.Series(dtype=float)
    y = returns_window.astype(float)
    f = factors_window.reindex(y.index).copy()
    x = pd.to_numeric(f["MKT"], errors="coerce")
    rf = pd.to_numeric(f["RF"], errors="coerce") if "RF" in f else pd.Series(0.0, index=f.index)
    y = y.sub(rf, axis=0)

    valid = y.notna() & x.notna().to_numpy()[:, None]
    n = valid.sum(axis=0).astype(float)
    enough = n >= int(min_obs)
    if not bool(enough.any()):
        return pd.Series(dtype=float)

    x_arr = x.to_numpy(dtype=float)
    x_df = pd.DataFrame(
        np.repeat(x_arr[:, None], y.shape[1], axis=1),
        index=y.index,
        columns=y.columns,
    )
    x_masked = x_df.where(valid)
    y_masked = y.where(valid)
    x_mean = x_masked.sum(axis=0) / n
    y_mean = y_masked.sum(axis=0) / n
    xy_mean = (x_masked * y_masked).sum(axis=0) / n
    xx_mean = (x_masked * x_masked).sum(axis=0) / n
    x_var = xx_mean - x_mean * x_mean
    usable = enough & x_var.gt(0)
    if not bool(usable.any()):
        return pd.Series(dtype=float)

    beta = (xy_mean - x_mean * y_mean) / x_var
    alpha = y_mean - beta * x_mean
    fitted = x_df.mul(beta, axis=1).add(alpha, axis=1)
    resid = (y - fitted).where(valid)
    vol = resid.std(axis=0, skipna=True, ddof=2) * np.sqrt(12)
    return vol[usable.reindex(vol.index).fillna(False)].replace([np.inf, -np.inf], np.nan).dropna()


def build_idiosyncratic_vol_cache(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    months=None,
    *,
    window_months: int = 24,
    min_obs: int = 24,
    floor: float = 0.10,
    cap: float = 0.80,
    winsor_lower: float = 0.05,
    winsor_upper: float = 0.95,
    progress=None,
) -> dict[pd.Timestamp, pd.Series]:
    """Build month -> PIT CAPM residual-vol series for CPS-style idea ranking.

    Uses only returns strictly before each month. The 24-month default and
    floor/cap/winsorization are pragmatic guardrails, not calibrated academic
    constants; they are surfaced in run manifests.
    """
    if factors is None or factors.empty or "MKT" not in factors:
        raise ValueError("CAPM idiosyncratic vol requires factors with an MKT column")
    rets = prices.replace([np.inf, -np.inf], np.nan).copy()
    rets.index = pd.to_datetime(rets.index).to_period("M").to_timestamp("M")
    rets = rets.sort_index()
    fac = factors.replace([np.inf, -np.inf], np.nan).copy()
    fac.index = pd.to_datetime(fac.index).to_period("M").to_timestamp("M")
    fac = fac.sort_index()
    if months is None:
        month_index = rets.index
    else:
        month_index = pd.Index(pd.to_datetime(months)).to_period("M").to_timestamp("M").sort_values()
    out: dict[pd.Timestamp, pd.Series] = {}
    for month_number, month in enumerate(month_index, start=1):
        hist_idx = rets.index[rets.index < pd.Timestamp(month)]
        if len(hist_idx) <= 0:
            out[pd.Timestamp(month)] = pd.Series(dtype=float)
            if progress is not None and (month_number == 1 or month_number % 12 == 0 or month_number == len(month_index)):
                progress(f"idio-vol month {month_number}/{len(month_index)} asof={pd.Timestamp(month).date()} covered=0")
            continue
        hist_idx = hist_idx[-int(window_months):]
        vol = _capm_residual_vol_window(
            rets.loc[hist_idx],
            fac.reindex(hist_idx),
            min_obs=int(min_obs),
        )
        if not vol.empty:
            if winsor_lower is not None and winsor_upper is not None and len(vol) >= 5:
                lo = vol.quantile(float(winsor_lower))
                hi = vol.quantile(float(winsor_upper))
                if pd.notna(lo) and pd.notna(hi) and hi >= lo:
                    vol = vol.clip(lower=float(lo), upper=float(hi))
            if floor is not None:
                vol = vol.clip(lower=float(floor))
            if cap is not None:
                vol = vol.clip(upper=float(cap))
            vol = vol[vol > 0]
        out[pd.Timestamp(month)] = vol.sort_index()
        if progress is not None and (month_number == 1 or month_number % 12 == 0 or month_number == len(month_index)):
            progress(
                f"idio-vol month {month_number}/{len(month_index)} "
                f"asof={pd.Timestamp(month).date()} covered={len(vol)}"
            )
    return out


def _versioned_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    missing_accession = "accession_number" not in holdings.columns
    missing_submission = "submission_type" not in holdings.columns
    if not missing_accession and not missing_submission:
        return holdings
    h = holdings.copy()
    if missing_accession:
        h["accession_number"] = (
            h["manager"].astype("string") + "|" +
            h["period_date"].astype("string") + "|" +
            h["filing_date"].astype("string")
        )
    if missing_submission:
        h["submission_type"] = pd.NA
    return h


def raw_filing_put_weights(holdings: pd.DataFrame) -> pd.DataFrame:
    """Compute exact-version PUT exposure before mapping or equity filtering."""
    holdings = _versioned_holdings(holdings)
    keys = ["manager", "period_date", "filing_date", "accession_number"]
    required = set(keys).union({"value", "sec_type"})
    missing = sorted(required.difference(holdings.columns))
    if missing:
        raise ValueError(f"raw_filing_put_weights missing columns: {missing}")
    if holdings.empty:
        return pd.DataFrame(columns=keys + ["filing_put_weight"])

    work = holdings.loc[:, keys + ["value", "sec_type"]].copy()
    work["value"] = pd.to_numeric(work["value"], errors="raise")
    work["_put_value"] = work["value"].where(
        work["sec_type"].fillna("SH").astype(str).str.upper().eq("PUT"),
        0.0,
    )
    metrics = (
        work.groupby(keys, dropna=False, sort=False)
        .agg(total_value=("value", "sum"), put_value=("_put_value", "sum"))
        .reset_index()
    )
    metrics["filing_put_weight"] = (
        metrics["put_value"] / metrics["total_value"].replace(0.0, np.nan)
    )
    return metrics[keys + ["filing_put_weight"]]


def _visible_manager_versions(chars: pd.DataFrame, asof, managers=None) -> pd.DataFrame:
    known = chars[chars["filing_date"] <= asof]
    if managers is not None:
        known = known[known["manager"].isin(managers)]
    if known.empty:
        return known
    return (known.sort_values(["manager", "period_date", "filing_date", "accession_number"])
                 .groupby("manager", as_index=False)
                 .tail(1))


def _filter_fresh_versions(latest_versions: pd.DataFrame, asof, cfg: UniverseConfig) -> tuple[pd.DataFrame, dict]:
    """Drop latest-visible manager books that are too stale for a live decision."""
    diag = {
        "visible_managers": int(len(latest_versions)),
        "stale_managers_dropped": 0,
        "stale_filing_managers": 0,
        "stale_period_managers": 0,
    }
    if latest_versions.empty:
        return latest_versions, diag

    latest = latest_versions.copy()
    keep = pd.Series(True, index=latest.index)
    asof_ts = pd.Timestamp(asof)

    if cfg.max_stale_filing_months is not None:
        filing_cutoff = asof_ts - pd.DateOffset(months=int(cfg.max_stale_filing_months))
        stale_filing = pd.to_datetime(latest["filing_date"]) < filing_cutoff
        diag["stale_filing_managers"] = int(stale_filing.sum())
        keep &= ~stale_filing

    if cfg.max_stale_period_months is not None:
        period_cutoff = asof_ts - pd.DateOffset(months=int(cfg.max_stale_period_months))
        stale_period = pd.to_datetime(latest["period_date"]) < period_cutoff
        diag["stale_period_managers"] = int(stale_period.sum())
        keep &= ~stale_period

    fresh = latest.loc[keep].reset_index(drop=True)
    diag["stale_managers_dropped"] = int(len(latest) - len(fresh))
    return fresh, diag


def _add_freshness_audit_columns(
    frame: pd.DataFrame,
    asof,
    cfg: UniverseConfig,
) -> pd.DataFrame:
    """Attach the exact stale cutoffs and decisions used for version auditing."""
    audit = frame.copy()
    asof_ts = pd.Timestamp(asof).normalize()
    filing_cutoff = (
        asof_ts - pd.DateOffset(months=int(cfg.max_stale_filing_months))
        if cfg.max_stale_filing_months is not None else pd.NaT
    )
    period_cutoff = (
        asof_ts - pd.DateOffset(months=int(cfg.max_stale_period_months))
        if cfg.max_stale_period_months is not None else pd.NaT
    )
    audit["filing_date_cutoff"] = filing_cutoff
    audit["period_date_cutoff"] = period_cutoff
    audit["pass_filing_fresh"] = (
        True
        if pd.isna(filing_cutoff)
        else pd.to_datetime(audit["filing_date"]).ge(filing_cutoff)
    )
    audit["pass_period_fresh"] = (
        True
        if pd.isna(period_cutoff)
        else pd.to_datetime(audit["period_date"]).ge(period_cutoff)
    )
    audit["pass_freshness"] = audit["pass_filing_fresh"] & audit["pass_period_fresh"]
    return audit


def _selection_cache_diagnostics(selected_versions: pd.DataFrame) -> dict:
    visible = selected_versions.attrs.get("visible_managers")
    return {
        "visible_managers": int(visible) if visible is not None else int(len(selected_versions)),
        "stale_managers_dropped": int(selected_versions.attrs.get("stale_managers_dropped", 0)),
        "stale_filing_managers": int(selected_versions.attrs.get("stale_filing_managers", 0)),
        "stale_period_managers": int(selected_versions.attrs.get("stale_period_managers", 0)),
    }


def build_visible_versions_cache(chars: pd.DataFrame, months, *, progress=None) -> dict[pd.Timestamp, pd.DataFrame]:
    """Cache latest visible manager filing versions for each month/asof."""
    ordered_months = pd.Index(months).sort_values()
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for month_number, month in enumerate(ordered_months, start=1):
        asof = pd.Timestamp(month)
        out[asof] = _visible_manager_versions(chars, asof)
        if progress is not None and (month_number == 1 or month_number % 12 == 0 or month_number == len(ordered_months)):
            progress(
                f"visible snapshot {month_number}/{len(ordered_months)} "
                f"asof={asof.date()} managers={len(out[asof])}"
            )
    return out


def manager_characteristics(holdings: pd.DataFrame,
                            benchmark_weights: pd.Series | None = None,
                            *,
                            filing_put_weights: pd.DataFrame | None = None,
                            progress=None) -> pd.DataFrame:
    started = time.perf_counter()
    holdings = _versioned_holdings(holdings)
    group_cols = ["manager", "period_date", "filing_date", "accession_number"]
    required = set(group_cols).union({"ticker", "value", "submission_type"})
    missing = sorted(required.difference(holdings.columns))
    if missing:
        raise ValueError(f"manager_characteristics missing columns: {missing}")
    work_cols = group_cols + ["ticker", "value", "submission_type"]
    if "sec_type" in holdings:
        work_cols.append("sec_type")
    work = holdings.loc[:, work_cols].copy()
    work["value"] = pd.to_numeric(work["value"], errors="raise")
    sec_type = work["sec_type"].fillna("SH") if "sec_type" in work else pd.Series("SH", index=work.index)
    work["_sh_value"] = work["value"].where(sec_type.eq("SH"), 0.0)
    work["_put_value"] = work["value"].where(sec_type.eq("PUT"), 0.0)
    stats = (
        work.groupby(group_cols, dropna=False, sort=False)
        .agg(
            submission_type=("submission_type", "first"),
            aum=("_sh_value", "sum"),
            total_value=("value", "sum"),
            put_value=("_put_value", "sum"),
        )
        .reset_index()
    )
    group_count = len(stats)
    if progress is not None:
        progress(
            f"characteristics input rows={len(holdings):,}, managers={holdings['manager'].nunique():,}, "
            f"filing_versions={group_count:,}"
        )

    ticker_values = (
        work.loc[sec_type.eq("SH")]
        .groupby(group_cols + ["ticker"], dropna=False, sort=False)["value"]
        .sum()
    )
    if not ticker_values.empty:
        book_totals = ticker_values.groupby(level=list(range(len(group_cols)))).transform("sum")
        ticker_weights = ticker_values.where(book_totals.le(0), ticker_values / book_totals)
    else:
        ticker_weights = ticker_values.astype(float)

    def normalized_key(values) -> tuple:
        values = values if isinstance(values, tuple) else (values,)
        return tuple("<NA>" if pd.isna(value) else value for value in values)

    books: dict[tuple, pd.Series] = {}
    book_groups = ticker_weights.groupby(level=list(range(len(group_cols))), sort=False)
    for book_number, (key, weights) in enumerate(book_groups, start=1):
        ticker_index = weights.index.get_level_values("ticker")
        books[normalized_key(key)] = pd.Series(weights.to_numpy(dtype=float), index=ticker_index, dtype=float)
        if progress is not None and (book_number % 2500 == 0 or book_number == book_groups.ngroups):
            progress(
                f"characteristics books {book_number:,}/{group_count:,} "
                f"({time.perf_counter() - started:.1f}s)"
            )

    rows = []
    for record in stats.itertuples(index=False):
        key = tuple(getattr(record, col) for col in group_cols)
        w = books.get(normalized_key(key), pd.Series(dtype=float))
        tot = float(record.total_value)
        put_w = float(record.put_value) / tot if tot > 0 else 0.0
        top10 = w.sort_values(ascending=False).head(10).sum()
        rows.append(dict(manager=record.manager, period_date=record.period_date, filing_date=record.filing_date,
                         accession_number=record.accession_number,
                         submission_type=record.submission_type,
                         aum=float(record.aum), n_holdings=int((w > 0).sum()), top10_weight=top10,
                         put_weight=put_w, active_share=_active_share(w, benchmark_weights),
                         bw=w))
    cols = [
        "manager", "period_date", "filing_date", "accession_number",
        "submission_type", "aum", "n_holdings", "top10_weight", "put_weight",
        "active_share", "bw",
    ]
    if not rows:
        extra = ["filing_put_weight"] if filing_put_weights is not None else []
        return pd.DataFrame(columns=cols + extra + ["prev_bw", "turnover", "hist_q"])

    chars = (pd.DataFrame(rows)
             .sort_values(["manager", "period_date", "filing_date", "accession_number"])
             .reset_index(drop=True))
    if filing_put_weights is not None:
        metric_keys = ["manager", "period_date", "filing_date", "accession_number"]
        required_metrics = set(metric_keys).union({"filing_put_weight"})
        missing_metrics = sorted(required_metrics.difference(filing_put_weights.columns))
        if missing_metrics:
            raise ValueError(f"filing_put_weights missing columns: {missing_metrics}")
        metrics = filing_put_weights.loc[:, metric_keys + ["filing_put_weight"]].copy()
        chars = chars.merge(
            metrics,
            on=metric_keys,
            how="left",
            validate="one_to_one",
            indicator="_filing_put_metric_merge",
        )
        # A matched zero-total-value filing legitimately has an undefined PUT
        # ratio (NaN). Distinguish that from a missing exact-version metric;
        # the hedge screen later treats NaN fail-closed rather than fabricating
        # a zero exposure.
        missing_versions = chars["_filing_put_metric_merge"].ne("both")
        if missing_versions.any():
            sample = chars.loc[missing_versions, metric_keys].head(5).to_dict(orient="records")
            raise ValueError(
                "missing raw filing PUT weights for manager versions; "
                f"count={int(missing_versions.sum())}, sample={sample}"
            )
        chars = chars.drop(columns="_filing_put_metric_merge")
    chars["hist_q"] = chars.groupby("manager")["period_date"].rank(method="dense").astype(int)

    # Known issue: prior-period turnover uses the latest version of that prior
    # period available in the full dataset. If a prior period is amended after a
    # decision date, this can leak a small amount of post-asof information into
    # the turnover screen. Fix later by computing turnover as-of each rebalance.
    latest_period = chars.groupby(["manager", "period_date"], sort=False, as_index=False).tail(1).copy()
    latest_period["prev_bw"] = latest_period.groupby("manager", sort=False)["bw"].shift(1)
    previous_table = latest_period[["manager", "period_date", "prev_bw"]]
    chars["_row_order"] = np.arange(len(chars))
    chars = (
        chars.merge(previous_table, on=["manager", "period_date"], how="left", sort=False, validate="many_to_one")
        .sort_values("_row_order")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )
    chars["prev_bw"] = pd.Series(
        [value if isinstance(value, pd.Series) else None for value in chars["prev_bw"]],
        index=chars.index,
        dtype=object,
    )

    def overlap_turnover(current_bw, previous_bw) -> float:
        if not isinstance(previous_bw, pd.Series):
            return np.nan
        all_names = current_bw.index.union(previous_bw.index)
        current = current_bw.reindex(all_names).fillna(0.0)
        previous = previous_bw.reindex(all_names).fillna(0.0)
        return float(1.0 - np.minimum(current, previous).sum())

    chars["turnover"] = [
        overlap_turnover(current, previous)
        for current, previous in zip(chars["bw"], chars["prev_bw"])
    ]
    if progress is not None:
        progress(
            f"characteristics turnover complete managers={chars['manager'].nunique():,} "
            f"({time.perf_counter() - started:.1f}s)"
        )
    if progress is not None:
        progress(f"characteristics complete rows={len(chars):,} ({time.perf_counter() - started:.1f}s)")
    return chars


def _cap_weights(w: pd.Series, cap: float) -> pd.Series:
    """Enforce a per-name cap with iterative redistribution where feasible."""
    w = w[w > 0].astype(float)
    if w.empty:
        return w
    w = w / w.sum()
    if cap is None or cap <= 0 or cap >= 1:
        return w
    if len(w) * cap <= 1.0 + 1e-12:
        # Infeasible hard cap: not enough names to sum to one under this cap.
        # Equal weight is the least concentrated fallback; callers can inspect
        # effective_names vs max_name_weight in the rebalance audit.
        return pd.Series(1.0 / len(w), index=w.index)
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        free = ~over
        excess = float(w[over].sum() - cap * over.sum())
        w[over] = cap
        w[free] = w[free] + w[free] / w[free].sum() * excess
    return w


def _security_groups_for(tickers, security_groups=None) -> pd.Series:
    idx = pd.Index(tickers).astype(str).str.upper()
    if security_groups is None:
        return pd.Series(idx, index=idx)
    groups = pd.Series(security_groups, dtype="string")
    groups.index = groups.index.astype(str).str.upper()
    out = pd.Series(idx, index=idx, dtype="string")
    mapped = groups.reindex(idx).dropna().astype(str).str.upper()
    out.loc[mapped.index] = mapped
    return out.astype(str)


def _cap_weights_with_groups(
    w: pd.Series,
    max_name_weight: float,
    max_issuer_weight: float | None,
    security_groups=None,
) -> pd.Series:
    w = _cap_weights(w, max_name_weight)
    if w.empty or max_issuer_weight is None or max_issuer_weight <= 0 or max_issuer_weight >= 1:
        return w
    groups = _security_groups_for(w.index, security_groups).reindex(w.index)
    if groups.nunique() * max_issuer_weight <= 1.0 + 1e-12:
        return w
    raw = w.copy()
    for _ in range(100):
        w = _cap_weights(w, max_name_weight)
        group_sum = w.groupby(groups).sum()
        over = group_sum[group_sum > max_issuer_weight + 1e-12]
        if over.empty:
            break
        for group, total in over.items():
            names = groups[groups == group].index
            w.loc[names] *= max_issuer_weight / total
        deficit = 1.0 - float(w.sum())
        if deficit <= 1e-12:
            break
        group_sum = w.groupby(groups).sum()
        name_room = (max_name_weight - w).clip(lower=0.0)
        group_room = (max_issuer_weight - groups.map(group_sum)).clip(lower=0.0)
        room = pd.concat([name_room.rename("name"), group_room.rename("group")], axis=1).min(axis=1)
        eligible = room[room > 1e-12]
        if eligible.empty:
            break
        base = raw.reindex(eligible.index).fillna(0.0).clip(lower=0.0)
        if base.sum() <= 0:
            base = eligible
        add = base / base.sum() * deficit
        add = pd.concat([add.rename("add"), eligible.rename("room")], axis=1).min(axis=1)
        w.loc[add.index] += add
        if abs(1.0 - float(w.sum())) <= 1e-10:
            break
    return w / w.sum() if w.sum() > 0 else w


def _book_value_pctl(w: pd.Series, vscores: pd.Series | None) -> float:
    if vscores is None:
        return np.nan
    common = w.index.intersection(vscores.dropna().index)
    if len(common) == 0:
        return np.nan
    return float((w[common] * vscores[common].rank(pct=True)).sum() / w[common].sum())


# --------------------------------------------------------------------------- #
def filter_universe_versions(latest_versions: pd.DataFrame, cfg: UniverseConfig, value_scores=None) -> pd.DataFrame:
    latest = latest_versions
    if latest.empty:
        return latest
    latest = latest.set_index("manager", drop=False)
    keep = pd.Series(True, index=latest.index)
    keep &= latest["hist_q"] >= cfg.min_history_quarters
    if cfg.use_size_band:
        keep &= latest["aum"].between(cfg.min_aum, cfg.max_aum)
    if cfg.use_concentration:
        keep &= (latest["top10_weight"] >= cfg.min_top_n_weight) & (latest["n_holdings"] <= cfg.max_holdings)
    if cfg.use_hedge_filter:
        put_weight = latest["filing_put_weight"] if "filing_put_weight" in latest else latest["put_weight"]
        keep &= put_weight.notna() & put_weight.le(cfg.hedge_put_max_weight)
    if cfg.use_active_share and latest["active_share"].notna().any():
        keep &= latest["active_share"].ge(cfg.min_active_share).reindex(keep.index).fillna(False)
    if cfg.use_low_turnover:
        t = latest["turnover"].dropna()
        if len(t) >= 3:
            keep &= latest["turnover"].le(t.quantile(cfg.turnover_quantile)).reindex(keep.index).fillna(False)
    if cfg.use_value_tilt and value_scores is not None:
        candidates = latest.loc[keep]
        vp = {m: _book_value_pctl(r["bw"], value_scores.loc[r["period_date"]]
                                  if r["period_date"] in value_scores.index else None)
              for m, r in candidates.iterrows()}
        keep &= pd.Series(vp).ge(cfg.value_tilt_min_pctl).reindex(keep.index, fill_value=False).astype(bool)
    return latest.loc[keep.values].reset_index(drop=True)


def select_universe_versions(chars, asof, cfg: UniverseConfig, value_scores=None) -> pd.DataFrame:
    latest, _ = _filter_fresh_versions(_visible_manager_versions(chars, asof), asof, cfg)
    return filter_universe_versions(latest, cfg, value_scores)


def select_universe(chars, asof, cfg: UniverseConfig, value_scores=None) -> list[str]:
    selected = select_universe_versions(chars, asof, cfg, value_scores)
    if selected.empty:
        return []
    return selected["manager"].tolist()


def _apply_manager_type_filter(
    selected_versions: pd.DataFrame,
    month,
    cfg: BacktestConfig,
    manager_classification: pd.DataFrame | None = None,
    manager_overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    mode = getattr(cfg, "manager_filter_mode", "all")
    if mode == "all":
        return selected_versions
    return filter_selected_versions(
        selected_versions,
        month,
        mode,
        manager_classification,
        manager_overrides,
    )


def _manager_filter_diagnostics(selected_versions: pd.DataFrame, cfg: BacktestConfig) -> dict:
    mode = getattr(cfg, "manager_filter_mode", "all")
    return {
        "manager_filter_mode": mode,
        "manager_filter_before": int(selected_versions.attrs.get("manager_filter_before", len(selected_versions))),
        "manager_filter_after": int(selected_versions.attrs.get("manager_filter_after", len(selected_versions))),
        "manager_filter_dropped": int(selected_versions.attrs.get("manager_filter_dropped", 0)),
        "manager_filter_missing_classification": int(selected_versions.attrs.get("manager_filter_missing_classification", 0)),
        "manager_filter_missing_classification_frac": float(
            selected_versions.attrs.get("manager_filter_missing_classification_frac", 0.0)
        ),
        "manager_filter_dirty_dropped": int(selected_versions.attrs.get("manager_filter_dirty_dropped", 0)),
        "manager_filter_non_dedicated_dropped": int(selected_versions.attrs.get("manager_filter_non_dedicated_dropped", 0)),
    }


def _manager_candidates_audit(
    visible_versions: pd.DataFrame,
    fresh_versions: pd.DataFrame,
    universe_selected: pd.DataFrame,
    final_selected: pd.DataFrame,
    month,
    cfg: BacktestConfig,
    value_scores=None,
    manager_classification: pd.DataFrame | None = None,
    manager_overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Describe every visible manager and the exact filters active at a rebalance."""
    if visible_versions is None or visible_versions.empty:
        return pd.DataFrame()
    audit = visible_versions.copy()
    audit["manager"] = audit["manager"].astype(str).str.zfill(10)
    asof = pd.Timestamp(month).normalize()
    u = cfg.universe
    audit = _add_freshness_audit_columns(audit, asof, u)
    audit["rebalance_month"] = asof.date().isoformat()
    audit["min_history_quarters_cutoff"] = int(u.min_history_quarters)
    audit["pass_min_history"] = audit["hist_q"].ge(u.min_history_quarters)
    audit["size_band_active"] = bool(u.use_size_band)
    audit["min_aum_cutoff"] = float(u.min_aum)
    audit["max_aum_cutoff"] = float(u.max_aum)
    audit["pass_size_band"] = (not u.use_size_band) | audit["aum"].between(u.min_aum, u.max_aum)
    audit["concentration_active"] = bool(u.use_concentration)
    audit["min_top_n_weight_cutoff"] = float(u.min_top_n_weight)
    audit["max_holdings_cutoff"] = int(u.max_holdings)
    concentration_pass = audit["top10_weight"].ge(u.min_top_n_weight) & audit["n_holdings"].le(u.max_holdings)
    audit["pass_concentration"] = (not u.use_concentration) | concentration_pass
    audit["hedge_filter_active"] = bool(u.use_hedge_filter)
    audit["hedge_put_max_weight_cutoff"] = float(u.hedge_put_max_weight)
    audit["investable_put_weight"] = audit["put_weight"]
    if "filing_put_weight" in audit:
        audit["hedge_put_weight_source"] = "raw_sec_filing_before_mapping"
        hedge_put_weight = audit["filing_put_weight"]
    else:
        audit["filing_put_weight"] = pd.NA
        audit["hedge_put_weight_source"] = "available_book_fallback"
        hedge_put_weight = audit["put_weight"]
    audit["hedge_put_weight"] = hedge_put_weight
    audit["pass_hedge_filter"] = (
        not u.use_hedge_filter
    ) | (hedge_put_weight.notna() & hedge_put_weight.le(u.hedge_put_max_weight))
    audit["active_share_filter_active"] = bool(u.use_active_share and audit["active_share"].notna().any())
    audit["min_active_share_cutoff"] = float(u.min_active_share)
    audit["pass_active_share"] = (~audit["active_share_filter_active"]) | audit["active_share"].ge(u.min_active_share)
    fresh_turnover = pd.to_numeric(fresh_versions.get("turnover", pd.Series(dtype=float)), errors="coerce").dropna()
    turnover_cutoff = float(fresh_turnover.quantile(u.turnover_quantile)) if len(fresh_turnover) >= 3 else np.nan
    audit["low_turnover_filter_active"] = bool(u.use_low_turnover and pd.notna(turnover_cutoff))
    audit["turnover_quantile_cutoff"] = float(u.turnover_quantile)
    audit["turnover_value_cutoff"] = turnover_cutoff
    audit["pass_low_turnover"] = (~audit["low_turnover_filter_active"]) | audit["turnover"].le(turnover_cutoff)
    value_tilt_active = bool(u.use_value_tilt and value_scores is not None)
    audit["value_tilt_filter_active"] = value_tilt_active
    audit["value_tilt_min_pctl_cutoff"] = float(u.value_tilt_min_pctl)
    if value_tilt_active:
        audit["book_value_pctl"] = [
            _book_value_pctl(
                row.bw,
                value_scores.loc[row.period_date] if row.period_date in value_scores.index else None,
            )
            for row in audit.itertuples(index=False)
        ]
    else:
        audit["book_value_pctl"] = np.nan
    audit["pass_value_tilt"] = (~audit["value_tilt_filter_active"]) | audit["book_value_pctl"].ge(u.value_tilt_min_pctl)

    mode = cfg.manager_filter_mode
    audit["manager_filter_mode"] = mode
    if mode == "all":
        audit["pass_manager_type"] = True
    elif manager_classification is None or manager_classification.empty:
        audit["manager_style"] = pd.NA
        audit["dirty_flag"] = True
        audit["dirty_reason"] = "missing_classification"
        audit["pass_manager_type"] = False
    else:
        classification = manager_classification[
            pd.to_datetime(manager_classification["asof_month"]).eq(asof)
        ].copy()
        classification["manager"] = classification["manager"].astype(str).str.zfill(10)
        classification = apply_manager_overrides(classification, manager_overrides)
        meta_cols = ["manager", "manager_style", "dirty_flag", "dirty_reason", "classification_source"]
        audit = audit.merge(classification[meta_cols], on="manager", how="left", suffixes=("", "_class"))
        dirty = audit["dirty_flag"].map(lambda value: bool(value) if pd.notna(value) else True)
        if mode == "exclude_dirty":
            audit["pass_manager_type"] = ~dirty
        elif mode == "dedicated_like":
            audit["pass_manager_type"] = (~dirty) & audit["manager_style"].eq("dedicated")
        else:
            audit["pass_manager_type"] = True

    universe_ids = set(universe_selected["manager"].astype(str).str.zfill(10))
    final_ids = set(final_selected["manager"].astype(str).str.zfill(10))
    audit["pass_universe_filters"] = audit["manager"].isin(universe_ids)
    audit["pass_final"] = audit["manager"].isin(final_ids)
    columns = [
        "rebalance_month", "manager", "period_date", "filing_date", "accession_number",
        "aum", "n_holdings", "top10_weight", "put_weight", "active_share", "turnover", "hist_q",
        "filing_date_cutoff", "period_date_cutoff", "pass_filing_fresh", "pass_period_fresh", "pass_freshness",
        "min_history_quarters_cutoff", "pass_min_history",
        "size_band_active", "min_aum_cutoff", "max_aum_cutoff", "pass_size_band",
        "concentration_active", "min_top_n_weight_cutoff", "max_holdings_cutoff", "pass_concentration",
        "hedge_filter_active", "hedge_put_max_weight_cutoff", "filing_put_weight",
        "investable_put_weight", "hedge_put_weight", "hedge_put_weight_source", "pass_hedge_filter",
        "active_share_filter_active", "min_active_share_cutoff", "pass_active_share",
        "low_turnover_filter_active", "turnover_quantile_cutoff", "turnover_value_cutoff", "pass_low_turnover",
        "value_tilt_filter_active", "book_value_pctl", "value_tilt_min_pctl_cutoff", "pass_value_tilt",
        "manager_filter_mode", "manager_style", "dirty_flag", "dirty_reason", "classification_source",
        "pass_manager_type", "pass_universe_filters", "pass_final",
    ]
    for column in columns:
        if column not in audit:
            audit[column] = pd.NA
    return audit[columns]


# --------------------------------------------------------------------------- #
def _idea_scores(
    cur: pd.Series,
    prev: pd.Series | None,
    cfg: PortfolioConfig,
    active_benchmark_weights: pd.Series | None = None,
    idiosyncratic_vol: pd.Series | None = None,
    eligible_tickers=None,
    active_benchmark_source: str = "visible_13f_aggregate",
) -> pd.Series:
    if eligible_tickers is not None:
        eligible = pd.Index(eligible_tickers).astype(str).str.upper()
        cur = cur[cur.index.astype(str).str.upper().isin(eligible)]
        if prev is not None:
            prev = prev[prev.index.astype(str).str.upper().isin(eligible)]
    if cfg.idea_signal == "level":
        s = cur
    elif cfg.idea_signal == "change":
        if prev is None:
            s = cur
        else:
            alln = cur.index.union(prev.index)
            s = (cur.reindex(alln).fillna(0.0) - prev.reindex(alln).fillna(0.0)).clip(lower=0)
    elif cfg.idea_signal == "initiation":
        new = cur.index.difference(prev.index) if prev is not None else cur.index
        s = cur.reindex(new).fillna(0.0)
    elif _needs_active_benchmark_weights(cfg.idea_signal):
        bench = _require_active_benchmark_weights(cfg, active_benchmark_weights)
        if int((cur > 0).sum()) < cfg.min_active_weight_holdings:
            return pd.Series(dtype=float)
        if _uses_manager_held_mcap_benchmark(active_benchmark_source):
            # Market-cap data is never zero-filled. A manager's benchmark is
            # the market-cap-weighted portfolio of the names it actually held.
            cur, held_bench = _manager_held_mcap_weights(cur, bench)
            if len(cur) < cfg.min_active_weight_holdings:
                return pd.Series(dtype=float)
            if held_bench.empty:
                return pd.Series(dtype=float)
            active_cur = (cur - held_bench).clip(lower=0)
        else:
            active_cur = (cur - bench.reindex(cur.index).fillna(0.0)).clip(lower=0)
        if cfg.idea_signal == "active_weight":
            s = active_cur
        elif cfg.idea_signal == "active_weight_change":
            if prev is None:
                s = active_cur
            else:
                alln = cur.index.union(prev.index)
                inc = (cur.reindex(alln).fillna(0.0) - prev.reindex(alln).fillna(0.0)).clip(lower=0)
                s = active_cur[(inc.reindex(cur.index).fillna(0.0) > 0) & (active_cur > 0)]
        elif cfg.idea_signal == "active_weight_initiation":
            new = cur.index.difference(prev.index) if prev is not None else cur.index
            s = active_cur.reindex(new).fillna(0.0)
        elif _needs_idiosyncratic_vol(cfg.idea_signal):
            vol = _require_idiosyncratic_vol(cfg, idiosyncratic_vol)
            cps_score = (active_cur * vol.reindex(active_cur.index)).replace([np.inf, -np.inf], np.nan).dropna()
            cps_score = cps_score[cps_score > 0]
            if cfg.idea_signal == "cps_ir":
                s = cps_score
            elif cfg.idea_signal == "cps_ir_change":
                if prev is None:
                    s = cps_score
                else:
                    alln = cur.index.union(prev.index)
                    inc = (cur.reindex(alln).fillna(0.0) - prev.reindex(alln).fillna(0.0)).clip(lower=0)
                    s = cps_score[(inc.reindex(cps_score.index).fillna(0.0) > 0) & (cps_score > 0)]
            elif cfg.idea_signal == "cps_ir_initiation":
                new = cur.index.difference(prev.index) if prev is not None else cur.index
                s = cps_score.reindex(new).dropna()
            else:
                raise ValueError(cfg.idea_signal)
    else:
        raise ValueError(cfg.idea_signal)
    return s[s > 0].sort_values(ascending=False).head(cfg.top_n_ideas)


def _manager_idea_weights(cur: pd.Series, picks: pd.Series) -> pd.Series:
    """Allocate one manager budget across selected ideas by reported book weight."""
    if picks.empty:
        return pd.Series(dtype=float)
    selected = cur.reindex(picks.index).fillna(0.0).clip(lower=0.0)
    total = float(selected.sum())
    if total > 0:
        return selected / total
    return pd.Series(1.0 / len(picks), index=picks.index, dtype=float)


def _idea_component_rows(
    manager_version,
    cur: pd.Series,
    picks: pd.Series,
    cfg: PortfolioConfig,
    active_benchmark_weights: pd.Series | None,
    idiosyncratic_vol: pd.Series | None,
    eligible_tickers=None,
    active_benchmark_source: str = "visible_13f_aggregate",
) -> list[dict]:
    """Return the score components for the exact manager ideas that were selected."""
    if picks.empty:
        return []
    current = cur.copy()
    if eligible_tickers is not None:
        eligible = pd.Index(eligible_tickers).astype(str).str.upper()
        current = current[current.index.astype(str).str.upper().isin(eligible)]
    benchmark = pd.Series(dtype=float)
    active = pd.Series(dtype=float)
    if _needs_active_benchmark_weights(cfg.idea_signal):
        benchmark_input = _require_active_benchmark_weights(cfg, active_benchmark_weights)
        if _uses_manager_held_mcap_benchmark(active_benchmark_source):
            current, benchmark = _manager_held_mcap_weights(current, benchmark_input)
        else:
            benchmark = benchmark_input.reindex(current.index).fillna(0.0)
        active = (current - benchmark.reindex(current.index).fillna(0.0)).clip(lower=0.0)
    rows = []
    manager = str(getattr(manager_version, "manager", ""))
    period_date = getattr(manager_version, "period_date", pd.NaT)
    filing_date = getattr(manager_version, "filing_date", pd.NaT)
    accession_number = getattr(manager_version, "accession_number", pd.NA)
    manager_idea_weights = _manager_idea_weights(current, picks)
    for rank, (ticker, signal_score) in enumerate(picks.items(), start=1):
        active_weight = float(active.get(ticker, np.nan)) if not active.empty else np.nan
        idio = float(idiosyncratic_vol.get(ticker, np.nan)) if idiosyncratic_vol is not None else np.nan
        cps = active_weight * idio if pd.notna(active_weight) and pd.notna(idio) else np.nan
        rows.append({
            "manager": manager,
            "period_date": period_date,
            "filing_date": filing_date,
            "accession_number": accession_number,
            "ticker": str(ticker),
            "manager_book_weight": float(current.get(ticker, np.nan)),
            "benchmark_weight": float(benchmark.get(ticker, np.nan)) if not benchmark.empty else np.nan,
            "active_weight": active_weight,
            "idio_vol": idio,
            "cps_score": cps,
            "signal_score": float(signal_score),
            "manager_idea_weight": float(manager_idea_weights.get(ticker, np.nan)),
            "manager_idea_rank": int(rank),
        })
    return rows


def target_weights_from_versions(
    latest_versions: pd.DataFrame,
    cfg: PortfolioConfig,
    active_benchmark_weights: pd.Series | None = None,
    idiosyncratic_vol: pd.Series | None = None,
    eligible_tickers=None,
    return_diagnostics: bool = False,
    active_benchmark_source: str = "visible_13f_aggregate",
) -> pd.Series:
    diagnostics = {
        "selected_managers": int(len(latest_versions)),
        "active_eligible_managers": 0,
        "zero_contributor_managers": 0,
        "raw_idea_rows": 0,
        "raw_idea_names": 0,
        "consensus_idea_names": 0,
        "target_names_before_caps": 0,
        "idio_vol_covered_names": 0,
        "market_cap_covered_names": 0,
        "market_cap_eligible_managers": 0,
        "market_cap_mean_book_coverage": np.nan,
        "max_distinct_ideas_upper_bound": int(len(latest_versions) * max(int(cfg.top_n_ideas), 0)),
        "_idea_audit": pd.DataFrame(),
        "_target_weights_pre_cap": pd.Series(dtype=float),
        "_source_managers": {},
    }
    if latest_versions.empty:
        empty = pd.Series(dtype=float)
        return (empty, diagnostics) if return_diagnostics else empty
    active_benchmark_weights = _require_active_benchmark_weights(cfg, active_benchmark_weights)
    idiosyncratic_vol = _require_idiosyncratic_vol(cfg, idiosyncratic_vol)
    if idiosyncratic_vol is not None:
        diagnostics["idio_vol_covered_names"] = int(len(idiosyncratic_vol))
    if _uses_manager_held_mcap_benchmark(active_benchmark_source) and active_benchmark_weights is not None:
        diagnostics["market_cap_covered_names"] = int(len(active_benchmark_weights))
    score: dict[str, float] = {}
    max_score: dict[str, float] = {}
    count: dict[str, int] = {}
    source_managers: dict[str, list[str]] = {}
    idea_rows: list[dict] = []
    market_cap_book_coverages: list[float] = []
    for cur_row in latest_versions.itertuples(index=False):
        cur = getattr(cur_row, "bw")
        prev = getattr(cur_row, "prev_bw", None)
        cur_diag = cur
        if eligible_tickers is not None:
            eligible = pd.Index(eligible_tickers).astype(str).str.upper()
            cur_diag = cur[cur.index.astype(str).str.upper().isin(eligible)]
        if _needs_active_benchmark_weights(cfg.idea_signal) and int((cur_diag > 0).sum()) >= cfg.min_active_weight_holdings:
            diagnostics["active_eligible_managers"] += 1
        if _uses_manager_held_mcap_benchmark(active_benchmark_source) and active_benchmark_weights is not None:
            covered_names = cur_diag.index.intersection(active_benchmark_weights.index)
            market_cap_book_coverages.append(float(cur_diag.reindex(covered_names).sum()))
            if len(covered_names) >= cfg.min_active_weight_holdings:
                diagnostics["market_cap_eligible_managers"] += 1
        picks = _idea_scores(
            cur,
            prev,
            cfg,
            active_benchmark_weights,
            idiosyncratic_vol,
            eligible_tickers,
            active_benchmark_source,
        )
        idea_rows.extend(_idea_component_rows(
            cur_row,
            cur,
            picks,
            cfg,
            active_benchmark_weights,
            idiosyncratic_vol,
            eligible_tickers,
            active_benchmark_source,
        ))
        if picks.empty:
            diagnostics["zero_contributor_managers"] += 1
        diagnostics["raw_idea_rows"] += int(len(picks))
        manager_idea_weights = _manager_idea_weights(cur, picks)
        for tkr, wt in picks.items():
            aggregation = cfg.idea_aggregation or ("score" if cfg.consensus_weight else "manager_count")
            if aggregation not in {"manager_equal", "score", "manager_count", "equal_name"}:
                raise ValueError(f"unknown idea_aggregation={aggregation!r}")
            if aggregation == "manager_equal":
                contribution = float(manager_idea_weights.loc[tkr])
            elif aggregation == "score":
                contribution = float(wt)
            else:
                contribution = 1.0
            score[tkr] = score.get(tkr, 0.0) + contribution
            max_score[tkr] = max(max_score.get(tkr, float("-inf")), float(wt))
            count[tkr] = count.get(tkr, 0) + 1
            source_managers.setdefault(str(tkr), []).append(str(getattr(cur_row, "manager", "")))
    diagnostics["raw_idea_names"] = int(len(score))
    if market_cap_book_coverages:
        diagnostics["market_cap_mean_book_coverage"] = float(np.mean(market_cap_book_coverages))
    idea_audit = pd.DataFrame(idea_rows)
    if not idea_audit.empty:
        idea_audit["contributor_count"] = idea_audit["ticker"].map(count).fillna(0).astype(int)
        idea_audit["aggregate_score"] = idea_audit["ticker"].map(score)
        idea_audit["selected_after_consensus_and_name_limit"] = False
    diagnostics["_idea_audit"] = idea_audit
    diagnostics["_source_managers"] = {
        ticker: sorted(set(managers)) for ticker, managers in source_managers.items()
    }
    s = pd.Series(score)
    if s.empty:
        return (s, diagnostics) if return_diagnostics else s
    if cfg.min_consensus_funds > 1:
        keep = pd.Series(count).ge(cfg.min_consensus_funds)
        s = s[keep.reindex(s.index).fillna(False)]
    diagnostics["consensus_idea_names"] = int(len(s))
    if s.empty:
        return (s, diagnostics) if return_diagnostics else s
    aggregation = cfg.idea_aggregation or ("score" if cfg.consensus_weight else "manager_count")
    if aggregation == "equal_name":
        # Duplicated manager picks satisfy an optional consensus gate but do
        # not receive extra portfolio weight. Max conviction is used only to
        # choose names when max_portfolio_names binds.
        s = pd.Series(max_score, dtype=float).reindex(s.index)
    if cfg.max_portfolio_names is not None and cfg.max_portfolio_names > 0:
        s = s.sort_values(ascending=False).head(cfg.max_portfolio_names)
    diagnostics["target_names_before_caps"] = int(len(s))
    if aggregation == "equal_name":
        s = pd.Series(1.0, index=s.index, dtype=float)
    pre_cap = s / s.sum() if s.sum() > 0 else s
    target = _cap_weights(s, cfg.max_name_weight)
    if not idea_audit.empty:
        idea_audit["selected_after_consensus_and_name_limit"] = idea_audit["ticker"].isin(target.index)
    diagnostics["_idea_audit"] = idea_audit
    diagnostics["_target_weights_pre_cap"] = pre_cap
    return (target, diagnostics) if return_diagnostics else target


def target_weights(chars, managers, asof, cfg: PortfolioConfig) -> pd.Series:
    latest = _visible_manager_versions(chars, asof, managers)
    return target_weights_from_versions(latest, cfg)


# --------------------------------------------------------------------------- #
def _rebalance_months(holdings: pd.DataFrame, prices: pd.DataFrame) -> list[pd.Timestamp]:
    months = prices.index.sort_values()
    fil = holdings["filing_date"].drop_duplicates().sort_values()
    return sorted({months[months >= f][0] for f in fil if (months >= f).any()})


def _apply_rebalance_target(cur: pd.Series,
                            last_in_tgt: dict[str, pd.Timestamp],
                            tgt: pd.Series,
                            cfg: BacktestConfig,
                            month: pd.Timestamp,
                            security_groups=None,
                            return_pre_cap: bool = False):
    tgt = tgt / tgt.sum()
    for n in tgt.index:
        last_in_tgt[n] = month

    eff = {n: float(w) for n, w in tgt.items()}
    carried: list[str] = []
    H = cfg.portfolio.holding_horizon_q
    cur_q = month.to_period("Q")
    for n in list(last_in_tgt):
        if n in eff:
            continue
        q_gap = (cur_q - last_in_tgt[n].to_period("Q")).n
        if H > 0 and q_gap <= H and n in cur.index:
            eff[n] = float(cur[n])
            carried.append(n)
        elif H <= 0 or q_gap > H:
            last_in_tgt.pop(n, None)

    pre_cap = pd.Series(eff, dtype=float)
    pre_cap = pre_cap / pre_cap.sum() if pre_cap.sum() > 0 else pre_cap
    eff_s = _cap_weights_with_groups(
        pre_cap,
        cfg.portfolio.max_name_weight,
        cfg.portfolio.max_issuer_weight,
        security_groups,
    )
    alln = eff_s.index.union(cur.index)
    traded = 0.5 * (eff_s.reindex(alln).fillna(0) - cur.reindex(alln).fillna(0)).abs().sum()
    if return_pre_cap:
        return eff_s, float(traded), carried, pre_cap
    return eff_s, float(traded), carried


def _portfolio_summary_stats(weights: pd.Series) -> dict[str, float]:
    w = weights[weights > 0].astype(float)
    if w.empty:
        return {
            "max_weight": 0.0,
            "top5_weight": 0.0,
            "top10_weight": 0.0,
            "hhi": 0.0,
            "effective_number": 0.0,
        }
    w = w / w.sum()
    hhi = float((w ** 2).sum())
    return {
        "max_weight": float(w.max()),
        "top5_weight": float(w.sort_values(ascending=False).head(5).sum()),
        "top10_weight": float(w.sort_values(ascending=False).head(10).sum()),
        "hhi": hhi,
        "effective_number": float(1.0 / hhi) if hhi > 0 else 0.0,
    }


def _issuer_exposure_summary(weights: pd.Series, security_groups=None) -> dict[str, float | str]:
    w = weights[weights > 0].astype(float)
    if w.empty:
        return {
            "issuer_groups": 0,
            "max_issuer_weight": 0.0,
            "top_issuer_exposures": "",
            "multi_class_exposures": "",
        }
    groups = _security_groups_for(w.index, security_groups).reindex(w.index)
    issuer_w = w.groupby(groups).sum().sort_values(ascending=False)
    multi = []
    for group, names in groups.groupby(groups):
        tickers = list(names.index)
        held = [ticker for ticker in tickers if ticker in w.index and w.loc[ticker] > 0]
        if len(held) > 1:
            parts = ", ".join(f"{ticker}:{w.loc[ticker]:.2%}" for ticker in sorted(held))
            multi.append(f"{group}({parts}; total:{issuer_w.loc[group]:.2%})")
    return {
        "issuer_groups": int(len(issuer_w)),
        "max_issuer_weight": float(issuer_w.max()) if not issuer_w.empty else 0.0,
        "top_issuer_exposures": "; ".join(f"{k}:{v:.2%}" for k, v in issuer_w.head(12).items()),
        "multi_class_exposures": "; ".join(multi[:12]),
    }


def _constraint_feasibility(weights: pd.Series, cfg: BacktestConfig, security_groups=None) -> dict[str, bool]:
    w = weights[weights > 0]
    if w.empty:
        return {"name_cap_feasible": True, "issuer_cap_feasible": True}
    groups = _security_groups_for(w.index, security_groups).reindex(w.index)
    max_name = cfg.portfolio.max_name_weight
    max_issuer = cfg.portfolio.max_issuer_weight
    name_ok = max_name is None or max_name <= 0 or max_name >= 1 or len(w) * max_name >= 1.0 - 1e-12
    issuer_ok = (
        max_issuer is None
        or max_issuer <= 0
        or max_issuer >= 1
        or groups.nunique() * max_issuer >= 1.0 - 1e-12
    )
    return {"name_cap_feasible": bool(name_ok), "issuer_cap_feasible": bool(issuer_ok)}


def _trade_summary_stats(before: pd.Series, after: pd.Series) -> dict[str, int]:
    alln = before.index.union(after.index)
    b = before.reindex(alln).fillna(0.0)
    a = after.reindex(alln).fillna(0.0)
    changed = (a - b).abs() > 1e-12
    return {
        "traded_names": int(changed.sum()),
        "buy_names": int(((b <= 1e-12) & (a > 1e-12)).sum()),
        "sell_names": int(((b > 1e-12) & (a <= 1e-12)).sum()),
        "increased_names": int(((a - b) > 1e-12).sum()),
        "decreased_names": int(((b - a) > 1e-12).sum()),
    }


def _minimum_target_failure(tgt: pd.Series, cfg: BacktestConfig) -> str:
    min_names = int(cfg.portfolio.min_portfolio_names or 0)
    if tgt.sum() <= 0:
        return "no positive target weights"
    if min_names > 0 and len(tgt) < min_names:
        return f"target_names_below_min_portfolio_names({len(tgt)}<{min_names})"
    return ""


def _invalid_cash_rebalance(
    cur: pd.Series,
    cfg: BacktestConfig,
    security_groups=None,
) -> tuple[pd.Series, float, dict[str, float], dict[str, float | str], dict[str, bool], dict[str, int]]:
    eff = pd.Series(dtype=float)
    alln = cur.index.union(eff.index)
    traded = float(0.5 * (eff.reindex(alln).fillna(0.0) - cur.reindex(alln).fillna(0.0)).abs().sum())
    return (
        eff,
        traded,
        _portfolio_summary_stats(eff),
        _issuer_exposure_summary(eff, security_groups),
        _constraint_feasibility(eff, cfg, security_groups),
        _trade_summary_stats(cur, eff),
    )


def _apply_monthly_returns(cur: pd.Series,
                           prices: pd.DataFrame,
                           month: pd.Timestamp,
                           cfg: BacktestConfig) -> tuple[pd.Series, float]:
    if not len(cur):
        return cur, 0.0
    r = prices.loc[month, cur.index]
    missing = r[r.isna()].index
    if len(missing):
        policy = cfg.missing_price_policy
        if policy == "raise":
            sample = ", ".join(map(str, list(missing)[:10]))
            raise ValueError(
                f"Missing returns for {len(missing)} held names on {month.date()}: {sample}"
            )
        if policy == "zero":
            r = r.fillna(0.0)
        elif policy == "exit":
            keep = r.index.difference(missing)
            if len(keep) == 0:
                return pd.Series(dtype=float), 0.0
            cur = cur[keep]
            r = r[keep]
        else:
            raise ValueError(f"Unknown missing_price_policy={policy!r}")
    net = float((cur * r).sum())
    grown = cur * (1.0 + r)
    return (grown / grown.sum() if grown.sum() > 0 else cur), net


def rebalance_trace(holdings, prices, cfg: BacktestConfig,
                    value_scores=None, benchmark_weights=None, chars=None,
                    visible_versions_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
                    security_groups=None,
                    active_benchmark_weights_by_month: dict[pd.Timestamp, pd.Series] | None = None,
                    manager_classification: pd.DataFrame | None = None,
                    manager_overrides: pd.DataFrame | None = None,
                    idiosyncratic_vol_by_month: dict[pd.Timestamp, pd.Series] | None = None) -> dict[str, pd.DataFrame]:
    """Return auditable rebalance summary, holdings, and manager-selection tables."""
    if chars is None:
        chars = manager_characteristics(holdings, benchmark_weights)
    months = prices.index.sort_values()
    rebal_months = set(_rebalance_months(holdings, prices))

    cur = pd.Series(dtype=float)
    last_in_tgt: dict[str, pd.Timestamp] = {}
    last_sources: dict[str, list[str]] = {}
    summary_rows: list[dict] = []
    holding_rows: list[dict] = []
    holding_source_rows: list[dict] = []
    manager_rows: list[dict] = []
    manager_candidate_rows: list[dict] = []
    idea_rows: list[dict] = []

    for m in months:
        cur, _ = _apply_monthly_returns(cur, prices, m, cfg)
        if m not in rebal_months:
            continue

        visible_versions = visible_versions_cache.get(pd.Timestamp(m)) if visible_versions_cache is not None else None
        if visible_versions is None:
            visible_versions = _visible_manager_versions(chars, m)
        latest_versions, stale_diag = _filter_fresh_versions(visible_versions, m, cfg.universe)
        active_benchmark_weights = _active_benchmark_for_month(
            cfg,
            m,
            latest_versions,
            active_benchmark_weights_by_month,
        )
        idiosyncratic_vol = _idiosyncratic_vol_for_month(cfg, m, idiosyncratic_vol_by_month)
        universe_selected = filter_universe_versions(latest_versions, cfg.universe, value_scores)
        selected_versions = universe_selected
        selected_versions = _apply_manager_type_filter(
            selected_versions,
            m,
            cfg,
            manager_classification,
            manager_overrides,
        )
        candidate_audit = _manager_candidates_audit(
            visible_versions,
            latest_versions,
            universe_selected,
            selected_versions,
            m,
            cfg,
            value_scores,
            manager_classification,
            manager_overrides,
        )
        if not candidate_audit.empty:
            manager_candidate_rows.extend(candidate_audit.to_dict(orient="records"))
        manager_filter_diag = _manager_filter_diagnostics(selected_versions, cfg)
        selected_managers = selected_versions["manager"].tolist() if not selected_versions.empty else []
        for row in selected_versions.itertuples(index=False):
            manager_rows.append({
                "rebalance_month": m.date().isoformat(),
                "manager": getattr(row, "manager"),
                "manager_name": getattr(row, "manager_name", ""),
                "manager_style": getattr(row, "manager_style", ""),
                "dirty_flag": bool(getattr(row, "dirty_flag", False)),
                "dirty_reason": getattr(row, "dirty_reason", ""),
                "classification_source": getattr(row, "classification_source", ""),
                "factor_r2": getattr(row, "factor_r2", np.nan),
                "factor_r2_status": getattr(row, "factor_r2_status", ""),
                "etf_share_raw": getattr(row, "etf_share_raw", np.nan),
                "turnover_mean_trailing": getattr(row, "turnover_mean_trailing", np.nan),
            })

        priced_now = prices.columns[prices.loc[m].notna()]
        tgt, idea_diag = target_weights_from_versions(
            selected_versions,
            cfg.portfolio,
            active_benchmark_weights,
            idiosyncratic_vol,
            eligible_tickers=priced_now,
            return_diagnostics=True,
            active_benchmark_source=cfg.active_benchmark_source,
        )
        idea_audit = idea_diag.get("_idea_audit", pd.DataFrame())
        if isinstance(idea_audit, pd.DataFrame) and not idea_audit.empty:
            idea_audit = _add_freshness_audit_columns(idea_audit, m, cfg.universe)
            idea_audit.insert(0, "rebalance_month", m.date().isoformat())
            idea_rows.extend(idea_audit.to_dict(orient="records"))
        target_names_before_price_filter = int(len(tgt))
        tgt = tgt[tgt.index.isin(priced_now)]
        invalid_reason = _minimum_target_failure(tgt, cfg)
        if invalid_reason:
            eff, traded, portfolio_stats, issuer_stats, feasibility, trade_stats = _invalid_cash_rebalance(
                cur,
                cfg,
                security_groups,
            )
            cost_bps = traded * cfg.cost.bps_per_side
            summary_rows.append({
                "rebalance_month": m.date().isoformat(),
                "selected_managers": int(len(selected_versions)),
                **stale_diag,
                **manager_filter_diag,
                "active_eligible_managers": int(idea_diag["active_eligible_managers"]),
                "zero_contributor_managers": int(idea_diag["zero_contributor_managers"]),
                "raw_idea_rows": int(idea_diag["raw_idea_rows"]),
                "raw_idea_names": int(idea_diag["raw_idea_names"]),
                "consensus_idea_names": int(idea_diag["consensus_idea_names"]),
                "target_names": int(len(tgt)),
                "target_names_before_price_filter": target_names_before_price_filter,
                "target_names_before_caps": int(idea_diag["target_names_before_caps"]),
                "idio_vol_covered_names": int(idea_diag["idio_vol_covered_names"]),
                "market_cap_covered_names": int(idea_diag["market_cap_covered_names"]),
                "market_cap_eligible_managers": int(idea_diag["market_cap_eligible_managers"]),
                "market_cap_mean_book_coverage": float(idea_diag["market_cap_mean_book_coverage"]),
                "max_distinct_ideas_upper_bound": int(idea_diag["max_distinct_ideas_upper_bound"]),
                "effective_names": int(len(eff)),
                "carried_names": 0,
                "valid_rebalance": False,
                "turnover_one_way": float(traded),
                "cost_bps": float(cost_bps),
                **portfolio_stats,
                **issuer_stats,
                **feasibility,
                **trade_stats,
                "top_holdings": "",
                "note": invalid_reason,
            })
            cur = eff
            last_in_tgt.clear()
            last_sources.clear()
            continue

        current_sources = idea_diag.get("_source_managers", {})
        for ticker in tgt.index:
            last_sources[str(ticker)] = list(current_sources.get(str(ticker), []))
        eff, traded, carried, combined_pre_cap = _apply_rebalance_target(
            cur,
            last_in_tgt,
            tgt,
            cfg,
            m,
            security_groups,
            return_pre_cap=True,
        )
        last_sources = {ticker: sources for ticker, sources in last_sources.items() if ticker in last_in_tgt}
        cost_bps = traded * cfg.cost.bps_per_side
        portfolio_stats = _portfolio_summary_stats(eff)
        issuer_stats = _issuer_exposure_summary(eff, security_groups)
        feasibility = _constraint_feasibility(eff, cfg, security_groups)
        trade_stats = _trade_summary_stats(cur, eff)
        top = eff.sort_values(ascending=False).head(12)
        summary_rows.append({
            "rebalance_month": m.date().isoformat(),
            "selected_managers": int(len(selected_versions)),
            **stale_diag,
            **manager_filter_diag,
            "active_eligible_managers": int(idea_diag["active_eligible_managers"]),
            "zero_contributor_managers": int(idea_diag["zero_contributor_managers"]),
            "raw_idea_rows": int(idea_diag["raw_idea_rows"]),
            "raw_idea_names": int(idea_diag["raw_idea_names"]),
            "consensus_idea_names": int(idea_diag["consensus_idea_names"]),
            "target_names": int(len(tgt)),
            "target_names_before_price_filter": target_names_before_price_filter,
            "target_names_before_caps": int(idea_diag["target_names_before_caps"]),
            "idio_vol_covered_names": int(idea_diag["idio_vol_covered_names"]),
            "market_cap_covered_names": int(idea_diag["market_cap_covered_names"]),
            "market_cap_eligible_managers": int(idea_diag["market_cap_eligible_managers"]),
            "market_cap_mean_book_coverage": float(idea_diag["market_cap_mean_book_coverage"]),
            "max_distinct_ideas_upper_bound": int(idea_diag["max_distinct_ideas_upper_bound"]),
            "effective_names": int(len(eff)),
            "carried_names": int(len(carried)),
            "valid_rebalance": True,
            "turnover_one_way": float(traded),
            "cost_bps": float(cost_bps),
            **portfolio_stats,
            **issuer_stats,
            **feasibility,
            **trade_stats,
            "top_holdings": "; ".join(f"{k}:{v:.2%}" for k, v in top.items()),
            "note": "",
        })
        carried_set = set(carried)
        target_pre_cap = idea_diag.get("_target_weights_pre_cap", pd.Series(dtype=float))
        groups = _security_groups_for(eff.index, security_groups).reindex(eff.index)
        for rank, (ticker, weight) in enumerate(eff.sort_values(ascending=False).items(), start=1):
            sources = last_sources.get(str(ticker), [])
            carry_age_q = int((m.to_period("Q") - last_in_tgt[ticker].to_period("Q")).n) if ticker in last_in_tgt else 0
            holding_rows.append({
                "rebalance_month": m.date().isoformat(),
                "rank": int(rank),
                "ticker": ticker,
                "issuer_group": groups.loc[ticker],
                "weight": float(weight),
                "pre_cap_weight": float(combined_pre_cap.get(ticker, np.nan)),
                "target_weight_pre_cap": float(target_pre_cap.get(ticker, np.nan)),
                "target_weight_post_name_cap": float(tgt.get(ticker, np.nan)),
                "post_cap_weight": float(weight),
                "is_carried": bool(ticker in carried_set),
                "carry_age_q": carry_age_q,
                "source_manager_count": int(len(sources)),
                "source_managers": ";".join(sources),
            })
            for manager in sources:
                holding_source_rows.append({
                    "rebalance_month": m.date().isoformat(),
                    "ticker": ticker,
                    "manager": manager,
                    "is_carried": bool(ticker in carried_set),
                    "carry_age_q": carry_age_q,
                })
        cur = eff

    return {
        "summary": pd.DataFrame(summary_rows),
        "holdings": pd.DataFrame(
            holding_rows,
            columns=[
                "rebalance_month", "rank", "ticker", "issuer_group", "weight",
                "pre_cap_weight", "target_weight_pre_cap", "target_weight_post_name_cap",
                "post_cap_weight", "is_carried", "carry_age_q", "source_manager_count", "source_managers",
            ],
        ),
        "holding_sources": pd.DataFrame(
            holding_source_rows,
            columns=["rebalance_month", "ticker", "manager", "is_carried", "carry_age_q"],
        ),
        "manager_candidates_audit": pd.DataFrame(manager_candidate_rows),
        "ideas": pd.DataFrame(idea_rows),
        "managers": pd.DataFrame(
            manager_rows,
            columns=[
                "rebalance_month",
                "manager",
                "manager_name",
                "manager_style",
                "dirty_flag",
                "dirty_reason",
                "classification_source",
                "factor_r2",
                "factor_r2_status",
                "etf_share_raw",
                "turnover_mean_trailing",
            ],
        ),
    }


# --------------------------------------------------------------------------- #
def run_backtest(holdings, prices, cfg: BacktestConfig,
                 value_scores=None, benchmark_weights=None, chars=None,
                 visible_versions_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
                 security_groups=None,
                 active_benchmark_weights_by_month: dict[pd.Timestamp, pd.Series] | None = None,
                 manager_classification: pd.DataFrame | None = None,
                 manager_overrides: pd.DataFrame | None = None,
                 progress_label: str | None = None,
                 progress_every: int = 10,
                 capture_rebalance: bool = False,
                 idiosyncratic_vol_by_month: dict[pd.Timestamp, pd.Series] | None = None) -> pd.Series:
    if chars is None:
        chars = manager_characteristics(holdings, benchmark_weights)
    months = prices.index.sort_values()
    rebal_month_list = _rebalance_months(holdings, prices)
    rebal_months = set(rebal_month_list)
    total_rebalances = len(rebal_month_list)

    cur = pd.Series(dtype=float)
    last_in_tgt: dict[str, pd.Timestamp] = {}
    net = pd.Series(0.0, index=months)
    rebal_no = 0
    t0 = time.perf_counter()
    summary_rows: list[dict] = []

    for m in months:
        cur, net.loc[m] = _apply_monthly_returns(cur, prices, m, cfg)
        if m in rebal_months:
            rebal_no += 1
            latest_versions = visible_versions_cache.get(pd.Timestamp(m)) if visible_versions_cache is not None else None
            if latest_versions is None:
                latest_versions = _visible_manager_versions(chars, m)
            latest_versions, stale_diag = _filter_fresh_versions(latest_versions, m, cfg.universe)
            active_benchmark_weights = _active_benchmark_for_month(
                cfg,
                m,
                latest_versions,
                active_benchmark_weights_by_month,
            )
            idiosyncratic_vol = _idiosyncratic_vol_for_month(cfg, m, idiosyncratic_vol_by_month)
            selected_versions = filter_universe_versions(latest_versions, cfg.universe, value_scores)
            selected_versions = _apply_manager_type_filter(
                selected_versions,
                m,
                cfg,
                manager_classification,
                manager_overrides,
            )
            manager_filter_diag = _manager_filter_diagnostics(selected_versions, cfg)
            priced_now = prices.columns[prices.loc[m].notna()]
            tgt, idea_diag = target_weights_from_versions(
                selected_versions,
                cfg.portfolio,
                active_benchmark_weights,
                idiosyncratic_vol,
                eligible_tickers=priced_now,
                return_diagnostics=True,
                active_benchmark_source=cfg.active_benchmark_source,
            )
            target_names_before_price_filter = len(tgt)
            tgt = tgt[tgt.index.isin(priced_now)]
            invalid_reason = _minimum_target_failure(tgt, cfg)
            if invalid_reason:
                eff, traded, portfolio_stats, issuer_stats, feasibility, trade_stats = _invalid_cash_rebalance(
                    cur,
                    cfg,
                    security_groups,
                )
                net.loc[m] -= traded * cfg.cost.bps_per_side / 1e4
                cur = eff
                last_in_tgt.clear()
            else:
                before_rebalance = cur.copy()
                eff, traded, _ = _apply_rebalance_target(cur, last_in_tgt, tgt, cfg, m, security_groups)
                net.loc[m] -= traded * cfg.cost.bps_per_side / 1e4
                cur = eff
                portfolio_stats = _portfolio_summary_stats(eff)
                issuer_stats = _issuer_exposure_summary(eff, security_groups)
                feasibility = _constraint_feasibility(eff, cfg, security_groups)
                trade_stats = _trade_summary_stats(before_rebalance, eff)
            if capture_rebalance:
                summary_rows.append({
                    "rebalance_month": m.date().isoformat(),
                    "selected_managers": int(len(selected_versions)),
                    **stale_diag,
                    **manager_filter_diag,
                    "active_eligible_managers": int(idea_diag["active_eligible_managers"]),
                    "zero_contributor_managers": int(idea_diag["zero_contributor_managers"]),
                    "raw_idea_rows": int(idea_diag["raw_idea_rows"]),
                    "raw_idea_names": int(idea_diag["raw_idea_names"]),
                    "consensus_idea_names": int(idea_diag["consensus_idea_names"]),
                    "target_names": int(len(tgt)),
                    "target_names_before_price_filter": int(target_names_before_price_filter),
                    "target_names_before_caps": int(idea_diag["target_names_before_caps"]),
                    "idio_vol_covered_names": int(idea_diag["idio_vol_covered_names"]),
                    "market_cap_covered_names": int(idea_diag["market_cap_covered_names"]),
                    "market_cap_eligible_managers": int(idea_diag["market_cap_eligible_managers"]),
                    "market_cap_mean_book_coverage": float(idea_diag["market_cap_mean_book_coverage"]),
                    "max_distinct_ideas_upper_bound": int(idea_diag["max_distinct_ideas_upper_bound"]),
                    "effective_names": int(len(cur)),
                    "valid_rebalance": not bool(invalid_reason),
                    "turnover_one_way": float(traded),
                    "cost_bps": float(traded * cfg.cost.bps_per_side),
                    **portfolio_stats,
                    **issuer_stats,
                    **feasibility,
                    **trade_stats,
                    "note": invalid_reason,
                })
            if progress_label and (
                rebal_no == 1
                or rebal_no == total_rebalances
                or (progress_every > 0 and rebal_no % progress_every == 0)
            ):
                print(
                    f"      {progress_label}: rebalance {rebal_no}/{total_rebalances} "
                    f"{m.date()} selected={len(selected_versions)} "
                    f"target={target_names_before_price_filter}->{len(tgt)} "
                    f"held={len(cur)} elapsed={time.perf_counter() - t0:.1f}s"
                )
    if capture_rebalance:
        net.attrs["rebalance_summary"] = pd.DataFrame(summary_rows)
    return net


def build_rebalance_selection_cache(
    holdings,
    prices,
    cfg: BacktestConfig,
    value_scores=None,
    benchmark_weights=None,
    chars=None,
    visible_versions_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
    manager_classification: pd.DataFrame | None = None,
    manager_overrides: pd.DataFrame | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Precompute PIT universe selections for each rebalance month.

    This is useful for parameter sweeps where many portfolio rules share the
    same universe rule. It deliberately caches only the selected filing versions,
    not the final portfolio target, so portfolio parameters are still evaluated
    independently.
    """
    if chars is None:
        chars = manager_characteristics(holdings, benchmark_weights)
    selected_by_month: dict[pd.Timestamp, pd.DataFrame] = {}
    for m in _rebalance_months(holdings, prices):
        month = pd.Timestamp(m)
        latest_versions = visible_versions_cache.get(month) if visible_versions_cache is not None else None
        if latest_versions is None:
            latest_versions = _visible_manager_versions(chars, month)
        fresh_versions, stale_diag = _filter_fresh_versions(latest_versions, month, cfg.universe)
        selected = filter_universe_versions(fresh_versions, cfg.universe, value_scores)
        selected = _apply_manager_type_filter(
            selected,
            month,
            cfg,
            manager_classification,
            manager_overrides,
        )
        selected.attrs.update(stale_diag)
        selected_by_month[month] = selected
    return selected_by_month


def build_active_benchmark_weights_cache(
    holdings,
    prices,
    benchmark_weights=None,
    chars=None,
    visible_versions_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
    cfg: BacktestConfig | UniverseConfig | None = None,
    active_benchmark_weights_by_month: dict[pd.Timestamp, pd.Series] | None = None,
) -> dict[pd.Timestamp, pd.Series]:
    """Precompute PIT aggregate 13F book weights used by active_weight ideas."""
    if isinstance(cfg, BacktestConfig) and not _uses_visible_13f_active_benchmark(cfg):
        if active_benchmark_weights_by_month is None:
            raise ValueError(
                f"{cfg.active_benchmark_source} active benchmark requires active_benchmark_weights_by_month"
            )
        return {pd.Timestamp(k): v for k, v in active_benchmark_weights_by_month.items()}
    if chars is None:
        chars = manager_characteristics(holdings, benchmark_weights)
    universe_cfg = cfg.universe if isinstance(cfg, BacktestConfig) else (cfg if isinstance(cfg, UniverseConfig) else UniverseConfig())
    active_by_month: dict[pd.Timestamp, pd.Series] = {}
    for m in _rebalance_months(holdings, prices):
        month = pd.Timestamp(m)
        latest_versions = visible_versions_cache.get(month) if visible_versions_cache is not None else None
        if latest_versions is None:
            latest_versions = _visible_manager_versions(chars, month)
        latest_versions, _ = _filter_fresh_versions(latest_versions, month, universe_cfg)
        active_by_month[month] = _aggregate_book_weights(latest_versions["bw"]) if not latest_versions.empty else pd.Series(dtype=float)
    return active_by_month


def run_backtest_from_selection_cache(
    prices,
    cfg: BacktestConfig,
    selected_versions_by_month: dict[pd.Timestamp, pd.DataFrame],
    active_benchmark_weights_by_month: dict[pd.Timestamp, pd.Series] | None = None,
    security_groups=None,
    progress_label: str | None = None,
    progress_every: int = 10,
    capture_rebalance: bool = False,
    idiosyncratic_vol_by_month: dict[pd.Timestamp, pd.Series] | None = None,
) -> pd.Series:
    """Run the monthly portfolio simulation from precomputed PIT selections."""
    months = prices.index.sort_values()
    rebal_months = {pd.Timestamp(m) for m in selected_versions_by_month}
    total_rebalances = len(rebal_months)

    cur = pd.Series(dtype=float)
    last_in_tgt: dict[str, pd.Timestamp] = {}
    net = pd.Series(0.0, index=months)
    rebal_no = 0
    t0 = time.perf_counter()
    summary_rows: list[dict] = []

    for m in months:
        cur, net.loc[m] = _apply_monthly_returns(cur, prices, m, cfg)
        month = pd.Timestamp(m)
        if month in rebal_months:
            rebal_no += 1
            selected_versions = selected_versions_by_month.get(month, pd.DataFrame())
            stale_diag = _selection_cache_diagnostics(selected_versions)
            manager_filter_diag = _manager_filter_diagnostics(selected_versions, cfg)
            active_benchmark_weights = None
            if _needs_active_benchmark_weights(cfg.portfolio.idea_signal):
                if active_benchmark_weights_by_month is not None:
                    active_benchmark_weights = active_benchmark_weights_by_month.get(month)
            idiosyncratic_vol = _idiosyncratic_vol_for_month(cfg, month, idiosyncratic_vol_by_month)
            priced_now = prices.columns[prices.loc[m].notna()]
            tgt, idea_diag = target_weights_from_versions(
                selected_versions,
                cfg.portfolio,
                active_benchmark_weights,
                idiosyncratic_vol,
                eligible_tickers=priced_now,
                return_diagnostics=True,
                active_benchmark_source=cfg.active_benchmark_source,
            )
            target_names_before_price_filter = len(tgt)
            tgt = tgt[tgt.index.isin(priced_now)]
            invalid_reason = _minimum_target_failure(tgt, cfg)
            if invalid_reason:
                eff, traded, portfolio_stats, issuer_stats, feasibility, trade_stats = _invalid_cash_rebalance(
                    cur,
                    cfg,
                    security_groups,
                )
                net.loc[m] -= traded * cfg.cost.bps_per_side / 1e4
                cur = eff
                last_in_tgt.clear()
            else:
                before_rebalance = cur.copy()
                eff, traded, _ = _apply_rebalance_target(cur, last_in_tgt, tgt, cfg, m, security_groups)
                net.loc[m] -= traded * cfg.cost.bps_per_side / 1e4
                cur = eff
                portfolio_stats = _portfolio_summary_stats(eff)
                issuer_stats = _issuer_exposure_summary(eff, security_groups)
                feasibility = _constraint_feasibility(eff, cfg, security_groups)
                trade_stats = _trade_summary_stats(before_rebalance, eff)
            if capture_rebalance:
                summary_rows.append({
                    "rebalance_month": m.date().isoformat(),
                    "selected_managers": int(len(selected_versions)),
                    **stale_diag,
                    **manager_filter_diag,
                    "active_eligible_managers": int(idea_diag["active_eligible_managers"]),
                    "zero_contributor_managers": int(idea_diag["zero_contributor_managers"]),
                    "raw_idea_rows": int(idea_diag["raw_idea_rows"]),
                    "raw_idea_names": int(idea_diag["raw_idea_names"]),
                    "consensus_idea_names": int(idea_diag["consensus_idea_names"]),
                    "target_names": int(len(tgt)),
                    "target_names_before_price_filter": int(target_names_before_price_filter),
                    "target_names_before_caps": int(idea_diag["target_names_before_caps"]),
                    "idio_vol_covered_names": int(idea_diag["idio_vol_covered_names"]),
                    "market_cap_covered_names": int(idea_diag["market_cap_covered_names"]),
                    "market_cap_eligible_managers": int(idea_diag["market_cap_eligible_managers"]),
                    "market_cap_mean_book_coverage": float(idea_diag["market_cap_mean_book_coverage"]),
                    "max_distinct_ideas_upper_bound": int(idea_diag["max_distinct_ideas_upper_bound"]),
                    "effective_names": int(len(cur)),
                    "valid_rebalance": not bool(invalid_reason),
                    "turnover_one_way": float(traded),
                    "cost_bps": float(traded * cfg.cost.bps_per_side),
                    **portfolio_stats,
                    **issuer_stats,
                    **feasibility,
                    **trade_stats,
                    "note": invalid_reason,
                })
            if progress_label and (
                rebal_no == 1
                or rebal_no == total_rebalances
                or (progress_every > 0 and rebal_no % progress_every == 0)
            ):
                print(
                    f"      {progress_label}: rebalance {rebal_no}/{total_rebalances} "
                    f"{m.date()} selected={len(selected_versions)} "
                    f"target={target_names_before_price_filter}->{len(tgt)} "
                    f"held={len(cur)} elapsed={time.perf_counter() - t0:.1f}s"
                )
    if capture_rebalance:
        net.attrs["rebalance_summary"] = pd.DataFrame(summary_rows)
    return net


# --------------------------------------------------------------------------- #
def attribution(port_ret, factors, benchmark=None,
                factor_cols=("MKT", "SMB", "HML", "RMW", "CMA", "MOM")) -> dict:
    factors = factors if factors is not None else pd.DataFrame(index=port_ret.index)
    df = pd.concat([port_ret.rename("ret"), factors], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ret"])
    if len(df) < 12:
        return {"n_months": len(df), "note": "insufficient overlap"}
    if "RF" in df:
        y_basic = (df["ret"] - df["RF"]).replace([np.inf, -np.inf], np.nan).dropna()
    else:
        y_basic = df["ret"].replace([np.inf, -np.inf], np.nan).dropna()
    out = {
        "n_months": len(df),
        "ann_return": (1 + df["ret"]).prod() ** (12 / len(df)) - 1,
        "ann_vol": df["ret"].std() * np.sqrt(12),
        "sharpe": (
            (y_basic.mean() / y_basic.std()) * np.sqrt(12)
            if pd.notna(y_basic.std()) and y_basic.std() > 1e-12 else np.nan
        ),
    }
    if df["ret"].abs().max() <= 1e-12:
        out["note"] = "all-cash or constant-zero return stream; factor alpha is not meaningful"
        if benchmark is not None:
            active = pd.concat([df["ret"], benchmark.reindex(df.index).rename("bench")], axis=1).dropna()
            active = active["ret"] - active["bench"]
            out["ir_vs_benchmark"] = (
                (active.mean() / active.std()) * np.sqrt(12)
                if pd.notna(active.std()) and active.std() > 1e-12 else np.nan
            )
        return out
    missing_factor_cols = [c for c in factor_cols if c not in df.columns]
    if "RF" not in df.columns or missing_factor_cols:
        out["note"] = "factor regression unavailable"
        if benchmark is not None:
            active = pd.concat([df["ret"], benchmark.reindex(df.index).rename("bench")], axis=1)
            active = active.replace([np.inf, -np.inf], np.nan).dropna()
            active = active["ret"] - active["bench"]
            out["ir_vs_benchmark"] = (active.mean() / active.std()) * np.sqrt(12) if active.std() else np.nan
        return out

    reg_cols = ["ret", "RF", *factor_cols]
    reg_df = df[reg_cols].replace([np.inf, -np.inf], np.nan).dropna()
    out["factor_months_used"] = int(len(reg_df))
    min_reg_months = len(factor_cols) + 3
    if len(reg_df) < min_reg_months:
        out["note"] = "factor regression unavailable: insufficient complete factor rows"
        if benchmark is not None:
            active = pd.concat([df["ret"], benchmark.reindex(df.index).rename("bench")], axis=1)
            active = active.replace([np.inf, -np.inf], np.nan).dropna()
            active = active["ret"] - active["bench"]
            out["ir_vs_benchmark"] = (active.mean() / active.std()) * np.sqrt(12) if active.std() else np.nan
        return out

    y = reg_df["ret"] - reg_df["RF"]
    x = sm.add_constant(reg_df[list(factor_cols)], has_constant="add")
    res = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": min(6, len(reg_df) - 1)})
    out.update({
        "ann_alpha": (1 + res.params["const"]) ** 12 - 1,
        "alpha_t": res.tvalues["const"],
        "betas": {c: res.params[c] for c in factor_cols},
    })
    if benchmark is not None:
        active = pd.concat([df["ret"], benchmark.reindex(df.index).rename("bench")], axis=1)
        active = active.replace([np.inf, -np.inf], np.nan).dropna()
        active = active["ret"] - active["bench"]
        out["ir_vs_benchmark"] = (active.mean() / active.std()) * np.sqrt(12) if active.std() else np.nan
    return out


def _fmt_metric(value) -> str:
    return f"{value:.4g}" if isinstance(value, (int, float, np.floating)) and np.isfinite(value) else str(value)


def _active_ir_metric(port_ret: pd.Series, benchmark: pd.Series | None = None) -> float:
    if benchmark is None:
        r = port_ret.replace([np.inf, -np.inf], np.nan).dropna()
    else:
        active = pd.concat([port_ret.rename("ret"), benchmark.reindex(port_ret.index).rename("bench")], axis=1)
        active = active.replace([np.inf, -np.inf], np.nan).dropna()
        r = active["ret"] - active["bench"]
    return float((r.mean() / r.std()) * np.sqrt(12)) if r.std() else np.nan


def marginal_ir(holdings, prices, factors, cfg, benchmark=None, value_scores=None, benchmark_weights=None,
                chars=None, visible_versions_cache=None, security_groups=None,
                active_benchmark_weights_by_month=None,
                manager_classification: pd.DataFrame | None = None,
                manager_overrides: pd.DataFrame | None = None,
                verbose: bool = False,
                idiosyncratic_vol_by_month=None):
    ch = chars if chars is not None else manager_characteristics(holdings, benchmark_weights)
    visible_cache = visible_versions_cache if visible_versions_cache is not None else build_visible_versions_cache(ch, prices.index)
    if verbose:
        print("  marginal-ir 1/? running full stack")
        t0 = time.perf_counter()
    base_ret = run_backtest(
        holdings,
        prices,
        cfg,
        value_scores,
        benchmark_weights,
        ch,
        visible_cache,
        security_groups,
        active_benchmark_weights_by_month,
        manager_classification,
        manager_overrides,
        idiosyncratic_vol_by_month=idiosyncratic_vol_by_month,
        progress_label="marginal-ir full stack" if verbose else None,
    )
    bm = _active_ir_metric(base_ret, benchmark)
    rows = [dict(filter="(full stack)", metric=bm, delta=0.0)]
    if verbose:
        print(f"    done full stack in {time.perf_counter() - t0:.1f}s metric={_fmt_metric(bm)}")
    filters = [t for t in [
        "use_size_band",
        "use_concentration",
        "use_low_turnover",
        "use_hedge_filter",
        "use_value_tilt",
        "use_active_share",
    ] if getattr(cfg.universe, t)]
    total = len(filters) + 1
    for i, t in enumerate(filters, start=2):
        if verbose:
            print(f"  marginal-ir {i}/{total} running -{t}")
            t0 = time.perf_counter()
        if not getattr(cfg.universe, t):
            continue
        cfg2 = replace(cfg, universe=replace(cfg.universe, **{t: False}))
        ret = run_backtest(
            holdings,
            prices,
            cfg2,
            value_scores,
            benchmark_weights,
            ch,
            visible_cache,
            security_groups,
            active_benchmark_weights_by_month,
            manager_classification,
            manager_overrides,
            idiosyncratic_vol_by_month=idiosyncratic_vol_by_month,
            progress_label=f"marginal-ir -{t}" if verbose else None,
        )
        m = _active_ir_metric(ret, benchmark)
        if verbose:
            print(f"    done -{t} in {time.perf_counter() - t0:.1f}s metric={_fmt_metric(m)}")
        rows.append(dict(filter=f"-{t}", metric=m, delta=(bm - m)))
    return pd.DataFrame(rows)
