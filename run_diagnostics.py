"""Run-level diagnostics kept separate from orchestration and domain logic."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engine import BacktestConfig, attribution, rebalance_trace, run_backtest


def trace_core_diagnostics(trace: dict[str, pd.DataFrame]) -> dict[str, Any]:
    summary = trace.get("summary", pd.DataFrame())
    holdings = trace.get("holdings", pd.DataFrame())
    if summary.empty:
        return {
            "avg_top_issuer_exposure": np.nan,
            "max_top_issuer_exposure": np.nan,
            "avg_effective_number": np.nan,
            "core_90_overlap_frac": np.nan,
            "permanent_core_name_count": 0,
            "permanent_core_names": "",
            "latest_top_holdings": "",
        }
    avg_top_issuer = float(pd.to_numeric(summary.get("max_issuer_weight", pd.Series(dtype=float)), errors="coerce").mean())
    max_top_issuer = float(pd.to_numeric(summary.get("max_issuer_weight", pd.Series(dtype=float)), errors="coerce").max())
    avg_eff = float(pd.to_numeric(summary.get("effective_number", pd.Series(dtype=float)), errors="coerce").mean())
    latest_top = str(summary.iloc[-1].get("top_holdings", ""))
    if holdings.empty:
        return {
            "avg_top_issuer_exposure": avg_top_issuer,
            "max_top_issuer_exposure": max_top_issuer,
            "avg_effective_number": avg_eff,
            "core_90_overlap_frac": np.nan,
            "permanent_core_name_count": 0,
            "permanent_core_names": "",
            "latest_top_holdings": latest_top,
        }
    sets = [set(group["ticker"].astype(str)) for _, group in holdings.groupby("rebalance_month", sort=True)]
    overlaps = []
    for previous, current in zip(sets, sets[1:]):
        denominator = min(len(previous), len(current))
        if denominator:
            overlaps.append(len(previous.intersection(current)) / denominator)
    core_90_frac = float(np.mean([value >= 0.90 for value in overlaps])) if overlaps else np.nan
    months = holdings["rebalance_month"].nunique()
    frequency = holdings.groupby("ticker")["rebalance_month"].nunique().sort_values(ascending=False)
    permanent = frequency[frequency >= max(1, int(np.ceil(months * 0.90)))].index.tolist()
    return {
        "avg_top_issuer_exposure": avg_top_issuer,
        "max_top_issuer_exposure": max_top_issuer,
        "avg_effective_number": avg_eff,
        "core_90_overlap_frac": core_90_frac,
        "permanent_core_name_count": int(len(permanent)),
        "permanent_core_names": ";".join(permanent[:25]),
        "latest_top_holdings": latest_top,
    }


def write_manager_filter_acceptance(
    out_dir: Path,
    *,
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    base_cfg: BacktestConfig,
    factors: pd.DataFrame,
    benchmark: pd.Series | None,
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
    rows = []
    for mode in ["all", "dedicated_like"]:
        cfg = replace(base_cfg, manager_filter_mode=mode)
        ret = run_backtest(
            holdings, prices, cfg, value_scores, benchmark_weights, chars,
            visible_versions_cache, security_groups, active_benchmark_weights_by_month,
            manager_classification, manager_overrides,
            idiosyncratic_vol_by_month=idiosyncratic_vol_by_month,
        )
        trace = rebalance_trace(
            holdings, prices, cfg, value_scores=value_scores,
            benchmark_weights=benchmark_weights, chars=chars,
            visible_versions_cache=visible_versions_cache,
            security_groups=security_groups,
            active_benchmark_weights_by_month=active_benchmark_weights_by_month,
            manager_classification=manager_classification,
            manager_overrides=manager_overrides,
            idiosyncratic_vol_by_month=idiosyncratic_vol_by_month,
        )
        att = attribution(ret, factors, benchmark)
        rows.append({
            "manager_filter_mode": mode,
            "ann_return": att.get("ann_return"),
            "ann_alpha": att.get("ann_alpha"),
            "alpha_t": att.get("alpha_t"),
            "smb_beta": (att.get("betas") or {}).get("SMB"),
            **trace_core_diagnostics(trace),
        })
    frame = pd.DataFrame(rows)
    if len(frame) == 2:
        all_row = frame[frame["manager_filter_mode"].eq("all")].iloc[0]
        dedicated_row = frame[frame["manager_filter_mode"].eq("dedicated_like")].iloc[0]
        unchanged = (
            pd.notna(all_row.get("smb_beta"))
            and pd.notna(dedicated_row.get("smb_beta"))
            and abs(float(all_row["smb_beta"]) - float(dedicated_row["smb_beta"])) < 0.05
            and float(dedicated_row.get("core_90_overlap_frac", 0) or 0) >= 0.80
        )
        finding = (
            "manager_cleaning_did_not_materially_move_core_book"
            if unchanged else "manager_cleaning_changed_book_diagnostics"
        )
    else:
        finding = "insufficient_acceptance_rows"
    frame["finding"] = finding
    path = out_dir / "manager_filter_acceptance.csv"
    frame.to_csv(path, index=False)
    print(f"  Saved manager-filter acceptance diagnostics: {path}")
    print(frame[[
        "manager_filter_mode", "smb_beta", "avg_top_issuer_exposure",
        "avg_effective_number", "core_90_overlap_frac",
        "permanent_core_name_count", "finding",
    ]].to_string(index=False))
    return {"path": str(path), "finding": finding}


def value_unit_continuity_diagnostics(
    chars: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp = "2023-01-01",
) -> pd.DataFrame:
    """Check for suspicious AUM jumps around the historical SEC value-unit cutoff."""
    columns = [
        "manager", "prev_period_date", "period_date", "prev_filing_date",
        "filing_date", "prev_aum", "aum", "aum_ratio", "abs_log10_ratio",
        "suspicious_unit_jump",
    ]
    if chars is None or chars.empty:
        return pd.DataFrame(columns=columns)
    cutoff_timestamp = pd.Timestamp(cutoff)
    latest = (
        chars.sort_values(["manager", "period_date", "filing_date", "accession_number"])
        .groupby(["manager", "period_date"], as_index=False)
        .tail(1)
        .sort_values(["manager", "period_date"])
    )
    rows: list[dict[str, Any]] = []
    for manager, group in latest.groupby("manager", sort=False):
        previous = None
        for row in group.sort_values("period_date").itertuples(index=False):
            if previous is not None and pd.Timestamp(previous.filing_date) < cutoff_timestamp <= pd.Timestamp(row.filing_date):
                previous_aum = float(previous.aum)
                aum = float(row.aum)
                ratio = aum / previous_aum if previous_aum > 0 else np.nan
                log_ratio = abs(float(np.log10(ratio))) if ratio and np.isfinite(ratio) and ratio > 0 else np.nan
                rows.append({
                    "manager": manager,
                    "prev_period_date": pd.Timestamp(previous.period_date).date().isoformat(),
                    "period_date": pd.Timestamp(row.period_date).date().isoformat(),
                    "prev_filing_date": pd.Timestamp(previous.filing_date).date().isoformat(),
                    "filing_date": pd.Timestamp(row.filing_date).date().isoformat(),
                    "prev_aum": previous_aum,
                    "aum": aum,
                    "aum_ratio": ratio,
                    "abs_log10_ratio": log_ratio,
                    "suspicious_unit_jump": bool(pd.notna(ratio) and (ratio >= 50.0 or ratio <= 0.02)),
                })
            previous = row
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values("abs_log10_ratio", ascending=False, na_position="last")
    return result


def _pct(value: float) -> str:
    return f"{value:.1%}" if pd.notna(value) else "n/a"


def rebalance_summary_stats(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {"rebalance_months": 0}
    out: dict[str, Any] = {"rebalance_months": int(len(summary))}
    numeric_columns = [
        "selected_managers", "visible_managers", "stale_managers_dropped",
        "stale_filing_managers", "stale_period_managers", "active_eligible_managers",
        "zero_contributor_managers", "market_cap_covered_names",
        "market_cap_eligible_managers", "market_cap_mean_book_coverage",
        "max_distinct_ideas_upper_bound", "raw_idea_rows", "raw_idea_names",
        "consensus_idea_names", "effective_names", "target_names",
        "target_names_before_caps", "carried_names", "turnover_one_way", "cost_bps",
        "max_weight", "issuer_groups", "max_issuer_weight", "top5_weight",
        "top10_weight", "effective_number", "traded_names", "buy_names", "sell_names",
    ]
    for column in numeric_columns:
        if column not in summary:
            continue
        values = pd.to_numeric(summary[column], errors="coerce").dropna()
        if not values.empty:
            out[f"avg_{column}"] = float(values.mean())
            out[f"max_{column}"] = float(values.max())
    last = summary.iloc[-1]
    for column in ["name_cap_feasible", "issuer_cap_feasible"]:
        if column in summary:
            values = summary[column].astype(bool)
            out[f"{column}_months"] = int(values.sum())
            out[f"{column}_ratio"] = float(values.mean()) if len(values) else float("nan")
    if "valid_rebalance" in summary:
        values = summary["valid_rebalance"].astype(bool)
        out["valid_rebalance_months"] = int(values.sum())
        out["valid_rebalance_ratio"] = float(values.mean()) if len(values) else float("nan")
        out["invalid_rebalance_months"] = int((~values).sum())
    if "effective_names" in summary:
        invested = pd.to_numeric(summary["effective_names"], errors="coerce").fillna(0).gt(0)
        out["invested_month_frac"] = float(invested.mean()) if len(invested) else float("nan")
    if {"zero_contributor_managers", "selected_managers"}.issubset(summary.columns):
        zero = pd.to_numeric(summary["zero_contributor_managers"], errors="coerce").fillna(0).sum()
        selected = pd.to_numeric(summary["selected_managers"], errors="coerce").fillna(0).sum()
        out["zero_contributor_manager_frac"] = float(zero / selected) if selected > 0 else float("nan")
    out.update({
        "last_rebalance_month": str(last.get("rebalance_month", "")),
        "last_effective_names": int(last.get("effective_names", 0) or 0),
        "last_turnover_one_way": float(last.get("turnover_one_way", 0.0) or 0.0),
        "last_max_weight": float(last.get("max_weight", 0.0) or 0.0),
        "last_max_issuer_weight": float(last.get("max_issuer_weight", 0.0) or 0.0),
        "last_top_holdings": str(last.get("top_holdings", "")),
        "last_top_issuer_exposures": str(last.get("top_issuer_exposures", "")),
        "last_multi_class_exposures": str(last.get("multi_class_exposures", "")),
    })
    return out


def print_rebalance_summary(stats: dict[str, Any]) -> None:
    if not stats or stats.get("rebalance_months", 0) == 0:
        print("  Rebalance summary: no rebalance months")
        return
    print("\n  Rebalance Summary")
    print(f"  months                 {stats['rebalance_months']}")
    print(f"  holdings avg/max       {stats.get('avg_effective_names', float('nan')):.1f}/{stats.get('max_effective_names', float('nan')):.0f}")
    if "valid_rebalance_ratio" in stats:
        print(f"  valid/invested months  {_pct(stats.get('valid_rebalance_ratio', float('nan')))}/{_pct(stats.get('invested_month_frac', float('nan')))}")
    if "zero_contributor_manager_frac" in stats:
        print(f"  zero contributor mgrs  {_pct(stats.get('zero_contributor_manager_frac', float('nan')))}")
    if "avg_market_cap_mean_book_coverage" in stats:
        print(f"  market-cap book cover  avg={_pct(stats.get('avg_market_cap_mean_book_coverage', float('nan')))} | eligible_mgrs={stats.get('avg_market_cap_eligible_managers', float('nan')):.1f} avg | covered_names={stats.get('avg_market_cap_covered_names', float('nan')):.0f} avg")
    if "avg_max_distinct_ideas_upper_bound" in stats:
        print(f"  idea count upper bound avg/max={stats.get('avg_max_distinct_ideas_upper_bound', float('nan')):.1f}/{stats.get('max_max_distinct_ideas_upper_bound', float('nan')):.0f}")
    if "avg_stale_managers_dropped" in stats:
        print(f"  stale mgrs dropped avg/max {stats.get('avg_stale_managers_dropped', float('nan')):.1f}/{stats.get('max_stale_managers_dropped', float('nan')):.0f}")
    print(f"  traded names avg/max   {stats.get('avg_traded_names', float('nan')):.1f}/{stats.get('max_traded_names', float('nan')):.0f}")
    print(f"  one-way turnover avg/max {_pct(stats.get('avg_turnover_one_way', float('nan')))}/{_pct(stats.get('max_turnover_one_way', float('nan')))}")
    print(f"  max weight avg/max     {_pct(stats.get('avg_max_weight', float('nan')))}/{_pct(stats.get('max_max_weight', float('nan')))}")
    print(f"  max issuer avg/max     {_pct(stats.get('avg_max_issuer_weight', float('nan')))}/{_pct(stats.get('max_max_issuer_weight', float('nan')))}")
    print(f"  issuer groups avg/max  {stats.get('avg_issuer_groups', float('nan')):.1f}/{stats.get('max_issuer_groups', float('nan')):.0f}")
    if "name_cap_feasible_ratio" in stats or "issuer_cap_feasible_ratio" in stats:
        print(f"  cap feasible months    name={_pct(stats.get('name_cap_feasible_ratio', float('nan')))} | issuer={_pct(stats.get('issuer_cap_feasible_ratio', float('nan')))}")
    print(f"  top10 weight avg/max   {_pct(stats.get('avg_top10_weight', float('nan')))}/{_pct(stats.get('max_top10_weight', float('nan')))}")
    print(f"  cost bps avg/max       {stats.get('avg_cost_bps', float('nan')):.2f}/{stats.get('max_cost_bps', float('nan')):.2f}")
    print(f"  latest rebalance       {stats.get('last_rebalance_month')} | names={stats.get('last_effective_names')} | turnover={_pct(stats.get('last_turnover_one_way', float('nan')))} | max_weight={_pct(stats.get('last_max_weight', float('nan')))}")
    if stats.get("last_top_holdings"):
        print(f"  latest top holdings    {stats['last_top_holdings']}")
    if stats.get("last_top_issuer_exposures"):
        print(f"  latest top issuers     {stats['last_top_issuer_exposures']}")
    if stats.get("last_multi_class_exposures"):
        print(f"  latest multi-class     {stats['last_multi_class_exposures']}")
