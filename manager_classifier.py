"""Point-in-time manager-type classification for 13F idea universes.

The classifier is intentionally local-data only. It uses visible 13F filing
versions and public monthly return/factor inputs to label manager behavior for
audit and optional universe filtering. It does not use ADV, Bushee labels, or a
future-looking manager list.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    from data_adapters import _fund_like_mask
except Exception:  # pragma: no cover - keeps module importable in minimal tests
    _fund_like_mask = None


MANAGER_FILTER_MODES = {"all", "exclude_dirty", "dedicated_like"}


@dataclass(frozen=True)
class ManagerClassifierConfig:
    window_quarters: int = 8
    persistence_quarters: int = 2
    max_breadth: int = 200
    dedicated_max_breadth: int = 150
    dedicated_min_top10_weight: float = 0.50
    low_turnover_quantile: float = 0.34
    high_turnover_quantile: float = 0.66
    max_put_weight: float = 0.10
    max_etf_share: float = 0.50
    dedicated_max_etf_share: float = 0.10
    factor_r2_max: float = 0.90
    factor_r2_min_months: int = 12
    factor_r2_min_names: int = 3
    use_factor_r2_style: bool = False
    factor_cols: tuple[str, ...] = ("MKT", "SMB", "HML", "RMW", "CMA", "MOM")
    quasi_max_top10_weight: float = 0.35
    quasi_min_breadth: int = 100
    missing_classification_warn_frac: float = 0.10
    override_path: str = "data/manager_overrides.csv"


STATIC_DIRTY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("donor_advised_or_charity", r"\b(DONOR\s+ADVISED|CHARITABLE|CHARITY|FOUNDATION|ENDOWMENT)\b"),
    ("central_bank_or_swf", r"\b(CENTRAL\s+BANK|MONETARY\s+AUTHORITY|SOVEREIGN|INVESTMENT\s+AUTHORITY)\b"),
    ("broker_dealer_or_market_maker", r"\b(SECURITIES|BROKER[-\s]?DEALER|MARKET\s+MAKER|CLEARING|TRADING\s+LLC)\b"),
    ("bank_operating_holder", r"\b(NATIONAL\s+BANK|COMMERCIAL\s+BANK|BANK\s+OF)\b"),
    ("pension_or_retirement", r"\b(PENSION|RETIREMENT|TREASURY|PUBLIC\s+EMPLOYEES)\b"),
    ("etf_sponsor", r"\b(ISHARES|VANGUARD|SPDR|INVESCO|STATE\s+STREET\s+GLOBAL\s+ADVISORS)\b"),
)

FALSE_POSITIVE_GUARDS = (
    "BERKSHIRE",
    "MARKEL",
    "FAIRFAX",
    "WHITE MOUNTAINS",
)


def config_hash(config: ManagerClassifierConfig | dict[str, Any] | None = None) -> str:
    payload = asdict(config or ManagerClassifierConfig()) if not isinstance(config, dict) else dict(config)
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def frame_hash(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return hashlib.sha256(b"empty").hexdigest()[:16]
    stable = df.copy()
    for col in stable.columns:
        if pd.api.types.is_datetime64_any_dtype(stable[col]):
            stable[col] = pd.to_datetime(stable[col]).dt.strftime("%Y-%m-%d")
    stable = stable.sort_values([c for c in ["asof_month", "manager"] if c in stable.columns]).reset_index(drop=True)
    raw = stable.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def load_manager_overrides(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["manager", "action", "manager_type", "note"])
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["manager", "action", "manager_type", "note"])
    df = pd.read_csv(p)
    for col in ["manager", "action", "manager_type", "note"]:
        if col not in df:
            df[col] = ""
    out = df[["manager", "action", "manager_type", "note"]].copy()
    out["manager"] = out["manager"].astype(str).str.strip().str.zfill(10)
    out["action"] = out["action"].astype(str).str.strip().str.lower()
    out = out[out["manager"].ne("") & out["action"].isin(["allow", "deny"])]
    return out.drop_duplicates("manager", keep="last").reset_index(drop=True)


def override_file_hash(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _norm_ts(x) -> str:
    t = pd.to_datetime(x, errors="coerce")
    return t.normalize().strftime("%Y-%m-%d") if pd.notna(t) else ""


def _manager_name_map(raw_holdings: pd.DataFrame, filtered_holdings: pd.DataFrame) -> pd.Series:
    frames = [df for df in [raw_holdings, filtered_holdings] if df is not None and not df.empty]
    rows = []
    for df in frames:
        if "manager" not in df:
            continue
        tmp = df.copy()
        tmp["manager"] = tmp["manager"].astype(str).str.zfill(10)
        if "manager_name" not in tmp:
            tmp["manager_name"] = tmp["manager"]
        rows.append(tmp[["manager", "manager_name"]])
    if not rows:
        return pd.Series(dtype="string")
    names = pd.concat(rows, ignore_index=True).dropna(subset=["manager"])
    names["manager_name"] = names["manager_name"].astype(str).str.strip()
    return names.drop_duplicates("manager", keep="last").set_index("manager")["manager_name"]


def _static_dirty_reason(name: str) -> str:
    upper = str(name or "").upper()
    if any(guard in upper for guard in FALSE_POSITIVE_GUARDS):
        return ""
    for reason, pattern in STATIC_DIRTY_PATTERNS:
        if re.search(pattern, upper, flags=re.IGNORECASE):
            return reason
    return ""


def _raw_etf_share_tables(raw_holdings: pd.DataFrame) -> tuple[dict, dict]:
    """Return ETF-share lookup tables on raw, pre-security-filter books."""
    if raw_holdings is None or raw_holdings.empty or "manager" not in raw_holdings:
        return {}, {}
    h = raw_holdings.copy()
    h["manager"] = h["manager"].astype(str).str.zfill(10)
    h["_pd"] = h["period_date"].map(_norm_ts)
    h["_fd"] = h["filing_date"].map(_norm_ts)
    if "accession_number" not in h:
        h["accession_number"] = h["manager"] + "|" + h["_pd"] + "|" + h["_fd"]
    h["_acc"] = h["accession_number"].astype(str)
    sec = h.get("sec_type", pd.Series("SH", index=h.index)).fillna("SH").astype(str).str.upper()
    longs = h[sec.eq("SH")].copy()
    if longs.empty:
        return {}, {}
    if _fund_like_mask is not None:
        fund_like = _fund_like_mask(longs)
    else:
        fund_like = longs.get("is_fund_like", pd.Series(False, index=longs.index)).fillna(False).astype(bool)
    longs["_fund_like"] = np.asarray(fund_like).astype(bool)
    keys = ["manager", "_pd", "_fd", "_acc"]
    totals = longs.groupby(keys, dropna=False)["value"].sum()
    fund_vals = longs[longs["_fund_like"]].groupby(
        keys, dropna=False
    )["value"].sum()
    share = (fund_vals.reindex(totals.index).fillna(0.0) / totals.replace(0, np.nan)).fillna(0.0)
    full = {(m, pdt, fdt, acc): float(v) for (m, pdt, fdt, acc), v in share.items()}

    fb = share.rename("etf_share").reset_index()
    counts = fb.groupby(["manager", "_pd", "_fd"]).size()
    unique = counts[counts == 1].index
    fb = fb.set_index(["manager", "_pd", "_fd"])
    fallback = {idx: float(fb.loc[idx, "etf_share"]) for idx in unique}
    return full, fallback


def _factor_r2(book: pd.Series, prices: pd.DataFrame, factors: pd.DataFrame, asof, cfg: ManagerClassifierConfig) -> tuple[float, str]:
    if factors is None or factors.empty:
        return np.nan, "no_factors"
    missing_cols = [c for c in cfg.factor_cols if c not in factors.columns]
    if missing_cols or "RF" not in factors.columns:
        return np.nan, "no_factors"
    w = pd.Series(book, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    w = w[w > 0]
    tickers = [t for t in w.index if t in prices.columns]
    if len(tickers) < cfg.factor_r2_min_names:
        return np.nan, "insufficient_factor_r2"
    w = w.reindex(tickers)
    w = w / w.sum()
    end = pd.Timestamp(asof)
    window = prices.loc[prices.index <= end, tickers].tail(max(cfg.factor_r2_min_months, 12))
    if len(window.dropna(how="all")) < cfg.factor_r2_min_months:
        return np.nan, "insufficient_factor_r2"
    priced_weight = window.notna().mul(w, axis=1)
    denom = priced_weight.sum(axis=1)
    priced_count = window.notna().sum(axis=1)
    weighted_sum = window.mul(w, axis=1).sum(axis=1, min_count=cfg.factor_r2_min_names)
    port = (weighted_sum / denom.replace(0, np.nan)).where(priced_count >= cfg.factor_r2_min_names)
    df = pd.concat([port.rename("ret"), factors.reindex(port.index)], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ret", "RF", *cfg.factor_cols])
    if len(df) < cfg.factor_r2_min_months:
        return np.nan, "insufficient_factor_r2"
    try:
        y = df["ret"] - df["RF"]
        x = sm.add_constant(df[list(cfg.factor_cols)], has_constant="add")
        res = sm.OLS(y, x).fit()
        return float(res.rsquared), "ok"
    except Exception:
        return np.nan, "factor_r2_error"


def _reason_join(reasons: list[str]) -> str:
    return ";".join(dict.fromkeys([r for r in reasons if r]))


def _apply_dedicated_persistence(df: pd.DataFrame, cfg: ManagerClassifierConfig) -> pd.Series:
    """Calendar-quarter PIT persistence.

    A month in quarter Q can only use status confirmed through Q-1, not the
    quarter-end classification of Q. This prevents intra-quarter look-ahead when
    a quarter has multiple filing-driven rebalance months.
    """
    if df.empty:
        return pd.Series(False, index=df.index)
    out = pd.Series(False, index=df.index)
    work = df[["manager", "asof_month", "raw_dedicated"]].copy()
    work["asof_month"] = pd.to_datetime(work["asof_month"])
    work["quarter"] = work["asof_month"].dt.to_period("Q")
    k = max(1, int(cfg.persistence_quarters))
    for _, g in work.sort_values(["manager", "quarter", "asof_month"]).groupby("manager", sort=True):
        q = g.groupby("quarter", sort=True)["raw_dedicated"].last()
        active = False
        good_run = 0
        bad_run = 0
        status_after: dict[pd.Period, bool] = {}
        prev_q = None
        for quarter, is_good in q.items():
            if prev_q is not None and (quarter - prev_q).n != 1:
                active = False
                good_run = 0
                bad_run = 0
            if bool(is_good):
                good_run += 1
                bad_run = 0
            else:
                bad_run += 1
                good_run = 0
            if not active and good_run >= k:
                active = True
            elif active and bad_run >= k:
                active = False
            status_after[quarter] = active
            prev_q = quarter
        eligible_by_q = {quarter: bool(status_after.get(quarter - 1, False)) for quarter in q.index}
        out.loc[g.index] = g["quarter"].map(eligible_by_q).fillna(False).astype(bool).values
    return out


def build_manager_classification(
    raw_holdings: pd.DataFrame,
    filtered_holdings: pd.DataFrame,
    chars: pd.DataFrame,
    months,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    visible_versions_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
    config: ManagerClassifierConfig | None = None,
    progress=None,
) -> pd.DataFrame:
    cfg = config or ManagerClassifierConfig()
    if chars is None or chars.empty:
        return pd.DataFrame()
    names = _manager_name_map(raw_holdings, filtered_holdings)
    etf_full, etf_fallback = _raw_etf_share_tables(raw_holdings)
    months = pd.Index(pd.to_datetime(months)).sort_values()
    ch = chars.copy()
    ch["manager"] = ch["manager"].astype(str).str.zfill(10)
    ch["period_date"] = pd.to_datetime(ch["period_date"])
    ch["filing_date"] = pd.to_datetime(ch["filing_date"])
    rows: list[dict[str, Any]] = []
    r2_cache: dict[tuple[str, str], tuple[float, str]] = {}
    r2_versions_seen: set[tuple[str, str]] = set()
    etf_hits = 0
    etf_fallback_hits = 0
    etf_lookups = 0
    if progress is not None:
        progress(
            f"classifier input months={len(months)}, chars={len(ch):,}, "
            f"raw_rows={len(raw_holdings):,}, filtered_rows={len(filtered_holdings):,}"
        )
    for month_number, month in enumerate(months, start=1):
        asof = pd.Timestamp(month)
        latest = None if visible_versions_cache is None else visible_versions_cache.get(asof)
        if latest is None:
            latest = ch[ch["filing_date"] <= asof].sort_values(
                ["manager", "period_date", "filing_date", "accession_number"]
            ).groupby("manager", as_index=False).tail(1)
        if latest.empty:
            continue
        quarter_cutoff = asof.to_period("Q") - (cfg.window_quarters - 1)
        known = ch[ch["filing_date"] <= asof].copy()
        known = known[known["period_date"].dt.to_period("Q") >= quarter_cutoff]
        known_latest = (
            known.sort_values(["manager", "period_date", "filing_date", "accession_number"])
            .groupby(["manager", "period_date"], as_index=False)
            .tail(1)
        )
        turnover_by_manager = known_latest.groupby("manager")["turnover"].mean()
        latest = latest.copy().sort_values("manager")
        latest["manager"] = latest["manager"].astype(str).str.zfill(10)
        turnover_values = turnover_by_manager.replace([np.inf, -np.inf], np.nan).dropna()
        low_turnover = float(turnover_values.quantile(cfg.low_turnover_quantile)) if len(turnover_values) >= 3 else np.nan
        high_turnover = float(turnover_values.quantile(cfg.high_turnover_quantile)) if len(turnover_values) >= 3 else np.nan
        for r in latest.itertuples(index=False):
            manager = str(getattr(r, "manager")).zfill(10)
            manager_name = str(names.get(manager, manager))
            period_key = _norm_ts(getattr(r, "period_date"))
            filing_key = _norm_ts(getattr(r, "filing_date"))
            accession_key = str(getattr(r, "accession_number"))
            etf_lookups += 1
            full_key = (manager, period_key, filing_key, accession_key)
            fallback_key = (manager, period_key, filing_key)
            if full_key in etf_full:
                raw_etf = float(etf_full[full_key])
                etf_hits += 1
            elif fallback_key in etf_fallback:
                raw_etf = float(etf_fallback[fallback_key])
                etf_fallback_hits += 1
            else:
                raw_etf = 0.0
            turn = float(turnover_by_manager.get(manager, np.nan))
            n_holdings = int(getattr(r, "n_holdings", 0) or 0)
            top10 = float(getattr(r, "top10_weight", np.nan))
            put = float(getattr(r, "put_weight", 0.0) or 0.0)
            static_reason = _static_dirty_reason(manager_name)
            # Compute once when a filing version first becomes visible, then
            # carry that PIT-known diagnostic forward until a new filing takes
            # over. Including asof_month here caused the same unchanged filing
            # to be regressed every month.
            r2_key = (manager, accession_key)
            r2_versions_seen.add((manager, accession_key))
            if r2_key in r2_cache:
                r2, r2_status = r2_cache[r2_key]
            else:
                r2, r2_status = _factor_r2(getattr(r, "bw"), prices, factors, asof, cfg)
                r2_cache[r2_key] = (r2, r2_status)
            reasons = []
            source = "unclassified"
            if static_reason:
                reasons.append(static_reason)
                source = "name_static"
            if n_holdings > cfg.max_breadth:
                reasons.append("extreme_breadth")
                source = "behavior" if source == "unclassified" else source
            if raw_etf > cfg.max_etf_share:
                reasons.append("high_etf_share_raw")
                source = "behavior" if source == "unclassified" else source
            if put > cfg.max_put_weight:
                reasons.append("high_put_weight")
                source = "behavior" if source == "unclassified" else source
            if cfg.use_factor_r2_style and r2_status == "ok" and r2 >= cfg.factor_r2_max:
                manager_style = "quant_like"
            elif pd.notna(turn) and pd.notna(high_turnover) and turn >= high_turnover:
                manager_style = "transient"
            elif (
                pd.notna(turn)
                and pd.notna(low_turnover)
                and turn <= low_turnover
                and (top10 < cfg.quasi_max_top10_weight or n_holdings > cfg.quasi_min_breadth)
            ):
                manager_style = "quasi_indexer"
            elif (
                pd.notna(turn)
                and pd.notna(low_turnover)
                and turn <= low_turnover
                and n_holdings <= cfg.dedicated_max_breadth
                and top10 >= cfg.dedicated_min_top10_weight
                and put <= cfg.max_put_weight
                and raw_etf <= cfg.dedicated_max_etf_share
            ):
                manager_style = "dedicated"
            else:
                manager_style = "unclassified"
            if source == "unclassified" and manager_style != "unclassified":
                source = "behavior"
            rows.append(
                {
                    "asof_month": asof.normalize(),
                    "manager": manager,
                    "manager_name": manager_name,
                    "filer_static_type": static_reason or "",
                    "manager_style_raw": manager_style,
                    "manager_style": manager_style,
                    "dirty_flag": bool(reasons),
                    "dirty_reason": _reason_join(reasons),
                    "classification_source": source,
                    "turnover_mean_trailing": turn,
                    "n_holdings": n_holdings,
                    "top10_weight": top10,
                    "put_weight": put,
                    "etf_share_raw": raw_etf,
                    "factor_r2": r2,
                    "factor_r2_status": r2_status,
                    "history_quarters": int(getattr(r, "hist_q", 0) or 0),
                    "raw_dedicated": manager_style == "dedicated" and not bool(reasons),
                }
            )
        if progress is not None and (month_number == 1 or month_number % 12 == 0 or month_number == len(months)):
            progress(
                f"classifier month {month_number}/{len(months)} asof={asof.date()} "
                f"visible={len(latest):,} output_rows={len(rows):,} r2_runs={len(r2_cache):,}"
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if progress is not None:
        progress(f"classifier materializing {len(out):,} rows and applying quarterly persistence")
    out = out.sort_values(["manager", "asof_month"]).reset_index(drop=True)
    out["dedicated_persistent"] = _apply_dedicated_persistence(out, cfg)
    out.loc[out["dedicated_persistent"], "manager_style"] = "dedicated"
    out.loc[(out["manager_style_raw"].eq("dedicated")) & (~out["dedicated_persistent"]), "manager_style"] = "dedicated_pending"
    out = out.drop(columns=["raw_dedicated"])
    etf_match_rate = (etf_hits + etf_fallback_hits) / etf_lookups if etf_lookups else 0.0
    if (etf_full or etf_fallback) and etf_match_rate < 0.5:
        warnings.warn(
            f"manager_classifier: ETF-share key match rate is low ({etf_match_rate:.1%}); "
            "raw_holdings keys likely disagree with chars.",
            RuntimeWarning,
            stacklevel=2,
        )
    out.attrs["classification_config"] = asdict(cfg)
    out.attrs["classification_config_hash"] = config_hash(cfg)
    out.attrs["classification_hash"] = frame_hash(out)
    out.attrs["override_file_hash"] = override_file_hash(cfg.override_path)
    out.attrs["etf_key_match_rate"] = float(etf_match_rate)
    out.attrs["factor_r2_versions_computed"] = int(len(r2_versions_seen))
    out.attrs["factor_r2_regressions_computed"] = int(len(r2_cache))
    if progress is not None:
        progress(
            f"classifier complete rows={len(out):,}, r2_versions={len(r2_versions_seen):,}, "
            f"r2_runs={len(r2_cache):,}, etf_key_match={etf_match_rate:.1%}"
        )
    return out


def apply_manager_overrides(classification: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    if classification is None or classification.empty or overrides is None or overrides.empty:
        return classification
    out = classification.copy()
    ov = overrides.set_index("manager")
    common = out["manager"].astype(str).str.zfill(10).isin(ov.index)
    if not common.any():
        return out
    for idx in out.index[common]:
        manager = str(out.at[idx, "manager"]).zfill(10)
        row = ov.loc[manager]
        action = str(row["action"]).lower()
        note = str(row.get("note", "") or "").strip()
        if action == "deny":
            reasons = [out.at[idx, "dirty_reason"], "override_deny"]
            if note:
                reasons.append(note)
            out.at[idx, "dirty_flag"] = True
            out.at[idx, "dirty_reason"] = _reason_join(reasons)
            out.at[idx, "classification_source"] = "override"
            if str(row.get("manager_type", "") or "").strip():
                out.at[idx, "manager_style"] = str(row["manager_type"]).strip()
        elif action == "allow":
            out.at[idx, "dirty_flag"] = False
            out.at[idx, "dirty_reason"] = "override_allow" + (f";{note}" if note else "")
            out.at[idx, "manager_style"] = "dedicated"
            out.at[idx, "classification_source"] = "override_allow"
    out.attrs.update(classification.attrs)
    out.attrs["classification_hash"] = frame_hash(out)
    return out


def filter_selected_versions(
    selected: pd.DataFrame,
    month,
    mode: str,
    classification: pd.DataFrame | None,
    overrides: pd.DataFrame | None = None,
    *,
    config: ManagerClassifierConfig | None = None,
) -> pd.DataFrame:
    if mode == "all":
        return selected
    if mode not in MANAGER_FILTER_MODES:
        raise ValueError(f"Unknown manager_filter_mode={mode!r}")
    if selected is None or selected.empty:
        return selected
    warn_frac = (config or ManagerClassifierConfig()).missing_classification_warn_frac
    if classification is None or classification.empty:
        out = selected.iloc[0:0].copy()
        out.attrs.update(selected.attrs)
        out.attrs.update({
            "manager_filter_mode": mode,
            "manager_filter_before": int(len(selected)),
            "manager_filter_after": 0,
            "manager_filter_dropped": int(len(selected)),
            "manager_filter_missing_classification": int(len(selected)),
            "manager_filter_missing_classification_frac": 1.0 if len(selected) else 0.0,
        })
        warnings.warn(
            f"manager_filter mode={mode} has no classification rows for {pd.Timestamp(month).date()}; "
            "all managers dropped.",
            RuntimeWarning,
            stacklevel=2,
        )
        return out
    asof = pd.Timestamp(month).normalize()
    c = classification[pd.to_datetime(classification["asof_month"]).eq(asof)].copy()
    c["manager"] = c["manager"].astype(str).str.zfill(10)
    if mode != "all":
        c = apply_manager_overrides(c, overrides)
    selected2 = selected.copy()
    selected2["manager"] = selected2["manager"].astype(str).str.zfill(10)
    meta_cols = [
        "manager",
        "manager_name",
        "manager_style",
        "dirty_flag",
        "dirty_reason",
        "classification_source",
        "turnover_mean_trailing",
        "etf_share_raw",
        "factor_r2",
        "factor_r2_status",
    ]
    merged = selected2.merge(c[meta_cols], on="manager", how="left", suffixes=("", "_class"))
    missing = merged["manager_style"].isna()
    missing_frac = float(missing.mean()) if len(merged) else 0.0
    if missing_frac > warn_frac:
        warnings.warn(
            f"manager_filter mode={mode} @ {asof.date()}: {missing_frac:.1%} of selected managers "
            "have no classification row and will be dropped. This usually means a join/dtype mismatch.",
            RuntimeWarning,
            stacklevel=2,
        )
    dirty = merged["dirty_flag"].map(lambda x: bool(x) if pd.notna(x) else True)
    if mode == "exclude_dirty":
        keep = ~dirty
    elif mode == "dedicated_like":
        keep = (~dirty) & merged["manager_style"].eq("dedicated")
    else:
        keep = pd.Series(True, index=merged.index)
    out = merged.loc[keep].reset_index(drop=True)
    out.attrs.update(selected.attrs)
    out.attrs.update({
        "manager_filter_mode": mode,
        "manager_filter_before": int(len(selected)),
        "manager_filter_after": int(len(out)),
        "manager_filter_dropped": int(len(selected) - len(out)),
        "manager_filter_missing_classification": int(missing.sum()),
        "manager_filter_missing_classification_frac": missing_frac,
        "manager_filter_dirty_dropped": int((~keep & dirty).sum()),
        "manager_filter_non_dedicated_dropped": int((~keep & ~dirty).sum())
        if mode == "dedicated_like" else 0,
    })
    return out


def classification_summary(classification: pd.DataFrame) -> dict[str, Any]:
    if classification is None or classification.empty:
        return {"rows": 0}
    latest_month = pd.to_datetime(classification["asof_month"]).max()
    latest = classification[pd.to_datetime(classification["asof_month"]).eq(latest_month)]
    def counts(col: str) -> dict[str, int]:
        return {str(k): int(v) for k, v in latest[col].fillna("").value_counts().sort_index().items()}
    reason_counts: dict[str, int] = {}
    for reasons in latest["dirty_reason"].fillna(""):
        for reason in str(reasons).split(";"):
            reason = reason.strip()
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "rows": int(len(classification)),
        "latest_month": latest_month.date().isoformat() if pd.notna(latest_month) else None,
        "latest_managers": int(latest["manager"].nunique()),
        "style_counts_latest": counts("manager_style"),
        "source_counts_latest": counts("classification_source"),
        "dirty_reason_counts_latest": dict(sorted(reason_counts.items())),
        "classification_hash": classification.attrs.get("classification_hash"),
        "classification_config_hash": classification.attrs.get("classification_config_hash"),
        "override_file_hash": classification.attrs.get("override_file_hash"),
        "etf_key_match_rate": classification.attrs.get("etf_key_match_rate"),
        "factor_r2_versions_computed": classification.attrs.get("factor_r2_versions_computed"),
        "factor_r2_regressions_computed": classification.attrs.get("factor_r2_regressions_computed"),
    }
