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
    factor_cols: tuple[str, ...] = ("MKT", "SMB", "HML", "RMW", "CMA", "MOM")
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


def _raw_etf_share_by_accession(raw_holdings: pd.DataFrame) -> pd.Series:
    if raw_holdings is None or raw_holdings.empty or "manager" not in raw_holdings:
        return pd.Series(dtype=float)
    h = raw_holdings.copy()
    if "accession_number" not in h:
        h["accession_number"] = (
            h["manager"].astype(str) + "|" + h["period_date"].astype(str) + "|" + h["filing_date"].astype(str)
        )
    h["manager"] = h["manager"].astype(str).str.zfill(10)
    sec = h.get("sec_type", pd.Series("SH", index=h.index)).fillna("SH").astype(str).str.upper()
    longs = h[sec.eq("SH")].copy()
    if longs.empty:
        return pd.Series(dtype=float)
    if _fund_like_mask is not None:
        fund_like = _fund_like_mask(longs)
    else:
        fund_like = longs.get("is_fund_like", pd.Series(False, index=longs.index)).fillna(False).astype(bool)
    longs["_fund_like"] = fund_like.astype(bool)
    totals = longs.groupby(["manager", "period_date", "filing_date", "accession_number"], dropna=False)["value"].sum()
    fund_vals = longs[longs["_fund_like"]].groupby(
        ["manager", "period_date", "filing_date", "accession_number"], dropna=False
    )["value"].sum()
    share = (fund_vals / totals.replace(0, np.nan)).fillna(0.0)
    return share


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
    port = (window * w).sum(axis=1, min_count=max(1, min(len(tickers), cfg.factor_r2_min_names)))
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
    if df.empty:
        return pd.Series(False, index=df.index)
    out = pd.Series(False, index=df.index)
    work = df[["manager", "asof_month", "raw_dedicated"]].copy()
    work["asof_month"] = pd.to_datetime(work["asof_month"])
    work["quarter"] = work["asof_month"].dt.to_period("Q")
    for manager, g in work.sort_values(["manager", "quarter", "asof_month"]).groupby("manager", sort=True):
        q = g.groupby("quarter", sort=True)["raw_dedicated"].last()
        active = False
        good_run = 0
        bad_run = 0
        q_active: dict[pd.Period, bool] = {}
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
            if not active and good_run >= cfg.persistence_quarters:
                active = True
            elif active and bad_run >= cfg.persistence_quarters:
                active = False
            q_active[quarter] = active
            prev_q = quarter
        idx = g.index
        out.loc[idx] = g["quarter"].map(q_active).fillna(False).astype(bool).values
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
) -> pd.DataFrame:
    cfg = config or ManagerClassifierConfig()
    if chars is None or chars.empty:
        return pd.DataFrame()
    names = _manager_name_map(raw_holdings, filtered_holdings)
    etf_share = _raw_etf_share_by_accession(raw_holdings)
    months = pd.Index(pd.to_datetime(months)).sort_values()
    ch = chars.copy()
    ch["manager"] = ch["manager"].astype(str).str.zfill(10)
    ch["period_date"] = pd.to_datetime(ch["period_date"])
    ch["filing_date"] = pd.to_datetime(ch["filing_date"])
    rows: list[dict[str, Any]] = []
    for month in months:
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
        turnover_by_manager = known.groupby("manager")["turnover"].mean()
        latest = latest.copy().sort_values("manager")
        latest["manager"] = latest["manager"].astype(str).str.zfill(10)
        turnover_values = turnover_by_manager.replace([np.inf, -np.inf], np.nan).dropna()
        low_turnover = float(turnover_values.quantile(cfg.low_turnover_quantile)) if len(turnover_values) >= 3 else np.nan
        high_turnover = float(turnover_values.quantile(cfg.high_turnover_quantile)) if len(turnover_values) >= 3 else np.nan
        for r in latest.itertuples(index=False):
            manager = str(getattr(r, "manager")).zfill(10)
            manager_name = str(names.get(manager, manager))
            key = (manager, getattr(r, "period_date"), getattr(r, "filing_date"), getattr(r, "accession_number"))
            raw_etf = float(etf_share.get(key, 0.0))
            turn = float(turnover_by_manager.get(manager, np.nan))
            n_holdings = int(getattr(r, "n_holdings", 0) or 0)
            top10 = float(getattr(r, "top10_weight", np.nan))
            put = float(getattr(r, "put_weight", 0.0) or 0.0)
            static_reason = _static_dirty_reason(manager_name)
            r2, r2_status = _factor_r2(getattr(r, "bw"), prices, factors, asof, cfg)
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
            if r2_status == "ok" and r2 >= cfg.factor_r2_max:
                manager_style = "quant_like"
            elif pd.notna(turn) and pd.notna(high_turnover) and turn >= high_turnover:
                manager_style = "transient"
            elif (
                pd.notna(turn)
                and pd.notna(low_turnover)
                and turn <= low_turnover
                and (top10 < 0.35 or n_holdings > 100)
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
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["manager", "asof_month"]).reset_index(drop=True)
    out["dedicated_persistent"] = _apply_dedicated_persistence(out, cfg)
    out.loc[out["dedicated_persistent"], "manager_style"] = "dedicated"
    out.loc[(out["manager_style_raw"].eq("dedicated")) & (~out["dedicated_persistent"]), "manager_style"] = "dedicated_pending"
    out = out.drop(columns=["raw_dedicated"])
    out.attrs["classification_config"] = asdict(cfg)
    out.attrs["classification_config_hash"] = config_hash(cfg)
    out.attrs["classification_hash"] = frame_hash(out)
    out.attrs["override_file_hash"] = override_file_hash(cfg.override_path)
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
            out.at[idx, "classification_source"] = "override"
    out.attrs.update(classification.attrs)
    out.attrs["classification_hash"] = frame_hash(out)
    return out


def filter_selected_versions(
    selected: pd.DataFrame,
    month,
    mode: str,
    classification: pd.DataFrame | None,
    overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if mode == "all":
        return selected
    if mode not in MANAGER_FILTER_MODES:
        raise ValueError(f"Unknown manager_filter_mode={mode!r}")
    if selected is None or selected.empty:
        return selected
    if classification is None or classification.empty:
        out = selected.iloc[0:0].copy()
        out.attrs.update(selected.attrs)
        out.attrs.update({
            "manager_filter_mode": mode,
            "manager_filter_before": int(len(selected)),
            "manager_filter_after": 0,
            "manager_filter_dropped": int(len(selected)),
            "manager_filter_missing_classification": int(len(selected)),
        })
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
    if mode == "exclude_dirty":
        keep = merged["dirty_flag"].fillna(True).eq(False)
    elif mode == "dedicated_like":
        keep = merged["dirty_flag"].fillna(True).eq(False) & merged["manager_style"].eq("dedicated")
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
        "manager_filter_dirty_dropped": int((~keep & merged["dirty_flag"].fillna(False).astype(bool)).sum()),
        "manager_filter_non_dedicated_dropped": int((~keep & merged["dirty_flag"].fillna(False).eq(False)).sum())
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
    }
