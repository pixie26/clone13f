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


# --------------------------------------------------------------------------- #
@dataclass
class UniverseConfig:
    min_aum: float = 1e9
    max_aum: float = 30e9
    top_n_concentration: int = 10
    min_top_n_weight: float = 0.50
    max_holdings: int = 40
    turnover_quantile: float = 0.34
    min_history_quarters: int = 4
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
    idea_signal: str = "level"              # "level" | "change" | "initiation" | "active_weight"
    consensus_weight: bool = True
    min_consensus_funds: int = 1            # drop names held by < this many in-universe funds
    max_name_weight: float = 0.05
    max_issuer_weight: float = 0.075
    holding_horizon_q: int = 0              # 0 = full rebalance; N = carry N extra quarters
    min_active_weight_holdings: int = 20    # active_weight needs a diversified book to be meaningful


@dataclass
class CostConfig:
    bps_per_side: float = 15.0


@dataclass
class BacktestConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    cost: CostConfig = field(default_factory=CostConfig)
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


def _versioned_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    h = holdings.copy()
    if "accession_number" not in h.columns:
        h["accession_number"] = (
            h["manager"].astype("string") + "|" +
            h["period_date"].astype("string") + "|" +
            h["filing_date"].astype("string")
        )
    if "submission_type" not in h.columns:
        h["submission_type"] = pd.NA
    return h


def _visible_manager_versions(chars: pd.DataFrame, asof, managers=None) -> pd.DataFrame:
    known = chars[chars["filing_date"] <= asof]
    if managers is not None:
        known = known[known["manager"].isin(managers)]
    if known.empty:
        return known
    return (known.sort_values(["manager", "period_date", "filing_date", "accession_number"])
                 .groupby("manager", as_index=False)
                 .tail(1))


def build_visible_versions_cache(chars: pd.DataFrame, months) -> dict[pd.Timestamp, pd.DataFrame]:
    """Cache latest visible manager filing versions for each month/asof."""
    return {pd.Timestamp(m): _visible_manager_versions(chars, pd.Timestamp(m)) for m in pd.Index(months).sort_values()}


def manager_characteristics(holdings: pd.DataFrame,
                            benchmark_weights: pd.Series | None = None) -> pd.DataFrame:
    holdings = _versioned_holdings(holdings)
    rows = []
    group_cols = ["manager", "period_date", "filing_date", "accession_number"]
    for (mgr, period, filing_date, accession), g in holdings.groupby(group_cols, dropna=False):
        w = _book_weights(g)
        sh = g.get("sec_type", pd.Series("SH", index=g.index)).fillna("SH")
        aum = g.loc[sh == "SH", "value"].sum()
        tot = g["value"].sum()
        put_w = g.loc[sh == "PUT", "value"].sum() / tot if tot > 0 else 0.0
        top10 = w.sort_values(ascending=False).head(10).sum()
        rows.append(dict(manager=mgr, period_date=period, filing_date=filing_date,
                         accession_number=accession,
                         submission_type=g["submission_type"].iloc[0],
                         aum=aum, n_holdings=int((w > 0).sum()), top10_weight=top10,
                         put_weight=put_w, active_share=_active_share(w, benchmark_weights),
                         bw=w))
    cols = [
        "manager", "period_date", "filing_date", "accession_number",
        "submission_type", "aum", "n_holdings", "top10_weight", "put_weight",
        "active_share", "bw",
    ]
    if not rows:
        return pd.DataFrame(columns=cols + ["prev_bw", "turnover", "hist_q"])

    chars = (pd.DataFrame(rows)
             .sort_values(["manager", "period_date", "filing_date", "accession_number"])
             .reset_index(drop=True))
    chars["hist_q"] = chars.groupby("manager")["period_date"].rank(method="dense").astype(int)

    # Known issue: prior-period turnover uses the latest version of that prior
    # period available in the full dataset. If a prior period is amended after a
    # decision date, this can leak a small amount of post-asof information into
    # the turnover screen. Fix later by computing turnover as-of each rebalance.
    turn = pd.Series(np.nan, index=chars.index, dtype=float)
    prev_bw = pd.Series([None] * len(chars), index=chars.index, dtype=object)
    for _, g in chars.groupby("manager", sort=False):
        prev_period_bw = None
        current_period = None
        current_period_latest_bw = None
        for idx, r in g.iterrows():
            if current_period is None or r["period_date"] != current_period:
                prev_period_bw = current_period_latest_bw
                current_period = r["period_date"]
            if prev_period_bw is not None:
                prev_bw.at[idx] = prev_period_bw
                alln = r["bw"].index.union(prev_period_bw.index)
                a = r["bw"].reindex(alln).fillna(0.0)
                b = prev_period_bw.reindex(alln).fillna(0.0)
                turn.loc[idx] = 1.0 - np.minimum(a, b).sum()
            current_period_latest_bw = r["bw"]
    chars["prev_bw"] = prev_bw
    chars["turnover"] = turn
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
        keep &= (latest["top10_weight"] >= cfg.min_top_n_weight) | (latest["n_holdings"] <= cfg.max_holdings)
    if cfg.use_hedge_filter:
        keep &= latest["put_weight"] <= cfg.hedge_put_max_weight
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
    return filter_universe_versions(_visible_manager_versions(chars, asof), cfg, value_scores)


def select_universe(chars, asof, cfg: UniverseConfig, value_scores=None) -> list[str]:
    selected = select_universe_versions(chars, asof, cfg, value_scores)
    if selected.empty:
        return []
    return selected["manager"].tolist()


# --------------------------------------------------------------------------- #
def _idea_scores(
    cur: pd.Series,
    prev: pd.Series | None,
    cfg: PortfolioConfig,
    active_benchmark_weights: pd.Series | None = None,
) -> pd.Series:
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
    elif cfg.idea_signal == "active_weight":
        if active_benchmark_weights is None or len(active_benchmark_weights) == 0:
            return pd.Series(dtype=float)
        if int((cur > 0).sum()) < cfg.min_active_weight_holdings:
            return pd.Series(dtype=float)
        bench = active_benchmark_weights / active_benchmark_weights.sum()
        s = (cur - bench.reindex(cur.index).fillna(0.0)).clip(lower=0)
    else:
        raise ValueError(cfg.idea_signal)
    return s[s > 0].sort_values(ascending=False).head(cfg.top_n_ideas)


def target_weights_from_versions(
    latest_versions: pd.DataFrame,
    cfg: PortfolioConfig,
    active_benchmark_weights: pd.Series | None = None,
) -> pd.Series:
    if latest_versions.empty:
        return pd.Series(dtype=float)
    if cfg.idea_signal == "active_weight" and active_benchmark_weights is None:
        active_benchmark_weights = _aggregate_book_weights(latest_versions["bw"])
    score: dict[str, float] = {}
    count: dict[str, int] = {}
    for cur_row in latest_versions.itertuples(index=False):
        cur = getattr(cur_row, "bw")
        prev = getattr(cur_row, "prev_bw", None)
        picks = _idea_scores(cur, prev, cfg, active_benchmark_weights)
        for tkr, wt in picks.items():
            score[tkr] = score.get(tkr, 0.0) + (float(wt) if cfg.consensus_weight else 1.0)
            count[tkr] = count.get(tkr, 0) + 1
    s = pd.Series(score)
    if s.empty:
        return s
    if cfg.min_consensus_funds > 1:
        keep = pd.Series(count).ge(cfg.min_consensus_funds)
        s = s[keep.reindex(s.index).fillna(False)]
    if s.empty:
        return s
    return _cap_weights(s, cfg.max_name_weight)


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
                            security_groups=None) -> tuple[pd.Series, float, list[str]]:
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

    eff_s = _cap_weights_with_groups(
        pd.Series(eff, dtype=float),
        cfg.portfolio.max_name_weight,
        cfg.portfolio.max_issuer_weight,
        security_groups,
    )
    alln = eff_s.index.union(cur.index)
    traded = 0.5 * (eff_s.reindex(alln).fillna(0) - cur.reindex(alln).fillna(0)).abs().sum()
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
                    security_groups=None) -> dict[str, pd.DataFrame]:
    """Return auditable rebalance summary, holdings, and manager-selection tables."""
    if chars is None:
        chars = manager_characteristics(holdings, benchmark_weights)
    months = prices.index.sort_values()
    rebal_months = set(_rebalance_months(holdings, prices))

    cur = pd.Series(dtype=float)
    last_in_tgt: dict[str, pd.Timestamp] = {}
    summary_rows: list[dict] = []
    holding_rows: list[dict] = []
    manager_rows: list[dict] = []

    for m in months:
        cur, _ = _apply_monthly_returns(cur, prices, m, cfg)
        if m not in rebal_months:
            continue

        latest_versions = visible_versions_cache.get(pd.Timestamp(m)) if visible_versions_cache is not None else None
        if latest_versions is None:
            latest_versions = _visible_manager_versions(chars, m)
        active_benchmark_weights = (
            _aggregate_book_weights(latest_versions["bw"])
            if cfg.portfolio.idea_signal == "active_weight"
            else None
        )
        selected_versions = filter_universe_versions(latest_versions, cfg.universe, value_scores)
        selected_managers = selected_versions["manager"].tolist() if not selected_versions.empty else []
        for manager in selected_managers:
            manager_rows.append({"rebalance_month": m.date().isoformat(), "manager": manager})

        tgt = target_weights_from_versions(selected_versions, cfg.portfolio, active_benchmark_weights)
        target_names_before_price_filter = int(len(tgt))
        priced_now = prices.columns[prices.loc[m].notna()]
        tgt = tgt[tgt.index.isin(priced_now)]
        if tgt.sum() <= 0:
            portfolio_stats = _portfolio_summary_stats(cur)
            issuer_stats = _issuer_exposure_summary(cur, security_groups)
            feasibility = _constraint_feasibility(cur, cfg, security_groups)
            summary_rows.append({
                "rebalance_month": m.date().isoformat(),
                "selected_managers": int(len(selected_versions)),
                "target_names": int(len(tgt)),
                "target_names_before_price_filter": target_names_before_price_filter,
                "effective_names": int(len(cur)),
                "carried_names": 0,
                "turnover_one_way": 0.0,
                "cost_bps": 0.0,
                **portfolio_stats,
                **issuer_stats,
                **feasibility,
                "traded_names": 0,
                "buy_names": 0,
                "sell_names": 0,
                "increased_names": 0,
                "decreased_names": 0,
                "top_holdings": "",
                "note": "no positive target weights",
            })
            continue

        eff, traded, carried = _apply_rebalance_target(cur, last_in_tgt, tgt, cfg, m, security_groups)
        cost_bps = traded * cfg.cost.bps_per_side
        portfolio_stats = _portfolio_summary_stats(eff)
        issuer_stats = _issuer_exposure_summary(eff, security_groups)
        feasibility = _constraint_feasibility(eff, cfg, security_groups)
        trade_stats = _trade_summary_stats(cur, eff)
        top = eff.sort_values(ascending=False).head(12)
        summary_rows.append({
            "rebalance_month": m.date().isoformat(),
            "selected_managers": int(len(selected_versions)),
            "target_names": int(len(tgt)),
            "target_names_before_price_filter": target_names_before_price_filter,
            "effective_names": int(len(eff)),
            "carried_names": int(len(carried)),
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
        groups = _security_groups_for(eff.index, security_groups).reindex(eff.index)
        for rank, (ticker, weight) in enumerate(eff.sort_values(ascending=False).items(), start=1):
            holding_rows.append({
                "rebalance_month": m.date().isoformat(),
                "rank": int(rank),
                "ticker": ticker,
                "issuer_group": groups.loc[ticker],
                "weight": float(weight),
                "is_carried": bool(ticker in carried_set),
            })
        cur = eff

    return {
        "summary": pd.DataFrame(summary_rows),
        "holdings": pd.DataFrame(holding_rows),
        "managers": pd.DataFrame(manager_rows),
    }


# --------------------------------------------------------------------------- #
def run_backtest(holdings, prices, cfg: BacktestConfig,
                 value_scores=None, benchmark_weights=None, chars=None,
                 visible_versions_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
                 security_groups=None,
                 progress_label: str | None = None,
                 progress_every: int = 10) -> pd.Series:
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

    for m in months:
        cur, net.loc[m] = _apply_monthly_returns(cur, prices, m, cfg)
        if m in rebal_months:
            rebal_no += 1
            latest_versions = visible_versions_cache.get(pd.Timestamp(m)) if visible_versions_cache is not None else None
            if latest_versions is None:
                latest_versions = _visible_manager_versions(chars, m)
            active_benchmark_weights = (
                _aggregate_book_weights(latest_versions["bw"])
                if cfg.portfolio.idea_signal == "active_weight"
                else None
            )
            selected_versions = filter_universe_versions(latest_versions, cfg.universe, value_scores)
            tgt = target_weights_from_versions(selected_versions, cfg.portfolio, active_benchmark_weights)
            target_names_before_price_filter = len(tgt)
            priced_now = prices.columns[prices.loc[m].notna()]
            tgt = tgt[tgt.index.isin(priced_now)]
            if tgt.sum() > 0:
                eff, traded, _ = _apply_rebalance_target(cur, last_in_tgt, tgt, cfg, m, security_groups)
                net.loc[m] -= traded * cfg.cost.bps_per_side / 1e4
                cur = eff
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
    return net


def build_rebalance_selection_cache(
    holdings,
    prices,
    cfg: BacktestConfig,
    value_scores=None,
    benchmark_weights=None,
    chars=None,
    visible_versions_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
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
        selected_by_month[month] = filter_universe_versions(latest_versions, cfg.universe, value_scores)
    return selected_by_month


def build_active_benchmark_weights_cache(
    holdings,
    prices,
    benchmark_weights=None,
    chars=None,
    visible_versions_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> dict[pd.Timestamp, pd.Series]:
    """Precompute PIT aggregate 13F book weights used by active_weight ideas."""
    if chars is None:
        chars = manager_characteristics(holdings, benchmark_weights)
    active_by_month: dict[pd.Timestamp, pd.Series] = {}
    for m in _rebalance_months(holdings, prices):
        month = pd.Timestamp(m)
        latest_versions = visible_versions_cache.get(month) if visible_versions_cache is not None else None
        if latest_versions is None:
            latest_versions = _visible_manager_versions(chars, month)
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

    for m in months:
        cur, net.loc[m] = _apply_monthly_returns(cur, prices, m, cfg)
        month = pd.Timestamp(m)
        if month in rebal_months:
            rebal_no += 1
            selected_versions = selected_versions_by_month.get(month, pd.DataFrame())
            active_benchmark_weights = None
            if cfg.portfolio.idea_signal == "active_weight":
                if active_benchmark_weights_by_month is not None:
                    active_benchmark_weights = active_benchmark_weights_by_month.get(month)
                if active_benchmark_weights is None and not selected_versions.empty:
                    active_benchmark_weights = _aggregate_book_weights(selected_versions["bw"])
            tgt = target_weights_from_versions(selected_versions, cfg.portfolio, active_benchmark_weights)
            target_names_before_price_filter = len(tgt)
            priced_now = prices.columns[prices.loc[m].notna()]
            tgt = tgt[tgt.index.isin(priced_now)]
            if tgt.sum() > 0:
                eff, traded, _ = _apply_rebalance_target(cur, last_in_tgt, tgt, cfg, m, security_groups)
                net.loc[m] -= traded * cfg.cost.bps_per_side / 1e4
                cur = eff
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
        "sharpe": (y_basic.mean() / y_basic.std()) * np.sqrt(12) if y_basic.std() else np.nan,
    }
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
                chars=None, visible_versions_cache=None, security_groups=None, verbose: bool = False):
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
            progress_label=f"marginal-ir -{t}" if verbose else None,
        )
        m = _active_ir_metric(ret, benchmark)
        if verbose:
            print(f"    done -{t} in {time.perf_counter() - t0:.1f}s metric={_fmt_metric(m)}")
        rows.append(dict(filter=f"-{t}", metric=m, delta=(bm - m)))
    return pd.DataFrame(rows)
