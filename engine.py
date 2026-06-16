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
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
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
    idea_signal: str = "level"              # "level" | "change" | "initiation"
    consensus_weight: bool = True
    min_consensus_funds: int = 1            # drop names held by < this many in-universe funds
    max_name_weight: float = 0.05
    holding_horizon_q: int = 0              # 0 = full rebalance; N = carry N extra quarters


@dataclass
class CostConfig:
    bps_per_side: float = 15.0


@dataclass
class BacktestConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    missing_price_policy: str = "raise"     # "raise" | "zero"


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
        return pd.DataFrame(columns=cols + ["turnover", "hist_q"])

    chars = (pd.DataFrame(rows)
             .sort_values(["manager", "period_date", "filing_date", "accession_number"])
             .reset_index(drop=True))
    chars["hist_q"] = chars.groupby("manager")["period_date"].rank(method="dense").astype(int)

    turn = pd.Series(np.nan, index=chars.index, dtype=float)
    for _, g in chars.groupby("manager", sort=False):
        prev_period_bw = None
        current_period = None
        current_period_latest_bw = None
        for idx, r in g.iterrows():
            if current_period is None or r["period_date"] != current_period:
                prev_period_bw = current_period_latest_bw
                current_period = r["period_date"]
            if prev_period_bw is not None:
                alln = r["bw"].index.union(prev_period_bw.index)
                a = r["bw"].reindex(alln).fillna(0.0)
                b = prev_period_bw.reindex(alln).fillna(0.0)
                turn.loc[idx] = 1.0 - np.minimum(a, b).sum()
            current_period_latest_bw = r["bw"]
    chars["turnover"] = turn
    return chars


def _book_value_pctl(w: pd.Series, vscores: pd.Series | None) -> float:
    if vscores is None:
        return np.nan
    common = w.index.intersection(vscores.dropna().index)
    if len(common) == 0:
        return np.nan
    return float((w[common] * vscores[common].rank(pct=True)).sum() / w[common].sum())


# --------------------------------------------------------------------------- #
def select_universe(chars, asof, cfg: UniverseConfig, value_scores=None) -> list[str]:
    latest = _visible_manager_versions(chars, asof).set_index("manager")
    if latest.empty:
        return []
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
        vp = {m: _book_value_pctl(r["bw"], value_scores.loc[r["period_date"]]
                                  if r["period_date"] in value_scores.index else None)
              for m, r in latest.iterrows()}
        keep &= pd.Series(vp).ge(cfg.value_tilt_min_pctl).reindex(keep.index).fillna(False)
    return list(latest.index[keep.values])


# --------------------------------------------------------------------------- #
def _idea_scores(cur: pd.Series, prev: pd.Series | None, cfg: PortfolioConfig) -> pd.Series:
    if cfg.idea_signal == "level" or prev is None:
        s = cur
    elif cfg.idea_signal == "change":
        alln = cur.index.union(prev.index)
        s = (cur.reindex(alln).fillna(0.0) - prev.reindex(alln).fillna(0.0)).clip(lower=0)
    elif cfg.idea_signal == "initiation":
        new = cur.index.difference(prev.index)
        s = cur.reindex(new).fillna(0.0)
    else:
        raise ValueError(cfg.idea_signal)
    return s[s > 0].sort_values(ascending=False).head(cfg.top_n_ideas)


def target_weights(chars, managers, asof, cfg: PortfolioConfig) -> pd.Series:
    known = chars[(chars["filing_date"] <= asof) & (chars["manager"].isin(managers))]
    if known.empty:
        return pd.Series(dtype=float)
    score: dict[str, float] = {}
    count: dict[str, int] = {}
    for mgr, g in known.groupby("manager"):
        g = g.sort_values(["period_date", "filing_date", "accession_number"])
        cur_row = g.iloc[-1]
        cur = cur_row["bw"]
        prior = g[g["period_date"] < cur_row["period_date"]]
        prev = prior.iloc[-1]["bw"] if len(prior) else None
        picks = _idea_scores(cur, prev, cfg)
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
    s = (s / s.sum()).clip(upper=cfg.max_name_weight)
    return s / s.sum()


# --------------------------------------------------------------------------- #
def run_backtest(holdings, prices, cfg: BacktestConfig,
                 value_scores=None, benchmark_weights=None, chars=None) -> pd.Series:
    if chars is None:
        chars = manager_characteristics(holdings, benchmark_weights)
    months = prices.index.sort_values()
    fil = holdings["filing_date"].drop_duplicates().sort_values()
    rebal_months = sorted({months[months >= f][0] for f in fil if (months >= f).any()})

    cur = pd.Series(dtype=float)
    age: dict[str, int] = {}
    net = pd.Series(0.0, index=months)
    H = cfg.portfolio.holding_horizon_q

    for m in months:
        if len(cur):
            r = prices.loc[m, cur.index]
            missing = r[r.isna()].index.tolist()
            if missing:
                if cfg.missing_price_policy == "raise":
                    sample = ", ".join(map(str, missing[:10]))
                    raise ValueError(
                        f"Missing returns for {len(missing)} held names on {m.date()}: {sample}"
                    )
                if cfg.missing_price_policy != "zero":
                    raise ValueError(f"Unknown missing_price_policy={cfg.missing_price_policy!r}")
                r = r.fillna(0.0)
            net.loc[m] = float((cur * r).sum())
            grown = cur * (1.0 + r)
            cur = grown / grown.sum() if grown.sum() > 0 else cur
        if m in rebal_months:
            u = select_universe(chars, m, cfg.universe, value_scores)
            tgt = target_weights(chars, u, m, cfg.portfolio)
            tgt = tgt[tgt.index.isin(prices.columns)]
            if tgt.sum() > 0:
                tgt = tgt / tgt.sum()
                for n in list(age):                # age everything held
                    age[n] += 1
                for n in tgt.index:                # reset names back in target
                    age[n] = 0
                eff = {n: float(w) for n, w in tgt.items()}
                if H > 0:                          # carry recent names (patient capital)
                    for n, a in list(age.items()):
                        if 0 < a <= H and n not in eff and n in cur.index:
                            eff[n] = float(cur[n])
                for n in [n for n, a in age.items() if a > H]:
                    age.pop(n, None)
                eff = pd.Series(eff)
                eff = (eff.clip(upper=cfg.portfolio.max_name_weight))
                eff = eff / eff.sum()
                alln = eff.index.union(cur.index)
                traded = 0.5 * (eff.reindex(alln).fillna(0) - cur.reindex(alln).fillna(0)).abs().sum()
                net.loc[m] -= traded * cfg.cost.bps_per_side / 1e4
                cur = eff
    return net


# --------------------------------------------------------------------------- #
def attribution(port_ret, factors, benchmark=None,
                factor_cols=("MKT", "SMB", "HML", "RMW", "CMA", "MOM")) -> dict:
    factors = factors if factors is not None else pd.DataFrame(index=port_ret.index)
    df = pd.concat([port_ret.rename("ret"), factors], axis=1).dropna(subset=["ret"])
    if len(df) < 12:
        return {"n_months": len(df), "note": "insufficient overlap"}
    if "RF" in df:
        y = df["ret"] - df["RF"]
    else:
        y = df["ret"]
    out = {
        "n_months": len(df),
        "ann_return": (1 + df["ret"]).prod() ** (12 / len(df)) - 1,
        "ann_vol": df["ret"].std() * np.sqrt(12),
        "sharpe": (y.mean() / y.std()) * np.sqrt(12) if y.std() else np.nan,
    }
    missing_factor_cols = [c for c in factor_cols if c not in df.columns]
    if "RF" not in df.columns or missing_factor_cols:
        out["note"] = "factor regression unavailable"
        if benchmark is not None:
            active = (df["ret"] - benchmark.reindex(df.index)).dropna()
            out["ir_vs_benchmark"] = (active.mean() / active.std()) * np.sqrt(12) if active.std() else np.nan
        return out

    res = sm.OLS(y, sm.add_constant(df[list(factor_cols)])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    out.update({
        "ann_alpha": (1 + res.params["const"]) ** 12 - 1,
        "alpha_t": res.tvalues["const"],
        "betas": {c: res.params[c] for c in factor_cols},
    })
    if benchmark is not None:
        active = (df["ret"] - benchmark.reindex(df.index)).dropna()
        out["ir_vs_benchmark"] = (active.mean() / active.std()) * np.sqrt(12) if active.std() else np.nan
    return out


def marginal_ir(holdings, prices, factors, cfg, benchmark=None, value_scores=None, benchmark_weights=None):
    ch = manager_characteristics(holdings, benchmark_weights)
    base = attribution(run_backtest(holdings, prices, cfg, value_scores, benchmark_weights, ch), factors, benchmark)
    bm = base.get("ir_vs_benchmark", base.get("sharpe"))
    rows = [dict(filter="(full stack)", metric=bm, delta=0.0)]
    for t in ["use_size_band", "use_concentration", "use_low_turnover", "use_hedge_filter", "use_value_tilt", "use_active_share"]:
        if not getattr(cfg.universe, t):
            continue
        cfg2 = replace(cfg, universe=replace(cfg.universe, **{t: False}))
        att = attribution(run_backtest(holdings, prices, cfg2, value_scores, benchmark_weights, ch), factors, benchmark)
        m = att.get("ir_vs_benchmark", att.get("sharpe"))
        rows.append(dict(filter=f"-{t}", metric=m, delta=(bm - m)))
    return pd.DataFrame(rows)
