"""
Parameter sweep harness — treats the filters/thresholds as the hypotheses under test.

Two outputs, kept strictly separate:
  1. grid_eval()  -> IN-SAMPLE metric over the whole grid. Use ONLY for the
     robustness/"plateau" heatmap. Never quote its best cell as performance.
  2. walk_forward() -> the honest OOS track: on each TRAIN window pick the best
     config, lock it, measure on the next TEST window, roll. Concatenate test
     windows. Then deflated_sharpe() haircuts the result for how many configs we
     tried, so a winner that's just the max of N noisy trials gets exposed.
"""
from __future__ import annotations
import itertools, math
import numpy as np
import pandas as pd
from dataclasses import replace
from scipy.stats import norm
try:
    from .engine import BacktestConfig, run_backtest, attribution, manager_characteristics
except ImportError:
    from engine import BacktestConfig, run_backtest, attribution, manager_characteristics


def iter_configs(base: BacktestConfig, axes: dict) -> list[tuple[dict, BacktestConfig]]:
    """axes: {('universe'|'portfolio', field): [values]} -> list of (label, cfg)."""
    keys = list(axes.keys())
    out = []
    for combo in itertools.product(*[axes[k] for k in keys]):
        u, p = dict(), dict()
        label = {}
        for (scope, field), val in zip(keys, combo):
            (u if scope == "universe" else p)[field] = val
            label[field] = val
        cfg = replace(base, universe=replace(base.universe, **u), portfolio=replace(base.portfolio, **p))
        out.append((label, cfg))
    return out


def _periodic_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return r.mean() / r.std() if r.std() else 0.0


def grid_eval(holdings, prices, factors, base, axes, benchmark=None,
              value_scores=None, benchmark_weights=None, metric="sharpe") -> pd.DataFrame:
    ch = manager_characteristics(holdings, benchmark_weights)
    rows = []
    for label, cfg in iter_configs(base, axes):
        ret = run_backtest(holdings, prices, cfg, value_scores, benchmark_weights, ch)
        att = attribution(ret, factors, benchmark)
        rows.append({**label, "sharpe": att.get("sharpe"),
                     "ann_alpha": att.get("ann_alpha"), "alpha_t": att.get("alpha_t"),
                     "ir": att.get("ir_vs_benchmark")})
    return pd.DataFrame(rows)


def walk_forward(holdings, prices, factors, base, axes, benchmark=None,
                 value_scores=None, benchmark_weights=None,
                 train_m=36, test_m=12, select_on="sharpe"):
    """Rolling OOS. Returns (oos_returns, fold_log, n_trials)."""
    configs = iter_configs(base, axes)
    n_trials = len(configs)
    ch = manager_characteristics(holdings, benchmark_weights)
    months = prices.index.sort_values()
    oos = []
    log = []
    start = 0
    while start + train_m + test_m <= len(months):
        tr = months[start:start + train_m]
        te = months[start + train_m:start + train_m + test_m]
        best, best_score, best_lbl = None, -np.inf, None
        for label, cfg in configs:
            ret = run_backtest(holdings, prices.loc[tr], cfg, value_scores, benchmark_weights, ch)
            sc = _periodic_sharpe(ret)
            if sc > best_score:
                best, best_score, best_lbl = cfg, sc, label
        te_ret = run_backtest(holdings, prices.loc[months[:start + train_m + test_m]],
                              best, value_scores, benchmark_weights, ch).reindex(te)
        oos.append(te_ret)
        log.append({"test_start": te[0], "test_end": te[-1], "train_sharpe": best_score, **best_lbl})
        start += test_m
    oos_ret = pd.concat(oos).sort_index() if oos else pd.Series(dtype=float)
    return oos_ret, pd.DataFrame(log), n_trials


def deflated_sharpe(returns: pd.Series, n_trials: int, sr_variance: float | None = None) -> dict:
    """
    Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.
    sr_variance = variance of the (periodic) Sharpes across the trials you ran;
    if None, a conservative default is used. Returns periodic SR, the expected max
    SR under the null of N trials, and DSR = P(true SR > 0 | selection of best of N).
    """
    r = returns.dropna()
    T = len(r)
    if T < 12:
        return {"note": "insufficient OOS", "T": T, "n_trials": n_trials}
    sr = r.mean() / r.std() if r.std() else 0.0
    g = r.skew()
    k = r.kurtosis() + 3.0
    if sr_variance is None or sr_variance <= 0:
        sr_variance = (1.0 / T)            # conservative fallback
    emc = 0.5772156649
    e = math.e
    sr0 = math.sqrt(sr_variance) * ((1 - emc) * norm.ppf(1 - 1.0 / n_trials)
                                    + emc * norm.ppf(1 - 1.0 / (n_trials * e)))
    denom = math.sqrt(max(1e-9, 1 - g * sr + (k - 1) / 4.0 * sr ** 2))
    dsr = norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom)
    return {"periodic_SR": sr, "ann_SR": sr * math.sqrt(12), "expected_max_SR_null": sr0,
            "n_trials": n_trials, "T": T, "DSR": dsr}
