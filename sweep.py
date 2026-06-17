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
import time
import numpy as np
import pandas as pd
from dataclasses import fields, replace
from scipy.stats import norm
try:
    from .engine import (
        BacktestConfig,
        attribution,
        build_active_benchmark_weights_cache,
        build_rebalance_selection_cache,
        build_visible_versions_cache,
        manager_characteristics,
        run_backtest,
        run_backtest_from_selection_cache,
    )
except ImportError:
    from engine import (
        BacktestConfig,
        attribution,
        build_active_benchmark_weights_cache,
        build_rebalance_selection_cache,
        build_visible_versions_cache,
        manager_characteristics,
        run_backtest,
        run_backtest_from_selection_cache,
    )


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


def _label_key(label: dict) -> tuple:
    return tuple(sorted(label.items()))


def _universe_key(cfg) -> tuple:
    return tuple((f.name, getattr(cfg, f.name)) for f in fields(cfg))


def _periodic_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return r.mean() / r.std() if r.std() else 0.0


def active_return_stream(returns: pd.Series, benchmark: pd.Series | None) -> pd.Series:
    if benchmark is None:
        return returns.dropna()
    active = pd.concat([returns.rename("ret"), benchmark.reindex(returns.index).rename("bench")], axis=1)
    active = active.replace([np.inf, -np.inf], np.nan).dropna()
    return active["ret"] - active["bench"]


def _score_series(returns: pd.Series, benchmark: pd.Series | None, metric: str) -> pd.Series:
    if metric in {"active", "active_sharpe", "ir", "ir_vs_benchmark"}:
        return active_return_stream(returns, benchmark)
    return returns.dropna()


def _fmt_metric(value) -> str:
    return f"{value:.4g}" if isinstance(value, (int, float, np.floating)) and np.isfinite(value) else str(value)


def _basic_sharpe(returns: pd.Series) -> float:
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return _periodic_sharpe(r) * math.sqrt(12)


def _needs_factor_attribution(metric: str) -> bool:
    return metric not in {"active", "active_sharpe"}


def grid_eval(holdings, prices, factors, base, axes, benchmark=None,
              value_scores=None, benchmark_weights=None, metric="sharpe",
              chars=None, visible_versions_cache=None, verbose: bool = False,
              include_returns: bool = False, security_groups=None,
              include_factor_metrics: bool | None = None,
              use_selection_cache: bool = True) -> pd.DataFrame:
    ch = chars if chars is not None else manager_characteristics(holdings, benchmark_weights)
    visible_cache = visible_versions_cache if visible_versions_cache is not None else build_visible_versions_cache(ch, prices.index)
    run_attribution = _needs_factor_attribution(metric) if include_factor_metrics is None else bool(include_factor_metrics)
    selection_caches: dict[tuple, dict[pd.Timestamp, pd.DataFrame]] = {}
    active_benchmark_cache: dict[pd.Timestamp, pd.Series] | None = None
    rows = []
    returns_by_config: dict[tuple, pd.Series] = {}
    configs = iter_configs(base, axes)
    total = len(configs)
    for i, (label, cfg) in enumerate(configs, start=1):
        if verbose:
            print(f"  grid {i}/{total} running {label}")
            t0 = time.perf_counter()
        if use_selection_cache:
            ukey = _universe_key(cfg.universe)
            if ukey not in selection_caches:
                selection_caches[ukey] = build_rebalance_selection_cache(
                    holdings,
                    prices,
                    cfg,
                    value_scores,
                    benchmark_weights,
                    ch,
                    visible_cache,
                )
            if cfg.portfolio.idea_signal == "active_weight" and active_benchmark_cache is None:
                active_benchmark_cache = build_active_benchmark_weights_cache(
                    holdings,
                    prices,
                    benchmark_weights,
                    ch,
                    visible_cache,
                )
            ret = run_backtest_from_selection_cache(
                prices,
                cfg,
                selection_caches[ukey],
                active_benchmark_cache,
                security_groups,
            )
        else:
            ret = run_backtest(holdings, prices, cfg, value_scores, benchmark_weights, ch, visible_cache, security_groups)
        if include_returns:
            returns_by_config[_label_key(label)] = ret
        active_sharpe = _periodic_sharpe(active_return_stream(ret, benchmark)) * math.sqrt(12)
        if run_attribution:
            att = attribution(ret, factors, benchmark)
            row_metrics = {
                "sharpe": att.get("sharpe"),
                "ann_alpha": att.get("ann_alpha"),
                "alpha_t": att.get("alpha_t"),
                "ir": att.get("ir_vs_benchmark"),
            }
        else:
            row_metrics = {
                "sharpe": _basic_sharpe(ret),
                "ann_alpha": np.nan,
                "alpha_t": np.nan,
                "ir": active_sharpe,
            }
        if verbose:
            chosen = active_sharpe if metric in {"active", "active_sharpe"} else row_metrics.get(metric, row_metrics.get("sharpe"))
            print(f"    done grid {i}/{total} in {time.perf_counter() - t0:.1f}s {metric}={_fmt_metric(chosen)}")
        rows.append({**label, **row_metrics, "active_sharpe": active_sharpe})
    out = pd.DataFrame(rows)
    if include_returns:
        out.attrs["returns_by_config"] = returns_by_config
    return out


def walk_forward(holdings, prices, factors, base, axes, benchmark=None,
                 value_scores=None, benchmark_weights=None,
                 train_m=36, test_m=12, select_on="sharpe",
                 chars=None, visible_versions_cache=None, verbose: bool = False,
                 precomputed_returns: dict[tuple, pd.Series] | None = None,
                 security_groups=None):
    """Rolling OOS. Returns (oos_returns, fold_log, n_trials)."""
    configs = iter_configs(base, axes)
    n_trials = len(configs)
    ch = chars if chars is not None else manager_characteristics(holdings, benchmark_weights)
    months = prices.index.sort_values()
    visible_cache = visible_versions_cache if visible_versions_cache is not None else build_visible_versions_cache(ch, months)
    oos = []
    log = []
    start = 0
    n_folds = max(0, (len(months) - train_m) // test_m)
    fold_no = 0
    while start + train_m + test_m <= len(months):
        fold_no += 1
        tr = months[start:start + train_m]
        te = months[start + train_m:start + train_m + test_m]
        if verbose:
            print(
                f"  walk-forward fold {fold_no}/{n_folds}: "
                f"train {tr[0].date()}..{tr[-1].date()}, test {te[0].date()}..{te[-1].date()}"
            )
        best, best_score, best_lbl = None, -np.inf, None
        for i, (label, cfg) in enumerate(configs, start=1):
            if verbose:
                print(f"    train config {i}/{n_trials} {label}")
                t0 = time.perf_counter()
            if precomputed_returns is not None and _label_key(label) in precomputed_returns:
                ret = precomputed_returns[_label_key(label)].reindex(tr)
            else:
                ret = run_backtest(holdings, prices.loc[tr], cfg, value_scores, benchmark_weights, ch, visible_cache, security_groups)
            sc = _periodic_sharpe(_score_series(ret, benchmark.reindex(tr) if benchmark is not None else None, select_on))
            if verbose:
                print(f"      done in {time.perf_counter() - t0:.1f}s train_{select_on}={_fmt_metric(sc)}")
            if sc > best_score:
                best, best_score, best_lbl = cfg, sc, label
        if verbose:
            print(f"    selected {best_lbl} train_{select_on}={_fmt_metric(best_score)}; running test window")
            t0 = time.perf_counter()
        if precomputed_returns is not None and _label_key(best_lbl) in precomputed_returns:
            te_ret = precomputed_returns[_label_key(best_lbl)].reindex(te)
        else:
            te_ret = run_backtest(holdings, prices.loc[months[:start + train_m + test_m]],
                                  best, value_scores, benchmark_weights, ch, visible_cache, security_groups).reindex(te)
        if verbose:
            print(f"    done test fold {fold_no}/{n_folds} in {time.perf_counter() - t0:.1f}s")
        oos.append(te_ret)
        log.append({"test_start": te[0], "test_end": te[-1], f"train_{select_on}": best_score, **best_lbl})
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
