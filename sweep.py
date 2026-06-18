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
from pathlib import Path
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
        _needs_active_benchmark_weights,
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
        _needs_active_benchmark_weights,
        run_backtest,
        run_backtest_from_selection_cache,
    )


def iter_configs(base: BacktestConfig, axes: dict) -> list[tuple[dict, BacktestConfig]]:
    """axes: {('universe'|'portfolio'|'backtest', field): [values]} -> configs."""
    keys = list(axes.keys())
    out = []
    for combo in itertools.product(*[axes[k] for k in keys]):
        u, p, b = dict(), dict(), dict()
        label = {}
        for (scope, field), val in zip(keys, combo):
            if scope == "universe" and field == "aum_band":
                band_label, min_aum, max_aum = val
                u["min_aum"] = min_aum
                u["max_aum"] = max_aum
                label["aum_band"] = band_label
            else:
                if scope == "universe":
                    u[field] = val
                elif scope == "portfolio":
                    p[field] = val
                elif scope == "backtest":
                    b[field] = val
                else:
                    raise ValueError(f"Unknown sweep scope={scope!r}")
                label[field] = val
        cfg = replace(base, universe=replace(base.universe, **u), portfolio=replace(base.portfolio, **p), **b)
        out.append((label, cfg))
    return out


def _label_key(label: dict) -> tuple:
    return tuple(sorted(label.items()))


def _config_id(label: dict) -> str:
    return "|".join(f"{key}={value}" for key, value in sorted(label.items()))


def _universe_key(cfg) -> tuple:
    return tuple((f.name, getattr(cfg, f.name)) for f in fields(cfg))


def _selection_key(cfg: BacktestConfig) -> tuple:
    return (
        ("manager_filter_mode", cfg.manager_filter_mode),
        ("active_benchmark_source", cfg.active_benchmark_source),
        ("missing_price_policy", cfg.missing_price_policy),
        *_universe_key(cfg.universe),
    )


def _manager_filter_kwargs(manager_classification=None, manager_overrides=None) -> dict:
    out = {}
    if manager_classification is not None:
        out["manager_classification"] = manager_classification
    if manager_overrides is not None:
        out["manager_overrides"] = manager_overrides
    return out


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


def _without_attrs(series: pd.Series) -> pd.Series:
    out = series.copy()
    out.attrs = {}
    return out


def _basic_sharpe(returns: pd.Series) -> float:
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return _periodic_sharpe(r) * math.sqrt(12)


def _max_drawdown(returns: pd.Series) -> float:
    r = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if r.empty:
        return np.nan
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())


def _return_metrics(returns: pd.Series) -> dict:
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {
            "total_return": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "max_drawdown": np.nan,
        }
    return {
        "total_return": float((1 + r).prod() - 1),
        "ann_return": float((1 + r).prod() ** (12 / len(r)) - 1),
        "ann_vol": float(r.std() * math.sqrt(12)),
        "max_drawdown": _max_drawdown(r),
    }


def _rebalance_validity_metrics(summary: pd.DataFrame, cfg: BacktestConfig) -> dict:
    if summary is None or summary.empty:
        return {
            "valid_config": False,
            "invested_month_frac": 0.0,
            "valid_rebalance_frac": 0.0,
            "invalid_rebalance_frac": 1.0,
            "avg_effective_names": 0.0,
            "avg_target_names": 0.0,
            "avg_max_weight": 0.0,
            "name_cap_feasible_ratio": 0.0,
            "issuer_cap_feasible_ratio": 0.0,
            "zero_contributor_manager_frac": np.nan,
        }

    def mean_col(col: str, default=np.nan) -> float:
        if col not in summary:
            return float(default)
        s = pd.to_numeric(summary[col], errors="coerce").dropna()
        return float(s.mean()) if not s.empty else float(default)

    invested = pd.to_numeric(summary.get("effective_names", pd.Series(dtype=float)), errors="coerce").fillna(0).gt(0)
    valid = summary.get("valid_rebalance", pd.Series(False, index=summary.index)).astype(bool)
    name_cap = summary.get("name_cap_feasible", pd.Series(False, index=summary.index)).astype(bool)
    issuer_cap = summary.get("issuer_cap_feasible", pd.Series(False, index=summary.index)).astype(bool)
    selected = pd.to_numeric(summary.get("selected_managers", pd.Series(dtype=float)), errors="coerce")
    zero = pd.to_numeric(summary.get("zero_contributor_managers", pd.Series(dtype=float)), errors="coerce")
    selected_sum = float(selected.fillna(0).sum())
    zero_frac = float(zero.fillna(0).sum() / selected_sum) if selected_sum > 0 else np.nan
    invested_frac = float(invested.mean()) if len(invested) else 0.0
    valid_frac = float(valid.mean()) if len(valid) else 0.0
    name_cap_ratio = float(name_cap.mean()) if len(name_cap) else 0.0
    issuer_cap_ratio = float(issuer_cap.mean()) if len(issuer_cap) else 0.0
    min_names = int(cfg.portfolio.min_portfolio_names or 0)
    avg_effective = mean_col("effective_names", 0.0)
    valid_config = (
        invested_frac >= 0.80
        and valid_frac >= 0.80
        and name_cap_ratio >= 0.80
        and issuer_cap_ratio >= 0.80
        and (min_names <= 0 or avg_effective >= min_names)
    )
    return {
        "valid_config": bool(valid_config),
        "invested_month_frac": invested_frac,
        "valid_rebalance_frac": valid_frac,
        "invalid_rebalance_frac": 1.0 - valid_frac,
        "avg_selected_managers": mean_col("selected_managers", 0.0),
        "avg_visible_managers": mean_col("visible_managers", 0.0),
        "avg_stale_managers_dropped": mean_col("stale_managers_dropped", 0.0),
        "avg_stale_filing_managers": mean_col("stale_filing_managers", 0.0),
        "avg_stale_period_managers": mean_col("stale_period_managers", 0.0),
        "avg_active_eligible_managers": mean_col("active_eligible_managers", 0.0),
        "avg_zero_contributor_managers": mean_col("zero_contributor_managers", 0.0),
        "zero_contributor_manager_frac": zero_frac,
        "avg_raw_idea_names": mean_col("raw_idea_names", 0.0),
        "avg_consensus_idea_names": mean_col("consensus_idea_names", 0.0),
        "avg_effective_names": avg_effective,
        "avg_target_names": mean_col("target_names", 0.0),
        "avg_max_weight": mean_col("max_weight", 0.0),
        "avg_max_issuer_weight": mean_col("max_issuer_weight", 0.0),
        "name_cap_feasible_ratio": name_cap_ratio,
        "issuer_cap_feasible_ratio": issuer_cap_ratio,
    }


def _needs_factor_attribution(metric: str) -> bool:
    return metric not in {"active", "active_sharpe"}


def _write_grid_checkpoint(rows: list[dict], checkpoint_dir, returns_by_config_id: dict[str, pd.Series] | None = None) -> None:
    if checkpoint_dir is None:
        return
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "sweep_grid_partial.csv", index=False)
    if returns_by_config_id:
        ret_rows = []
        for config_id, ret in returns_by_config_id.items():
            for date, value in ret.replace([np.inf, -np.inf], np.nan).dropna().items():
                ret_rows.append({
                    "config_id": config_id,
                    "date": pd.Timestamp(date).date().isoformat(),
                    "return": float(value),
                })
        pd.DataFrame(ret_rows).to_csv(path / "sweep_returns_partial.csv", index=False)


def grid_eval(holdings, prices, factors, base, axes, benchmark=None,
              value_scores=None, benchmark_weights=None, metric="sharpe",
              chars=None, visible_versions_cache=None, verbose: bool = False,
              include_returns: bool = False, security_groups=None,
              active_benchmark_weights_by_month=None,
              manager_classification=None,
              manager_overrides=None,
              include_factor_metrics: bool | None = None,
              use_selection_cache: bool = True,
              checkpoint_dir=None,
              checkpoint_every: int = 5) -> pd.DataFrame:
    ch = chars if chars is not None else manager_characteristics(holdings, benchmark_weights)
    visible_cache = visible_versions_cache if visible_versions_cache is not None else build_visible_versions_cache(ch, prices.index)
    run_attribution = _needs_factor_attribution(metric) if include_factor_metrics is None else bool(include_factor_metrics)
    selection_caches: dict[tuple, dict[pd.Timestamp, pd.DataFrame]] = {}
    active_benchmark_caches: dict[tuple, dict[pd.Timestamp, pd.Series]] = {}
    rows = []
    returns_by_config: dict[tuple, pd.Series] = {}
    returns_by_config_id: dict[str, pd.Series] = {}
    configs = iter_configs(base, axes)
    total = len(configs)
    for i, (label, cfg) in enumerate(configs, start=1):
        if verbose:
            print(f"  grid {i}/{total} running {label}")
            t0 = time.perf_counter()
        if use_selection_cache:
            ukey = _selection_key(cfg)
            if ukey not in selection_caches:
                selection_caches[ukey] = build_rebalance_selection_cache(
                    holdings,
                    prices,
                    cfg,
                    value_scores=value_scores,
                    benchmark_weights=benchmark_weights,
                    chars=ch,
                    visible_versions_cache=visible_cache,
                    **_manager_filter_kwargs(manager_classification, manager_overrides),
                )
            active_benchmark_cache = None
            if _needs_active_benchmark_weights(cfg.portfolio.idea_signal):
                active_key = (
                    cfg.active_benchmark_source,
                    cfg.universe.max_stale_filing_months,
                    cfg.universe.max_stale_period_months,
                )
                if cfg.active_benchmark_source not in {"visible_13f_aggregate", "13f_aggregate"}:
                    active_benchmark_cache = active_benchmark_weights_by_month
                else:
                    if active_key not in active_benchmark_caches:
                        active_benchmark_caches[active_key] = build_active_benchmark_weights_cache(
                            holdings,
                            prices,
                            benchmark_weights,
                            ch,
                            visible_cache,
                            cfg,
                            active_benchmark_weights_by_month,
                        )
                    active_benchmark_cache = active_benchmark_caches[active_key]
            ret = run_backtest_from_selection_cache(
                prices,
                cfg,
                selection_caches[ukey],
                active_benchmark_cache,
                security_groups,
                capture_rebalance=True,
            )
        else:
            ret = run_backtest(
                holdings,
                prices,
                cfg,
                value_scores=value_scores,
                benchmark_weights=benchmark_weights,
                chars=ch,
                visible_versions_cache=visible_cache,
                security_groups=security_groups,
                active_benchmark_weights_by_month=active_benchmark_weights_by_month,
                **_manager_filter_kwargs(manager_classification, manager_overrides),
                capture_rebalance=True,
            )
        if include_returns:
            returns_by_config[_label_key(label)] = _without_attrs(ret)
            returns_by_config_id[_config_id(label)] = _without_attrs(ret)
        active_sharpe = _periodic_sharpe(active_return_stream(ret, benchmark)) * math.sqrt(12)
        return_metrics = _return_metrics(ret)
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
        rows.append({
            "config_id": _config_id(label),
            "manager_filter_mode": cfg.manager_filter_mode,
            **label,
            **return_metrics,
            **row_metrics,
            "active_sharpe": active_sharpe,
            **_rebalance_validity_metrics(ret.attrs.get("rebalance_summary"), cfg),
        })
        if checkpoint_dir is not None and (i == total or (checkpoint_every > 0 and i % checkpoint_every == 0)):
            _write_grid_checkpoint(
                rows,
                checkpoint_dir,
                returns_by_config_id if include_returns else None,
            )
    out = pd.DataFrame(rows)
    if include_returns:
        out.attrs["returns_by_config"] = returns_by_config
        out.attrs["returns_by_config_id"] = returns_by_config_id
    return out


def walk_forward(holdings, prices, factors, base, axes, benchmark=None,
                 value_scores=None, benchmark_weights=None,
                 train_m=36, test_m=12, select_on="sharpe",
                 chars=None, visible_versions_cache=None, verbose: bool = False,
                 precomputed_returns: dict[tuple, pd.Series] | None = None,
                 security_groups=None,
                 active_benchmark_weights_by_month=None,
                 manager_classification=None,
                 manager_overrides=None):
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
                ret = _without_attrs(precomputed_returns[_label_key(label)].reindex(tr))
            else:
                ret = run_backtest(
                    holdings,
                    prices.loc[tr],
                    cfg,
                    value_scores=value_scores,
                    benchmark_weights=benchmark_weights,
                    chars=ch,
                    visible_versions_cache=visible_cache,
                    security_groups=security_groups,
                    active_benchmark_weights_by_month=active_benchmark_weights_by_month,
                    **_manager_filter_kwargs(manager_classification, manager_overrides),
                )
            sc = _periodic_sharpe(_score_series(ret, benchmark.reindex(tr) if benchmark is not None else None, select_on))
            if verbose:
                print(f"      done in {time.perf_counter() - t0:.1f}s train_{select_on}={_fmt_metric(sc)}")
            if sc > best_score:
                best, best_score, best_lbl = cfg, sc, label
        if verbose:
            print(f"    selected {best_lbl} train_{select_on}={_fmt_metric(best_score)}; running test window")
            t0 = time.perf_counter()
        if precomputed_returns is not None and _label_key(best_lbl) in precomputed_returns:
            te_ret = _without_attrs(precomputed_returns[_label_key(best_lbl)].reindex(te))
        else:
            te_ret = run_backtest(
                holdings,
                prices.loc[months[:start + train_m + test_m]],
                best,
                value_scores=value_scores,
                benchmark_weights=benchmark_weights,
                chars=ch,
                visible_versions_cache=visible_cache,
                security_groups=security_groups,
                active_benchmark_weights_by_month=active_benchmark_weights_by_month,
                **_manager_filter_kwargs(manager_classification, manager_overrides),
            ).reindex(te)
            te_ret = _without_attrs(te_ret)
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
