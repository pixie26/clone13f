"""
Strategy dashboard: one figure that tells the whole story honestly.

Panels: cumulative net A vs B vs benchmark, drawdown, rolling 12-month factor
alpha, marginal-IR ablation, parameter plateau heatmap, and OOS/DSR scorecard.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1b1b1f"
A_C = "#0b6e4f"
B_C = "#b0413e"
BM_C = "#9aa0a6"
ACC = "#c79a3a"
INTERACTIVE_TEMPLATE = Path(__file__).with_name("interactive_results_template.html")
_INTERACTIVE_TOKENS = {
    "data": "__DATA_PAYLOAD__",
    "portfolio": "__PORTFOLIO_PAYLOAD__",
    "meta": "__META_PAYLOAD__",
}


def _format_param_lines(parameter_summary) -> list[str]:
    if not parameter_summary:
        return []
    if isinstance(parameter_summary, dict):
        lines = []
        for key, value in parameter_summary.items():
            if isinstance(value, dict):
                body = ", ".join(f"{k}={v}" for k, v in value.items())
            elif isinstance(value, (list, tuple)):
                body = ", ".join(map(str, value))
            else:
                body = str(value)
            lines.append(f"{key}: {body}")
        return lines
    return [str(x) for x in parameter_summary]


def _cum(r: pd.Series, *, fill_missing: bool = False) -> pd.Series:
    x = r.fillna(0) if fill_missing else r
    return (1 + x).cumprod()


def _dd(r: pd.Series) -> pd.Series:
    c = _cum(r, fill_missing=True)
    return c / c.cummax() - 1


def _plot_series(ax, series: pd.Series, **kwargs) -> None:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return
    ax.plot(pd.to_datetime(s.index), s.values, **kwargs)


def _rolling_alpha(
    ret: pd.Series,
    factors: pd.DataFrame,
    win: int = 12,
    cols=("MKT", "SMB", "HML", "RMW", "CMA", "MOM"),
) -> pd.Series:
    required = set(cols).union({"RF"})
    if factors is None or not required.issubset(set(factors.columns)):
        return pd.Series(index=ret.index, dtype=float)
    df = pd.concat([ret.rename("ret"), factors], axis=1).dropna()
    out = pd.Series(index=df.index, dtype=float)
    for i in range(win, len(df) + 1):
        s = df.iloc[i - win:i]
        try:
            b = sm.OLS(s["ret"] - s["RF"], sm.add_constant(s[list(cols)])).fit()
            out.iloc[i - 1] = (1 + b.params["const"]) ** 12 - 1
        except Exception:
            pass
    return out


def dashboard(
    retA,
    retB,
    benchmark,
    factors,
    ablation_df,
    grid_df,
    heat_x,
    heat_y,
    dsr_info,
    oos_log=None,
    parameter_summary=None,
    title="13F-clone strategy dashboard",
    path="strategy_dashboard.png",
):
    if benchmark is None:
        benchmark = pd.Series(0.0, index=retA.index, name="cash_benchmark")
    param_lines = _format_param_lines(parameter_summary)
    has_params = bool(param_lines)
    fig = plt.figure(figsize=(15, 10.8 if has_params else 9.5))
    fig.patch.set_facecolor("white")
    if has_params:
        gs = fig.add_gridspec(
            4,
            3,
            height_ratios=[1, 1, 1, 0.50],
            hspace=0.48,
            wspace=0.28,
            left=0.06,
            right=0.975,
            top=0.895,
            bottom=0.055,
        )
    else:
        gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.28,
                              left=0.06, right=0.975, top=0.91, bottom=0.07)
    fig.suptitle(title, x=0.06, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.06, 0.92 if has_params else 0.935,
             "filing-date rebalance | factor-adjusted | active-return walk-forward OOS | DSR haircut",
             ha="left", fontsize=9.5, color="#666")

    ax = fig.add_subplot(gs[0, :2])
    _plot_series(ax, _cum(retA, fill_missing=True), color=A_C, lw=2.2, label="A | thesis stack")
    _plot_series(ax, _cum(retB, fill_missing=True), color=B_C, lw=1.6, label="B | placebo")
    bm_name = getattr(benchmark, "name", None) or "benchmark"
    _plot_series(ax, _cum(benchmark.reindex(retA.index)), color=BM_C, lw=1.4, ls="--", label=f"benchmark | {bm_name}")
    ax.set_title("Cumulative net return (growth of 1)", fontsize=11, color=INK)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(alpha=.25)
    ax.set_xlabel("")

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    if "DSR" in dsr_info:
        metric_label = "OOS active Sharpe" if dsr_info.get("metric") == "active_return_vs_benchmark" else "OOS ann. Sharpe"
        lines = [
            (metric_label, f"{dsr_info.get('ann_SR', float('nan')):.2f}"),
            ("Expected max SR (null)", f"{dsr_info.get('expected_max_SR_null', float('nan')) * np.sqrt(12):.2f}"),
            ("# configs tried", f"{dsr_info.get('n_trials', 'n/a')}"),
            ("OOS months", f"{dsr_info.get('T', 'n/a')}"),
            ("Deflated Sharpe (DSR)", f"{dsr_info.get('DSR', float('nan')):.2f}"),
        ]
        verdict = "survives" if dsr_info.get("DSR", 0) > 0.95 else (
            "marginal" if dsr_info.get("DSR", 0) > 0.9 else "fails haircut"
        )
    else:
        lines = [
            ("OOS active Sharpe", "skipped"),
            ("Reason", dsr_info.get("note", "insufficient OOS")),
            ("# configs tried", f"{dsr_info.get('n_trials', 'n/a')}"),
            ("Price months", f"{dsr_info.get('price_months', dsr_info.get('T', 'n/a'))}/{dsr_info.get('required_months', 'n/a')}"),
            ("Deflated Sharpe (DSR)", "skipped"),
        ]
        verdict = "not evaluated"
    ax.text(0, 1.0, "Walk-forward OOS scorecard", fontsize=11, fontweight="bold", color=INK, va="top")
    for i, (k, v) in enumerate(lines):
        y = 0.80 - i * 0.16
        ax.text(0.0, y, k, fontsize=9.5, color="#555", va="center")
        ax.text(1.0, y, v, fontsize=11, fontweight="bold", color=INK, va="center", ha="right")
    vc = A_C if verdict == "survives" else (ACC if verdict in {"marginal", "not evaluated"} else B_C)
    ax.text(0.0, 0.80 - len(lines) * 0.16 - 0.04, f"DSR > 0.95 needed | {verdict}",
            fontsize=10, fontweight="bold", color=vc, va="center")

    ax = fig.add_subplot(gs[1, :2])
    d = _dd(retA)
    d_clean = d.replace([np.inf, -np.inf], np.nan).dropna()
    if not d_clean.empty:
        x = pd.to_datetime(d_clean.index)
        ax.fill_between(x, d_clean.values, 0, color=A_C, alpha=.25)
        ax.plot(x, d_clean.values, color=A_C, lw=1.2)
    ax.set_title("Drawdown | thesis (A)", fontsize=11, color=INK)
    ax.grid(alpha=.25)
    ax.set_xlabel("")

    ax = fig.add_subplot(gs[1, 2])
    ab = ablation_df[ablation_df["filter"] != "(full stack)"].copy()
    ab["filter"] = ab["filter"].str.replace("-use_", "", regex=False)
    colors = [A_C if x > 0 else B_C for x in ab["delta"]]
    ax.barh(ab["filter"], ab["delta"], color=colors)
    ax.axvline(0, color=INK, lw=.8)
    ax.set_title("Marginal IR per filter (delta when dropped)", fontsize=10.5, color=INK)
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=.25, axis="x")

    ax = fig.add_subplot(gs[2, :2])
    try:
        heat_value = "active_sharpe" if "active_sharpe" in grid_df.columns else "sharpe"
        piv = grid_df.pivot_table(index=heat_y, columns=heat_x, values=heat_value, aggfunc="mean")
        im = ax.imshow(piv.values, cmap="BuGn", aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, fontsize=8)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=8)
        ax.set_xlabel(heat_x, fontsize=9)
        ax.set_ylabel(heat_y, fontsize=9)
        for (i, j), val in np.ndenumerate(piv.values):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#333" if val < np.nanmax(piv.values) * .7 else "white")
        fig.colorbar(im, ax=ax, fraction=.035, pad=.02)
    except Exception as exc:
        ax.text(.5, .5, f"heatmap n/a\n{exc}", ha="center")
    heat_label = "active Sharpe" if "active_sharpe" in grid_df.columns else "Sharpe"
    ax.set_title(f"Parameter plateau (in-sample {heat_label}): want a broad region, not one hot cell",
                 fontsize=10, color=INK)

    ax = fig.add_subplot(gs[2, 2])
    ra = _rolling_alpha(retA, factors)
    _plot_series(ax, ra, color=ACC, lw=1.4)
    ax.axhline(0, color=INK, lw=.8)
    ax.set_title("Rolling 12m factor alpha (ann.)", fontsize=10.5, color=INK)
    ax.grid(alpha=.25)
    ax.set_xlabel("")

    if has_params:
        ax = fig.add_subplot(gs[3, :])
        ax.axis("off")
        ax.text(0.0, 0.96, "Run parameters", fontsize=10.5, fontweight="bold", color=INK, va="top")
        for i, line in enumerate(param_lines[:7]):
            ax.text(0.0, 0.74 - i * 0.12, line, fontsize=8.0, color="#444", va="top")

    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return path


def _json_clean(value):
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def _growth_of_one(returns: pd.Series) -> pd.Series:
    return (1 + returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)).cumprod()


def _drawdown(returns: pd.Series) -> pd.Series:
    growth = _growth_of_one(returns)
    return growth / growth.cummax() - 1


def _json_sanitize(value):
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _json_for_html(value) -> str:
    return (
        json.dumps(_json_sanitize(value), default=str, allow_nan=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _frame_records(frame: pd.DataFrame | None) -> list[dict]:
    if frame is None or frame.empty:
        return []
    return [
        {key: _json_clean(value) for key, value in row.items()}
        for row in frame.replace([np.inf, -np.inf], np.nan).to_dict(orient="records")
    ]


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value) if pd.notna(value) else False


def _portfolio_payload(
    holdings: pd.DataFrame | None,
    summary: pd.DataFrame | None,
    config_id: str | None,
) -> dict:
    empty = {
        "configId": config_id,
        "availableDates": [],
        "holdingsByDate": {},
        "eventsByDate": {},
        "summaryByDate": {},
        "monthStats": {},
    }
    if holdings is None or holdings.empty:
        return empty

    work = holdings.copy()
    required = {"rebalance_month", "ticker", "weight"}
    missing = sorted(required.difference(work.columns))
    if missing:
        raise ValueError(f"interactive holdings missing columns: {missing}")
    work["rebalance_month"] = pd.to_datetime(work["rebalance_month"]).dt.date.astype(str)
    work["ticker"] = work["ticker"].astype(str)
    if "issuer_group" not in work:
        work["issuer_group"] = work["ticker"]
    if "rank" not in work:
        work["rank"] = work.groupby("rebalance_month")["weight"].rank(method="first", ascending=False)
    if "is_carried" not in work:
        work["is_carried"] = False
    work = work.sort_values(["rebalance_month", "rank", "ticker"], kind="stable")

    holdings_by_date: dict[str, list[dict]] = {}
    events_by_date: dict[str, list[dict]] = {}
    month_stats: dict[str, dict] = {}
    previous: dict[str, dict] = {}
    tolerance = 1e-10
    for date, month in work.groupby("rebalance_month", sort=True):
        current: dict[str, dict] = {}
        current_rows: list[dict] = []
        for row in month.to_dict(orient="records"):
            ticker = str(row["ticker"])
            weight = float(row["weight"])
            prev = previous.get(ticker)
            prev_weight = float(prev["weight"]) if prev is not None else None
            delta = weight - (prev_weight or 0.0)
            carried = _as_bool(row.get("is_carried", False))
            status = "new" if prev is None else ("carry" if carried else "keep")
            if prev is None:
                action = "new"
            elif carried:
                action = "carry"
            elif delta > tolerance:
                action = "increase"
            elif delta < -tolerance:
                action = "decrease"
            else:
                action = "flat"
            item = {
                "date": date,
                "rank": int(row["rank"]) if pd.notna(row.get("rank")) else None,
                "ticker": ticker,
                "issuer_group": str(row.get("issuer_group") or ticker),
                "weight": weight,
                "prev_weight": prev_weight,
                "delta": delta,
                "status": status,
                "action": action,
                "is_carried": carried,
            }
            current[ticker] = item
            current_rows.append(item)

        events = list(current_rows)
        for ticker in sorted(set(previous).difference(current)):
            prev = previous[ticker]
            events.append({
                "date": date,
                "rank": None,
                "ticker": ticker,
                "issuer_group": prev["issuer_group"],
                "weight": 0.0,
                "prev_weight": float(prev["weight"]),
                "delta": -float(prev["weight"]),
                "status": "exit",
                "action": "sell",
                "is_carried": False,
            })
        events.sort(key=lambda x: (x["status"] == "exit", x["rank"] or 10**9, x["ticker"]))
        holdings_by_date[date] = current_rows
        events_by_date[date] = events

        status_counts = {name: sum(e["status"] == name for e in events) for name in ("new", "keep", "carry", "exit")}
        status_weights = {
            name: float(sum((e["prev_weight"] or 0.0) if name == "exit" else e["weight"] for e in events if e["status"] == name))
            for name in ("new", "keep", "carry", "exit")
        }
        action_counts = {name: sum(e["action"] == name for e in events) for name in ("new", "increase", "decrease", "flat", "carry", "sell")}
        action_abs_delta = {name: float(sum(abs(e["delta"]) for e in events if e["action"] == name)) for name in action_counts}
        month_stats[date] = {
            "current_names": len(current_rows),
            "current_weight": float(sum(e["weight"] for e in current_rows)),
            "status_counts": status_counts,
            "status_weights": status_weights,
            "action_counts": action_counts,
            "action_abs_delta": action_abs_delta,
            "top_current": sorted(current_rows, key=lambda x: (-x["weight"], x["ticker"]))[:20],
            "top_changes": sorted(events, key=lambda x: (-abs(x["delta"]), x["ticker"]))[:20],
            "top_increases": sorted((e for e in events if e["delta"] > tolerance), key=lambda x: (-x["delta"], x["ticker"]))[:20],
            "top_decreases": sorted((e for e in events if e["delta"] < -tolerance), key=lambda x: (x["delta"], x["ticker"]))[:20],
        }
        previous = current

    summary_by_date = {}
    for row in _frame_records(summary):
        raw_date = row.get("rebalance_month")
        if raw_date is not None:
            summary_by_date[pd.Timestamp(raw_date).date().isoformat()] = row

    return {
        "configId": config_id,
        "availableDates": sorted(holdings_by_date),
        "holdingsByDate": holdings_by_date,
        "eventsByDate": events_by_date,
        "summaryByDate": summary_by_date,
        "monthStats": month_stats,
    }


def _render_interactive_template(data: dict, portfolio: dict, meta: dict) -> str:
    template = INTERACTIVE_TEMPLATE.read_text(encoding="utf-8")
    payloads = {"data": data, "portfolio": portfolio, "meta": meta}
    for name, token in _INTERACTIVE_TOKENS.items():
        count = template.count(token)
        if count != 1:
            raise ValueError(f"interactive template must contain one {token}; found {count}")
        template = template.replace(token, _json_for_html(payloads[name]))
    return template


def single_config_result_grid(
    config_id: str,
    returns: pd.Series,
    *,
    config: dict | None = None,
    metrics: dict | None = None,
    rebalance_stats: dict | None = None,
) -> pd.DataFrame:
    """Build one interactive-result row when a parameter sweep was skipped."""
    clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    total_return = (
        float((1.0 + clean_returns).prod() - 1.0)
        if not clean_returns.empty
        else np.nan
    )
    max_drawdown = (
        float(_drawdown(clean_returns).min())
        if not clean_returns.empty
        else np.nan
    )
    row = {
        "config_id": config_id,
        **(config or {}),
        **(rebalance_stats or {}),
        **(metrics or {}),
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "valid_config": bool(not clean_returns.empty),
    }
    if "ir_vs_benchmark" in row:
        row.setdefault("ir", row["ir_vs_benchmark"])
        row.setdefault("active_sharpe", row["ir_vs_benchmark"])
    return pd.DataFrame([row])


def interactive_results(
    grid_df: pd.DataFrame,
    returns_by_config_id: dict[str, pd.Series],
    benchmark: pd.Series | None = None,
    path: str = "interactive_results.html",
    *,
    portfolio_holdings: pd.DataFrame | None = None,
    rebalance_summary: pd.DataFrame | None = None,
    portfolio_config_id: str | None = None,
    meta_payload: dict | None = None,
) -> str:
    """Write the self-contained sweep viewer using the repository HTML template."""
    table_cols = [
        "config_id",
        "manager_filter_mode",
        "aum_band",
        "use_concentration",
        "use_low_turnover",
        "use_value_tilt",
        "idea_signal",
        "top_n_ideas",
        "min_consensus_funds",
        "holding_horizon_q",
        "min_portfolio_names",
        "max_portfolio_names",
        "min_active_weight_holdings",
        "valid_config",
        "invested_month_frac",
        "valid_rebalance_frac",
        "invalid_rebalance_frac",
        "avg_effective_names",
        "avg_target_names",
        "avg_max_weight",
        "avg_max_issuer_weight",
        "name_cap_feasible_ratio",
        "issuer_cap_feasible_ratio",
        "zero_contributor_manager_frac",
        "total_return",
        "ann_return",
        "ann_vol",
        "max_drawdown",
        "sharpe",
        "active_sharpe",
        "ir",
    ]
    available_cols = [c for c in table_cols if c in grid_df.columns]
    records = [
        {k: _json_clean(v) for k, v in row.items()}
        for row in grid_df[available_cols].to_dict(orient="records")
    ]
    series_payload = {}
    for config_id, ret in returns_by_config_id.items():
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
        if ret.empty:
            continue
        bench = benchmark.reindex(ret.index).replace([np.inf, -np.inf], np.nan).fillna(0.0) if benchmark is not None else None
        active = ret - bench if bench is not None else ret
        series_payload[config_id] = {
            "dates": [pd.Timestamp(x).date().isoformat() for x in ret.index],
            "growth": [float(x) for x in _growth_of_one(ret).values],
            "benchmark": [float(x) for x in _growth_of_one(bench).values] if bench is not None else [],
            "active_growth": [float(x) for x in _growth_of_one(active).values],
            "drawdown": [float(x) for x in _drawdown(ret).values],
        }

    payload = {
        "records": records,
        "series": series_payload,
        "benchmarkName": getattr(benchmark, "name", None) if benchmark is not None else None,
    }
    portfolio = _portfolio_payload(portfolio_holdings, rebalance_summary, portfolio_config_id)
    html = _render_interactive_template(payload, portfolio, meta_payload or {})
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return str(output_path)
