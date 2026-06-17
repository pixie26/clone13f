"""
Strategy dashboard: one figure that tells the whole story honestly.

Panels: cumulative net A vs B vs benchmark, drawdown, rolling 12-month factor
alpha, marginal-IR ablation, parameter plateau heatmap, and OOS/DSR scorecard.
"""
from __future__ import annotations

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


def _cum(r: pd.Series, *, fill_missing: bool = False) -> pd.Series:
    x = r.fillna(0) if fill_missing else r
    return (1 + x).cumprod()


def _dd(r: pd.Series) -> pd.Series:
    c = _cum(r, fill_missing=True)
    return c / c.cummax() - 1


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
    title="13F-clone strategy dashboard",
    path="strategy_dashboard.png",
):
    if benchmark is None:
        benchmark = pd.Series(0.0, index=retA.index, name="cash_benchmark")
    fig = plt.figure(figsize=(15, 9.5))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.28,
                          left=0.06, right=0.975, top=0.91, bottom=0.07)
    fig.suptitle(title, x=0.06, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.06, 0.935, "filing-date rebalance | factor-adjusted | walk-forward OOS | DSR haircut",
             ha="left", fontsize=9.5, color="#666")

    ax = fig.add_subplot(gs[0, :2])
    _cum(retA, fill_missing=True).plot(ax=ax, color=A_C, lw=2.2, label="A | thesis stack")
    _cum(retB, fill_missing=True).plot(ax=ax, color=B_C, lw=1.6, label="B | placebo")
    bm_name = getattr(benchmark, "name", None) or "benchmark"
    _cum(benchmark.reindex(retA.index)).plot(ax=ax, color=BM_C, lw=1.4, ls="--", label=f"benchmark | {bm_name}")
    ax.set_title("Cumulative net return (growth of 1)", fontsize=11, color=INK)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(alpha=.25)
    ax.set_xlabel("")

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    if "DSR" in dsr_info:
        lines = [
            ("OOS ann. Sharpe", f"{dsr_info.get('ann_SR', float('nan')):.2f}"),
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
            ("OOS ann. Sharpe", "skipped"),
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
    ax.fill_between(d.index, d.values, 0, color=A_C, alpha=.25)
    d.plot(ax=ax, color=A_C, lw=1.2)
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
        piv = grid_df.pivot_table(index=heat_y, columns=heat_x, values="sharpe", aggfunc="mean")
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
    ax.set_title("Parameter plateau (in-sample Sharpe): want a broad region, not one hot cell",
                 fontsize=10, color=INK)

    ax = fig.add_subplot(gs[2, 2])
    ra = _rolling_alpha(retA, factors)
    ra.plot(ax=ax, color=ACC, lw=1.4)
    ax.axhline(0, color=INK, lw=.8)
    ax.set_title("Rolling 12m factor alpha (ann.)", fontsize=10.5, color=INK)
    ax.grid(alpha=.25)
    ax.set_xlabel("")

    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return path
