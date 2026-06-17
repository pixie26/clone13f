"""
Strategy dashboard: one figure that tells the whole story honestly.

Panels: cumulative net A vs B vs benchmark, drawdown, rolling 12-month factor
alpha, marginal-IR ablation, parameter plateau heatmap, and OOS/DSR scorecard.
"""
from __future__ import annotations

import json
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
    ra.plot(ax=ax, color=ACC, lw=1.4)
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


def interactive_results(
    grid_df: pd.DataFrame,
    returns_by_config_id: dict[str, pd.Series],
    benchmark: pd.Series | None = None,
    path: str = "interactive_results.html",
) -> str:
    """Write a self-contained HTML viewer for parameter-sweep results."""
    table_cols = [
        "config_id",
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
    payload_json = json.dumps(payload, default=str, allow_nan=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>13F Parameter Sweep Results</title>
  <style>
    :root {{
      --ink: #1b1b1f;
      --muted: #61636b;
      --line: #d8dbe2;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --accent: #0b6e4f;
      --bench: #7b8494;
      --bad: #b0413e;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; color: var(--ink); background: var(--bg); }}
    header {{ padding: 18px 24px 12px; border-bottom: 1px solid var(--line); background: var(--panel); }}
    h1 {{ margin: 0 0 6px; font-size: 20px; font-weight: 650; }}
    .sub {{ color: var(--muted); font-size: 13px; }}
    main {{
      display: grid;
      grid-template-columns: minmax(300px, var(--left-panel-width, 420px)) 8px minmax(0, 1fr);
      gap: 8px;
      padding: 16px 20px 24px;
      align-items: start;
      overflow-x: auto;
    }}
    aside, section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
    aside {{ padding: 14px; max-height: calc(100vh - 105px); overflow: auto; min-width: 0; }}
    section {{ padding: 14px; min-width: 0; }}
    #splitter {{
      align-self: stretch;
      min-height: calc(100vh - 105px);
      border-radius: 8px;
      cursor: col-resize;
      background: linear-gradient(to right, transparent 0 2px, var(--line) 2px 6px, transparent 6px 8px);
      opacity: 0.85;
      touch-action: none;
    }}
    #splitter:hover, body.dragging-splitter #splitter {{ background: linear-gradient(to right, transparent 0 2px, var(--accent) 2px 6px, transparent 6px 8px); opacity: 1; }}
    body.dragging-splitter {{ cursor: col-resize; user-select: none; }}
    .filters {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }}
    label {{ display: grid; gap: 4px; font-size: 12px; color: var(--muted); }}
    select {{ width: 100%; padding: 7px 8px; border: 1px solid var(--line); border-radius: 6px; background: white; color: var(--ink); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    #results {{ min-width: 1120px; }}
    th, td {{ padding: 7px 6px; border-bottom: 1px solid #eceef3; text-align: right; white-space: nowrap; }}
    th {{ position: sticky; top: 0; background: var(--panel); color: var(--muted); font-weight: 600; cursor: pointer; }}
    td:first-child, th:first-child {{ text-align: left; }}
    tr {{ cursor: pointer; }}
    tr.active {{ background: #eaf4ef; }}
    tr.invalid {{ color: #8b3b34; background: #fff7f5; }}
    .summary {{ display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 10px; margin-bottom: 14px; }}
    .metric {{ border: 1px solid #eceef3; border-radius: 8px; padding: 10px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    .metric strong {{ font-size: 18px; }}
    .chart-wrap {{ border: 1px solid #eceef3; border-radius: 8px; padding: 10px; }}
    svg {{ width: 100%; height: 390px; display: block; }}
    .legend {{ display: flex; gap: 16px; font-size: 12px; color: var(--muted); margin: 8px 0 0; }}
    .key {{ display: inline-flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 18px; height: 3px; display: inline-block; }}
    @media (max-width: 760px) {{
      main {{ grid-template-columns: 1fr; }}
      #splitter {{ display: none; }}
      aside {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>13F Parameter Sweep Results</h1>
    <div class="sub">Filter configurations, inspect historical growth of 1, and compare active-weight variants.</div>
  </header>
  <main>
    <aside>
      <div class="filters" id="filters"></div>
      <table id="results"></table>
    </aside>
    <div id="splitter" role="separator" aria-orientation="vertical" aria-label="Resize left and right panels" title="Drag to resize panels; double-click to reset"></div>
    <section>
      <div class="summary" id="summary"></div>
      <div class="chart-wrap">
        <svg id="chart" viewBox="0 0 900 390" role="img" aria-label="Historical return chart"></svg>
        <div class="legend">
          <span class="key"><span class="swatch" style="background: var(--accent)"></span>Strategy growth</span>
          <span class="key"><span class="swatch" style="background: var(--bench)"></span>Benchmark growth</span>
          <span class="key"><span class="swatch" style="background: var(--bad)"></span>Drawdown</span>
        </div>
      </div>
    </section>
  </main>
  <script>
    const DATA = {payload_json};
    const filterFields = ["valid_config", "aum_band", "idea_signal", "top_n_ideas", "min_consensus_funds", "holding_horizon_q", "min_portfolio_names", "max_portfolio_names", "use_concentration", "use_low_turnover", "use_value_tilt"];
    const tableFields = ["valid_config", "aum_band", "idea_signal", "top_n_ideas", "min_consensus_funds", "holding_horizon_q", "min_portfolio_names", "max_portfolio_names", "ann_return", "active_sharpe", "invested_month_frac", "avg_effective_names", "avg_max_weight", "name_cap_feasible_ratio", "max_drawdown"];
    let sortField = "active_sharpe";
    let sortDir = -1;
    let selectedId = null;

    function fmt(v, pct=false) {{
      if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
      if (pct) return (v * 100).toFixed(1) + "%";
      if (typeof v === "number") return v.toFixed(2);
      return String(v);
    }}

    function isPctField(field) {{
      return ["ann_return", "ann_vol", "max_drawdown", "total_return", "invested_month_frac", "valid_rebalance_frac", "invalid_rebalance_frac", "avg_max_weight", "avg_max_issuer_weight", "name_cap_feasible_ratio", "issuer_cap_feasible_ratio", "zero_contributor_manager_frac"].includes(field);
    }}

    function initFilters() {{
      const root = document.getElementById("filters");
      root.innerHTML = "";
      for (const field of filterFields) {{
        if (!DATA.records.some(r => Object.prototype.hasOwnProperty.call(r, field))) continue;
        const values = [...new Set(DATA.records.map(r => r[field]).filter(v => v !== null && v !== undefined))];
        const label = document.createElement("label");
        label.textContent = field;
        const select = document.createElement("select");
        select.dataset.field = field;
        select.innerHTML = `<option value="">All</option>` + values.map(v => `<option value="${{v}}">${{v}}</option>`).join("");
        if (field === "valid_config" && values.includes(true)) select.value = "true";
        select.addEventListener("change", render);
        label.appendChild(select);
        root.appendChild(label);
      }}
    }}

    function filteredRows() {{
      const selects = [...document.querySelectorAll("#filters select")];
      return DATA.records.filter(row => selects.every(sel => !sel.value || String(row[sel.dataset.field]) === sel.value))
        .sort((a, b) => {{
          const av = a[sortField], bv = b[sortField];
          if (av === bv) return 0;
          if (av === null || av === undefined) return 1;
          if (bv === null || bv === undefined) return -1;
          return av > bv ? sortDir : -sortDir;
        }});
    }}

    function renderTable(rows) {{
      const table = document.getElementById("results");
      const head = `<thead><tr>${{tableFields.map(f => `<th data-field="${{f}}">${{f}}</th>`).join("")}}</tr></thead>`;
      const body = rows.map(row => `<tr data-id="${{row.config_id}}" class="${{row.config_id === selectedId ? "active" : ""}} ${{row.valid_config === false ? "invalid" : ""}}">
        ${{tableFields.map(f => `<td>${{fmt(row[f], isPctField(f))}}</td>`).join("")}}
      </tr>`).join("");
      table.innerHTML = head + `<tbody>${{body}}</tbody>`;
      table.querySelectorAll("th").forEach(th => th.addEventListener("click", () => {{
        const field = th.dataset.field;
        if (sortField === field) sortDir *= -1; else {{ sortField = field; sortDir = -1; }}
        render();
      }}));
      table.querySelectorAll("tr[data-id]").forEach(tr => tr.addEventListener("click", () => {{
        selectedId = tr.dataset.id;
        render();
      }}));
    }}

    function scale(vals, minPx, maxPx) {{
      const finite = vals.filter(v => Number.isFinite(v));
      const min = Math.min(...finite), max = Math.max(...finite);
      const span = Math.max(1e-9, max - min);
      return v => maxPx - (v - min) / span * (maxPx - minPx);
    }}

    function pathFor(values, yScale) {{
      const n = values.length;
      return values.map((v, i) => `${{i === 0 ? "M" : "L"}} ${{50 + i * (820 / Math.max(1, n - 1))}} ${{yScale(v)}}`).join(" ");
    }}

    function renderChart(row) {{
      const svg = document.getElementById("chart");
      const s = DATA.series[row.config_id];
      if (!s) {{ svg.innerHTML = `<text x="450" y="195" text-anchor="middle" fill="#61636b">No return series</text>`; return; }}
      const combined = [...s.growth, ...(s.benchmark || [])];
      const y = scale(combined, 28, 260);
      const yDD = v => 350 - (v / Math.min(-0.01, ...s.drawdown)) * 70;
      const benchPath = s.benchmark && s.benchmark.length ? `<path d="${{pathFor(s.benchmark, y)}}" fill="none" stroke="#7b8494" stroke-width="2"/>` : "";
      svg.innerHTML = `
        <line x1="50" y1="260" x2="870" y2="260" stroke="#d8dbe2"/>
        <line x1="50" y1="350" x2="870" y2="350" stroke="#d8dbe2"/>
        <path d="${{pathFor(s.growth, y)}}" fill="none" stroke="#0b6e4f" stroke-width="3"/>
        ${{benchPath}}
        <path d="${{pathFor(s.drawdown, yDD)}}" fill="none" stroke="#b0413e" stroke-width="2"/>
        <text x="50" y="18" fill="#1b1b1f" font-size="14">${{row.config_id}}</text>
        <text x="50" y="382" fill="#61636b" font-size="12">${{s.dates[0]}}</text>
        <text x="870" y="382" fill="#61636b" font-size="12" text-anchor="end">${{s.dates[s.dates.length - 1]}}</text>`;
    }}

    function renderSummary(row) {{
      const items = [
        ["Valid config", fmt(row.valid_config)],
        ["Total return", fmt(row.total_return, true)],
        ["Ann. return", fmt(row.ann_return, true)],
        ["Invested months", fmt(row.invested_month_frac, true)],
        ["Avg names", fmt(row.avg_effective_names)],
        ["Name cap OK", fmt(row.name_cap_feasible_ratio, true)],
        ["Active Sharpe", fmt(row.active_sharpe)],
      ];
      document.getElementById("summary").innerHTML = items.map(([k, v]) => `<div class="metric"><span>${{k}}</span><strong>${{v}}</strong></div>`).join("");
    }}

    function initSplitter() {{
      const splitter = document.getElementById("splitter");
      const root = document.documentElement;
      if (!splitter) return;
      const saved = localStorage.getItem("sweepLeftPanelWidth");
      if (saved) root.style.setProperty("--left-panel-width", saved);
      let dragging = false;
      splitter.addEventListener("pointerdown", event => {{
        dragging = true;
        splitter.setPointerCapture(event.pointerId);
        document.body.classList.add("dragging-splitter");
      }});
      splitter.addEventListener("pointermove", event => {{
        if (!dragging) return;
        const left = Math.max(300, Math.min(760, event.clientX - 20));
        const value = `${{left}}px`;
        root.style.setProperty("--left-panel-width", value);
        localStorage.setItem("sweepLeftPanelWidth", value);
      }});
      function stopDrag(event) {{
        dragging = false;
        document.body.classList.remove("dragging-splitter");
        try {{ splitter.releasePointerCapture(event.pointerId); }} catch (_) {{}}
      }}
      splitter.addEventListener("pointerup", stopDrag);
      splitter.addEventListener("pointercancel", stopDrag);
      splitter.addEventListener("dblclick", () => {{
        localStorage.removeItem("sweepLeftPanelWidth");
        root.style.removeProperty("--left-panel-width");
      }});
    }}

    function render() {{
      const rows = filteredRows();
      if (!selectedId || !rows.some(r => r.config_id === selectedId)) selectedId = rows[0]?.config_id || null;
      renderTable(rows);
      const row = rows.find(r => r.config_id === selectedId);
      if (row) {{ renderSummary(row); renderChart(row); }}
    }}

    initSplitter();
    initFilters();
    render();
  </script>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
