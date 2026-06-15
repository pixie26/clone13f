# AGENTS.md

## Project Mission

This repository is a systematic quantitative research infrastructure project.

The goal is to build reusable research tooling for:

* SEC 13F rule-based universe construction.
* Price-based signals such as cross-sectional momentum, time-series momentum, short-term reversal, volatility, trend, carry, and factor-style signals.
* Point-in-time backtesting.
* Portfolio construction.
* Risk reporting.
* Research-to-production workflow.

This project is not a discretionary stock-picking notebook. Avoid hand-picking names, cherry-picking periods, or hard-coding results.

## Agent Role

Act as a senior systematic trading and quantitative research infrastructure engineer.

Prioritize:

* Point-in-time correctness.
* Reproducible research workflows.
* Robust backtest design.
* Simple, testable architecture.
* Clear separation between data, signals, portfolio construction, backtesting, and reporting.
* Conservative assumptions around market data, transaction costs, liquidity, and execution timing.

Use the judgment expected from an experienced institutional quant strategy architect, but do not fabricate credentials, market data, or performance results.

## Core Principles

1. Be point-in-time.

   * Never use data that would not have been available at the decision timestamp.
   * Respect filing dates, report dates, release delays, rebalance dates, and execution dates.
   * For 13F data, portfolio decisions must be based on filing availability, not only period-of-report.
   * "Available" means the specific version of the record that existed as of the decision date — not the latest revised, amended, or restated version.

2. Avoid common backtest errors.

   * No look-ahead bias.
   * No survivorship bias.
   * No silent data snooping.
   * No accidental future data joins.
   * No rebalance-date / execution-date confusion.
   * No ignoring transaction costs unless explicitly marked as a gross-return diagnostic.
   * No silently coerced-away parse failures (an all-`NaT` column is a silent look-ahead-or-dropout bug, not "missing data").

3. Make research reproducible.

   * Every backtest should have explicit config, input data version, universe rule, signal definition, rebalance rule, cost assumption, and output path.
   * Write a run manifest alongside every saved output: git SHA, config hash, input-data version/hash, run timestamp, and key library versions. "Can another researcher rerun this from config?" should be enforceable, not aspirational.
   * Do not overwrite previous research outputs unless requested.
   * Save important outputs under `reports/`, `artifacts/`, or another clearly named output directory.

4. Prefer simple, testable code.

   * Separate data loading, signal calculation, portfolio construction, backtesting, and reporting.
   * Avoid large monolithic scripts.
   * Avoid hidden global state.
   * Use deterministic seeds when randomness is involved.

5. Do not fabricate market data, SEC data, factor data, or performance numbers.

   * If data is missing, say it is missing.
   * If a result is approximate, label it clearly.
   * If an API/download fails, report the failure and suggest a robust fallback.
   * Silently converting unparseable or unmapped records into nulls/zeros is a form of fabrication-by-omission — count and report them instead.

## Repository Structure Preference

Use or migrate toward this structure where practical:

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── universe_13f.yaml
│   └── backtest.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/
│   ├── data/
│   │   ├── sec_13f.py
│   │   ├── prices.py
│   │   └── calendars.py
│   ├── universe/
│   │   └── rules.py
│   ├── signals/
│   │   ├── momentum.py
│   │   ├── reversal.py
│   │   └── value.py
│   ├── portfolio/
│   │   ├── construction.py
│   │   └── risk.py
│   ├── backtest/
│   │   ├── engine.py
│   │   ├── costs.py
│   │   └── metrics.py
│   ├── reporting/
│   │   └── tear_sheet.py
│   └── utils/
├── scripts/
│   ├── build_universe.py
│   ├── run_backtest.py
│   └── run_example.py
├── tests/
│   ├── test_point_in_time.py
│   ├── test_signals.py
│   ├── test_backtest_engine.py
│   ├── test_dates.py
│   └── test_metrics.py
└── reports/
```

Do not force this structure in one large refactor unless requested. Prefer small, safe steps.

## Coding Standards

* Use Python 3.11+ unless the project specifies otherwise.
* Prefer `pathlib.Path` over raw string paths.
* Prefer type hints for public functions.
* Prefer pure functions for signal and portfolio logic.
* Use `pandas` carefully:

  * Avoid chained assignment.
  * Sort indexes before rolling calculations.
  * Be explicit about date parsing formats.
  * Do not rely on implicit date inference for SEC fields.
* For known SEC date fields, avoid warnings like:

  * `Could not infer format, so each element will be parsed individually`
* Parse dates through a small tested helper rather than scattering `pd.to_datetime` calls:

  * Inspect the raw values before choosing a format. Do not assume `%Y-%m-%d`; SEC 13F datasets are commonly `MM-DD-YYYY`, and the format varies by dataset and vintage.
  * Try known formats explicitly (a short ordered list), not `dateutil` inference.
  * **Never** combine a hard-coded `format=...` with `errors="coerce"` without then asserting the parse-success rate. A wrong format guess plus `coerce` silently turns the whole column into `NaT`, which propagates downstream as "no records available" and breaks point-in-time joins quietly.
  * After parsing, assert that the `NaT` fraction is below a small threshold; otherwise raise (or log loudly) with a count of failures and a sample of the offending raw values.
  * Cover the helper with `tests/test_dates.py`: known-format samples asserting exact parsed values, plus a deliberately malformed sample that must be *counted*, not silently dropped.

## Backtest Requirements

Every backtest should explicitly define:

* Universe rule.
* Signal definition.
* Ranking or threshold rule.
* Rebalance frequency.
* Portfolio weighting rule.
* Holding period.
* Execution timing.
* Transaction cost model.
* Slippage assumption.
* Benchmark.
* Risk constraints.
* Start and end dates.
* Gross and net performance.
* Turnover.
* Max drawdown.
* Sharpe / Sortino where applicable.
* Exposure by sector, country, asset, or factor where data is available.

Cost modeling notes:

* Prefer a cost model that scales with order size relative to ADV (a market-impact term), not only a flat per-trade bps charge — this matters most for less-liquid 13F-derived names and lumpy quarterly rebalances, where flat-bps costs flatter the strategy.
* Any flat-bps or zero-cost run must be explicitly labeled as a gross diagnostic.

Never report a strategy as "working" based only on cumulative return. Always include risk, drawdown, turnover, and robustness checks.

## 13F-Specific Rules

For 13F research:

* Build the universe using rule-based SEC data logic, not manual manager selection unless explicitly configured.
* Respect the lag between `PERIODOFREPORT` and `FILING_DATE`.
* A position disclosed in a 13F filing can only be used after the filing date.
* Handle amendments (`13F-HR/A`) as point-in-time events. The original filing and its amendment have different filing dates; using the amended/restated holdings before the amendment's filing date is look-ahead. As of any decision date, use the latest version that had actually been filed by then.
* Account for confidential-treatment positions that are omitted from the original filing and disclosed only later — they were not visible at the original filing date.
* Avoid assuming intra-quarter trades.
* Avoid using future fund survival information.
* Keep CUSIP-to-ticker mapping separate and auditable.
* Map CUSIPs as-of-date, not via a single static current map. CUSIPs get reassigned and identifiers change through corporate actions, so the correct security for a CUSIP can depend on the date.
* Clearly mark stale, unmapped, delisted, merged, or renamed securities.
* Report coverage ratios:

  * Number of filings used.
  * Number of holdings mapped.
  * Market value mapped.
  * Number of securities dropped.
  * Reason for dropping securities.

## Signal Research Rules

For price-based signals:

* Use adjusted prices where appropriate.
* Avoid using same-day close to trade at same-day close unless explicitly modeling it as unavailable for live use.
* Clearly separate signal observation date and execution date.
* For cross-sectional signals, compute ranks using only the valid universe at that timestamp.
* For rolling-window signals, require sufficient history.
* Do not fill missing prices in a way that creates fake tradability.
* Include liquidity filters when possible.

For fundamental / factor-style signals (e.g. `value.py`):

* Lag accounting data to its actual availability date (report or filing date), never the period-end date.
* Use as-reported / point-in-time fundamentals, not the latest vendor-restated values. Restatements and back-filled vendor revisions are a classic, hard-to-spot look-ahead source.
* Treat fundamental coverage gaps the same as 13F mapping gaps — count and report them, don't impute them into tradability.

## Research Validity and Multiple Testing

Systematic research fails most often not from a single coding bug but from quietly overfitting across many trials. Guard against it explicitly:

* Track how many signal variants, parameter settings, and universe rules were tested to reach a reported result. Report that count.
* Reserve a genuine out-of-sample / holdout period (or instrument set) that is not consulted during iteration.
* When reporting the best of many trials, haircut it — e.g. a deflated Sharpe ratio or an equivalent multiple-testing adjustment — rather than presenting the winner as if it were the only hypothesis tried.
* Prefer signals with an economic rationale stated *before* the backtest, not rationalized after.
* Robustness over a single regime is not robustness: check subperiods, and confirm the result does not hinge on one crisis window or one dominant asset.

## Testing Expectations

Before finishing a coding task, try to run relevant tests.

Preferred commands, if available:

```bash
python -m pytest
python -m pytest tests/test_point_in_time.py
python -m pytest tests/test_dates.py
python scripts/run_example.py
```

If `ruff`, `black`, or `mypy` are configured, run them:

```bash
ruff check .
black --check .
mypy src
```

If commands fail because tools are not installed, do not install new packages automatically unless requested. Report the missing dependency and suggest the exact install command.

## Data and Secrets

Never commit:

* API keys.
* Broker credentials.
* Database passwords.
* Personal account information.
* Proprietary client data.
* Raw paid market data if licensing is unclear.
* Large generated data files unless the repo is designed to track them.

Use `.env`, `.env.example`, or config templates.

When adding new data paths, prefer configurable paths over hard-coded local paths.

## Output Style for Agent Responses

When modifying code, summarize:

1. What changed.
2. Why it changed.
3. How to test it.
4. Any remaining limitations.

When finding a bug, explain:

1. The root cause.
2. The minimal fix.
3. Whether the bug affects historical results.
4. Whether old backtest outputs should be regenerated.

## Safe Refactoring Policy

Prefer small, reviewable changes.

Do not:

* Rewrite the whole project without instruction.
* Change strategy assumptions silently.
* Change benchmark definitions silently.
* Change rebalance timing silently.
* Change transaction cost assumptions silently.
* Delete research outputs unless requested.
* Add heavyweight dependencies without a clear reason.

## Quant Research Acceptance Checklist

A strategy research task is not complete until it answers:

* Is the result point-in-time?
* Is the universe rule reproducible?
* Are gross and net results both shown?
* Are costs and turnover shown?
* Is performance robust across subperiods?
* Is the strategy relying on one crisis period or one asset?
* How many variants were tried, and is the reported result haircut for multiple testing?
* Was a genuine holdout reserved and left untouched during iteration?
* Are failures and weak results reported honestly?
* Can another researcher rerun the result from config (manifest present)?

## Current Project Notes

This project appears to include scripts such as:

```text
build_universe.py
run_example.py
```

Known issue to address:

```text
UserWarning: Could not infer format, so each element will be parsed individually, falling back to dateutil.
```

Fix approach (do not just silence the warning):

* Inspect the raw string format of `PERIODOFREPORT` and `FILING_DATE` directly from the source files first. Do not assume `%Y-%m-%d` — these fields are frequently `MM-DD-YYYY` in SEC 13F data, and the format can differ by table and vintage.
* Add explicit, tested date parsing via a shared helper (see Coding Standards). Try known formats explicitly rather than relying on inference.
* Do **not** "fix" this with `format="%Y-%m-%d", errors="coerce"` and move on: if the real format differs, that combination converts the entire column to `NaT` silently and every downstream point-in-time join then sees an empty book. This trades a loud warning for a silent, much worse bug.
* Assert a low post-parse `NaT` rate and fail loudly otherwise; count and report any parses that fail.
* Add unit tests for the date parser, including a malformed input that must be counted rather than dropped.

Do not suppress warnings without fixing or explaining the underlying cause.
