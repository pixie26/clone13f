# AGENTS.md

## Mission and role

Build reusable systematic-quant research infrastructure for SEC 13F universes,
signals, portfolio construction, point-in-time backtesting, risk, and reporting.
Act as a senior quantitative research infrastructure engineer. Do not fabricate
credentials, data, or performance.

## Rules that apply to every task

1. Preserve point-in-time correctness. Use the version available at the decision
   timestamp and distinguish report, filing, rebalance, and execution dates.
2. Do not introduce look-ahead, survivorship bias, future joins, silent parse
   coercion, or silent changes to strategy, benchmark, costs, or timing.
3. Keep data loading, signals, portfolio construction, backtesting, and reporting
   separate. Prefer small, reviewable changes over rewrites.
4. Make results reproducible. Saved research needs explicit configuration and a
   manifest containing git SHA, config/input hashes, timestamp, and key versions.
5. Count and report missing, unmapped, unparseable, stale, or dropped records.
   Missing data must never become fabricated zeroes or fake tradability.
6. Preserve `interactive_results_template.html` as the canonical interactive UI.
   `report.py` injects payloads into its placeholders; do not duplicate the HTML.
7. Preserve user changes and unrelated dirty-worktree edits.

## Coding and testing

- Use Python 3.11+, `pathlib.Path`, public-function type hints, pure functions where
  practical, deterministic seeds, and explicit date formats.
- Avoid chained pandas assignment; sort indexes before rolling operations.
- Parse SEC dates through a tested helper. If coercion is unavoidable, assert a
  low `NaT` rate and report counts and samples.
- Keep compatibility shims when extracting modules; remove them only in a
  separately requested cleanup.
- Run the narrowest relevant tests, then broader tests in proportion to risk.

Preferred checks:

```text
python -B -m pytest tests
python -B -m pytest tests/test_point_in_time.py
python -B -m pytest tests/test_dates.py
python -B run_example.py --mode synthetic
```

Run configured `ruff`, `black --check`, or `mypy` checks when present. Do not
install missing packages unless requested.

## Data and security

Never commit secrets, credentials, personal/client data, unclear-licence paid
data, or large generated data not intended for version control. Prefer
configurable paths and `.env` templates.

## Response contract

For code changes, report: what changed, why, how it was tested, and limitations.
For bugs, report: root cause, minimal fix, historical-result impact, and whether
old outputs need regeneration.

## Task-specific guidance routing

Read only the documents relevant to the current task:

- `docs/agent/quant-research.md`: backtests, signals, costs, robustness, and
  research acceptance.
- `docs/agent/point-in-time-13f.md`: SEC/13F parsing, amendments, mapping,
  availability, and universe construction.
- `docs/agent/repository-notes.md`: current naming contracts, data-source limits,
  known issues, and performance backlog.
- `docs/agent/architecture.md`: module boundaries, refactors, and repository
  structure changes.

Do not load every routed document for unrelated or narrowly scoped work.
