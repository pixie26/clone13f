# Project

Point-in-time SEC 13F manager-selection, portfolio construction, backtesting,
risk, and reporting infrastructure.

## Core invariants

- Use only information publicly available at the decision timestamp.
- Distinguish report, filing, rebalance, and execution dates.
- Preserve point-in-time universe membership and exact filing-version provenance.
- Model long equity holdings only unless the task explicitly changes that scope.
- Never fabricate identifiers, prices, tradability, holdings, or performance.
- Report missing, unmapped, unparseable, stale, and dropped records.
- Avoid look-ahead bias, survivorship bias, future joins, and silent coercion.
- Do not silently change strategy, benchmark, costs, timing, or output schemas.
- Keep saved research reproducible through explicit config and run manifests.
- Preserve user changes and unrelated dirty-worktree edits.

## Important paths

- `build_universe.py`: SEC filing ingestion and point-in-time universe building.
- `data_adapters.py`: external data access, mappings, prices, and caches.
- `engine.py`: signals, portfolio construction, execution, and backtesting.
- `run_example.py`: live and synthetic research orchestration and configuration.
- `report.py`: report payload generation.
- `interactive_results_template.html`: canonical interactive report UI.
- `runtime_support.py`: environment, progress, config hash, and manifest helpers.
- `run_diagnostics.py`: run diagnostics and trace summaries.
- `tests/`: regression and point-in-time tests.
- `docs/agent/`: detailed task-specific agent guidance.

## Working rules

- Inspect the relevant code and tests before editing.
- Make minimal, scoped, reviewable changes; do not rewrite unrelated modules.
- Keep data loading, signals, portfolio construction, backtesting, and reporting
  separated by responsibility.
- Preserve public imports, CLI behavior, and exported schemas unless requested.
- Keep compatibility shims during module extraction; remove them separately.
- Use deterministic seeds and explicit date handling where results depend on them.
- Do not install dependencies, change external data, or regenerate large outputs
  unless the task requires it.

## Context and large-file discipline

- Search first and read only the relevant functions, tests, and local context.
- Do not inspect `outputs/`, reports, caches, raw SEC data, or large datasets unless
  required by the task.
- For CSV, Parquet, or JSON data, inspect schema, dimensions, selected rows, and
  aggregates instead of printing entire files.
- Do not load every routed guidance document for a narrow task.

## Verification

- Run the narrowest relevant tests first.
- Broaden testing for point-in-time logic, strategy behavior, data contracts,
  timing, portfolio construction, reporting schemas, or cross-module changes.
- Use `python -B run_example.py --mode synthetic` for pipeline-level validation.
- Run configured lint, format, or type checks when relevant and available.
- Do not treat missing optional tooling as permission to install it.

## Security and generated data

- Never commit secrets, credentials, personal/client data, or restricted data.
- Keep paths configurable and provide `.env` templates rather than real secrets.
- Do not commit large generated artifacts unless they are intentionally versioned.

## Task-specific guidance

Read only the documents relevant to the current task:

- `docs/agent/quant-research.md`: signals, backtests, costs, robustness, and
  research acceptance.
- `docs/agent/point-in-time-13f.md`: SEC parsing, amendments, mappings,
  availability, and universe construction.
- `docs/agent/repository-notes.md`: naming contracts, source limitations, known
  issues, and current performance backlog.
- `docs/agent/architecture.md`: module boundaries and structural refactors.

## Handoff

- Report changed files, rationale, tests run, and remaining risks.
- For bug fixes, state root cause, historical-output impact, and whether prior
  outputs need regeneration.
