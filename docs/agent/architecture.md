# Architecture and refactoring guidance

Read this file for module extraction, package layout, dependency, or broad test
organization work.

## Target boundaries

Move gradually toward:

```text
src/clone13f/
|-- config.py
|-- data/          # SEC, identifiers, prices, factors, benchmarks
|-- research/      # manager characteristics, universes, signals
|-- portfolio/     # target construction and constraints
|-- backtest/      # execution, accounting, diagnostics, attribution
|-- reporting/     # artifacts and interactive reports
|-- runtime/       # manifests, cache keys, progress, orchestration support
`-- cli/           # argument parsing and thin entry points
```

Do not force this layout in one rewrite. Extract cohesive, low-risk code first,
retain compatibility imports, and keep each patch independently testable.

## Dependency direction

Data modules must not depend on portfolio/backtest/reporting. Research logic may
depend on normalized data, portfolio logic may depend on research outputs, and
reporting may consume all public result objects. CLI/orchestration wires the
layers together but should not contain domain calculations.

Keep PIT-sensitive transformations explicit and auditable. Module movement must
not silently change defaults, date semantics, cache keys, or strategy behavior.

Split tests by domain as modules stabilize; avoid moving implementation and a
large unrelated test set in the same patch.
