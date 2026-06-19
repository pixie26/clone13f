# Quant research requirements

Read this file for signal, portfolio, backtest, performance, or research-report
tasks.

## Backtest definition

Every backtest must explicitly define the universe, signal, ranking/threshold,
rebalance frequency, weighting, holding period, execution timing, transaction
costs, slippage, benchmark, risk constraints, and date range.

Report gross and net performance, turnover, drawdown, Sharpe/Sortino where
applicable, and available sector/country/asset/factor exposures. A flat-bps or
zero-cost run is a gross diagnostic. Prefer costs that scale with order size
relative to ADV.

Never call a strategy successful from cumulative return alone. Include risk,
drawdown, turnover, robustness, and cost sensitivity.

## Signal rules

- Use adjusted prices where appropriate.
- Separate observation and execution dates. Same-close trading is unavailable
  for live use unless explicitly modelled otherwise.
- Rank only the valid contemporaneous universe and require sufficient history.
- Do not fill missing prices in ways that create fake tradability.
- Lag accounting/factor data to actual availability and use as-reported,
  point-in-time values where possible.
- Apply liquidity filters where data permits.

## Multiple testing and robustness

- Track all tested signal variants, parameters, and universe rules.
- Reserve a genuine untouched holdout period.
- Apply a multiple-testing haircut such as deflated Sharpe when selecting among
  many trials.
- State the economic rationale before testing.
- Check subperiods and dependence on individual assets or crisis windows.

## Acceptance checklist

A research task is incomplete until it answers:

- Is it point-in-time and reproducible from config?
- Are universe rules, costs, turnover, gross/net results, and execution explicit?
- Is performance robust across subperiods and not dominated by one asset/event?
- How many variants were tried and was multiple testing addressed?
- Was the holdout untouched?
- Are weak results, failures, and coverage gaps reported honestly?
