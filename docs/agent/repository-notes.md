# Current repository contracts and known limitations

Read this file before changing live-data, cache, manager-characteristic,
walk-forward, or current-strategy behavior.

## Contracts

- `fetch_prices` returns monthly returns, despite compatibility variables named
  `prices`. Do not perform a broad rename outside an active requested refactor.
- For long live runs, set `sec_history_start` earlier than `start`. A reasonable
  2015-start setup is SEC history from `2013-10-01`, prices from at least 2014,
  and the backtest from `2015-01-01`.
- Current walk-forward settings (`train_m=48`, `test_m=12`) need at least 60
  months before walk-forward and deflated-Sharpe checks are meaningful.
- Default benchmark is `SPY`; use a style proxy such as `QQQ` only when the
  research question requires it.

## Data-source limitations

- yfinance is acceptable for infrastructure validation, not publishable
  delisting-sensitive research. Use CRSP/WRDS or equivalent for that purpose.
- `missing_price_policy="exit"` is a fallback, not true delisting returns.
- `manager_held_mcap` uses Yahoo historical shares and split-adjusted closes.
  Stored dates are used point-in-time, but Yahoo may revise history; never label
  this proxy strict vendor PIT.

## Known PIT and coverage issues

- Prior-period turnover in `manager_characteristics` may use a later amendment of
  the prior period. The strict fix is turnover recomputed as of each rebalance.
- OpenFIGI/CUSIP mapping remains incomplete and may be systematic around stale
  IDs, corporate actions, foreign issuers, and non-common instruments.

## Performance order of operations

1. Time SEC/cache reads, OpenFIGI, prices, `manager_characteristics`, backtest,
   grid evaluation, walk-forward, and report rendering.
2. Reuse/cache `chars` across marginal IR, grid evaluation, and walk-forward.
3. Only if profiling supports it, vectorize low-risk scalar manager metrics first.
4. Keep book weights, turnover, amendment handling, target generation, and carry
   horizon conservative until protected by tests.
5. Consider sparse holdings only if full-universe runs become memory-bound.
6. Upgrade flat costs to ADV/order-size-aware impact.
