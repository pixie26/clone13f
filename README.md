# clone13f

Research infrastructure for building and testing SEC 13F clone-style equity strategies.

This repository is intended as a reproducible systematic research sandbox, not a discretionary stock-picking notebook. The pipeline builds a rule-based 13F manager universe, maps CUSIPs to tradable tickers, downloads public market data, runs point-in-time backtests, and writes auditable reports.

## What It Does

- Builds a rule-based universe from SEC 13F datasets.
- Handles filing-date visibility and amendment versions in the backtest path.
- Maps CUSIPs through OpenFIGI, with cache support and coverage diagnostics.
- Downloads monthly returns from yfinance, with cache support and Yahoo Chart API fallback.
- Supports idea signals such as `level`, `change`, `initiation`, `active_weight`, `active_weight_change`, and `active_weight_initiation`.
- Runs thesis vs placebo backtests, marginal-IR ablations, grid sweeps, walk-forward selection, and deflated Sharpe checks.
- Writes dashboard PNGs, interactive sweep HTML, sweep CSVs, rebalance audit CSVs, rule summaries, and run manifests under `reports/`.

## Main Files

- `build_universe.py` - SEC 13F dataset discovery, parsing, caching, and rule-based universe construction.
- `data_adapters.py` - network-facing adapters for OpenFIGI, yfinance/Yahoo Chart, Fama-French factors, and mapping/price diagnostics.
- `engine.py` - pure-pandas portfolio construction, point-in-time backtest, attribution, rebalance trace, and risk/cost logic.
- `sweep.py` - parameter grid evaluation, walk-forward selection, active-return scoring, and deflated Sharpe.
- `report.py` - dashboard chart rendering.
- `run_example.py` - runnable synthetic/live research pipeline.
- `data/security_overrides.csv` - issuer-group overrides for multi-class securities such as `GOOG`/`GOOGL`.
- `data/fund_ticker_exclusions.csv` - supplemental ETF/ETN/fund ticker exclusions for equity-only research runs.

## Setup

Python 3.11+ is recommended.

```powershell
python -m pip install pandas numpy scipy statsmodels matplotlib requests yfinance pandas_datareader pytest pyarrow
```

For live 13F parsing, install the SEC EDGAR helper used by the adapter:

```powershell
python -m pip install edgartools
```

Create a local `.env` file for secrets and SEC identity. Do not commit it.

```text
OPENFIGI_API_KEY=your_openfigi_key
```

Also update `LIVE_CONFIG["identity"]` in `run_example.py` before live SEC downloads. SEC requests should use a real name/email user agent.

## Run

Offline synthetic smoke run:

```powershell
python -B run_example.py --mode synthetic
```

Live data-chain smoke run:

```powershell
python -B run_example.py --mode live-smoke --smoke-cusips 300 --smoke-tickers 200
```

Full live run:

```powershell
python -B run_example.py --mode live
```

The live thesis default uses `active_benchmark_source="spy_holdings"` for
`active_weight` signals. Prepare a point-in-time monthly SPY/S&P 500 weight file
at `data/processed/benchmark_weights_spy.parquet` or pass a path explicitly:

```powershell
python -B run_example.py --mode live --active-benchmark-weights data/processed/benchmark_weights_spy.parquet
```

The file may be CSV, Parquet, or XLSX, and must contain long-form columns:

```text
month_end,ticker,weight
2020-01-31,AAPL,0.045
2020-01-31,MSFT,0.038
```

Weights can be decimals or percentages. The loader normalizes tickers such as
`BRK.B` to `BRK-B`. The default requires one point-in-time snapshot for each
rebalance month and fails if a month is missing, so a current SPY snapshot is not
silently backfilled into historical tests.

ETF-excluded equity-only live run:

```powershell
python -B run_example.py --mode live --equity-only
```

The live default uses `--price-source chart` through `LIVE_CONFIG` to avoid
`yfinance` hangs on restricted networks. To compare against yfinance manually:

```powershell
python -B run_example.py --mode live --price-source auto
```

For faster diagnostics before a full run:

```powershell
python -B run_example.py --mode live --equity-only --skip-marginal --skip-sweep
```

To populate OpenFIGI security metadata for an older ticker-only cache, run once with:

```powershell
python -B run_example.py --mode live-smoke --equity-only --refresh-openfigi-metadata
```

Outputs are written to timestamped folders under `reports/`, including:

- `strategy_dashboard.png`
- `interactive_results.html`
- `sweep_grid.csv`
- `sweep_returns.csv`
- `manifest.json`
- `rebalance_summary_thesis.csv`
- `rebalance_holdings_thesis.csv`
- `rebalance_managers_thesis.csv`
- `rebalance_rules_thesis.json`

## Testing

```powershell
python -B -m pytest tests
```

## Current Caveats

- yfinance is suitable for first-pass infrastructure validation, not publishable delisting-sensitive research. CRSP/WRDS or an equivalent survivorship-aware source is the preferred production-grade source.
- CUSIP/OpenFIGI mapping coverage is incomplete and must be reviewed through the run diagnostics. Large unmapped value is a research-validity risk.
- `missing_price_policy="exit"` is a pragmatic public-data fallback, not a substitute for true delisting returns.
- Prior-period turnover can still use a later amendment of the prior period; this is tracked as a known point-in-time issue in `AGENTS.md`.
- Backtest results should be interpreted through active/factor-adjusted metrics, turnover, drawdown, and robustness checks. Do not judge the strategy by cumulative return alone.

## Git Hygiene

The repository intentionally ignores local data and generated artifacts:

- `.env`
- `13f_cache/`
- `reports/`
- `artifacts/`
- `openfigi_cache.parquet`
- `yfinance_close_cache.parquet`
- `yfinance_close_cache_coverage.parquet`

Regenerate these locally as needed.
