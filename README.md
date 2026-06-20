# clone13f

Research infrastructure for building and testing SEC 13F clone-style equity strategies.

This repository is intended as a reproducible systematic research sandbox, not a discretionary stock-picking notebook. The pipeline builds a rule-based 13F manager universe, maps CUSIPs to tradable tickers, downloads public market data, runs point-in-time backtests, and writes auditable reports.

## Research Wiki / 研究概览

> Status: research infrastructure and an evolving replication study, not an
> investable product or a claim of proven alpha. / 当前状态：研究基础设施与持续迭代的复现研究，
> 不是可直接交易的产品，也不代表超额收益已经得到确认。

### 中文摘要

本项目研究一个可检验的问题：公开披露的 SEC Form 13F 能否在信息实际可得之后，
用于识别机构投资者的高确信度股票，并构造扣除交易成本后仍有解释力的组合？研究动机来自
Cohen、Polk 与 Silli 的 *Best Ideas*：基金持仓中相对基准更大的主动超配，可能比普通持仓
包含更多经理人特有信息。本项目没有把论文结论当成既定事实，而是把它工程化为可审计的
点时（point-in-time, PIT）数据、信号、组合、回测、稳健性检验和报告流水线。

当前实现可以从 SEC 13F 数据建立经理与持仓历史，按 `filing_date` 控制可见性和修订版本，
经 OpenFIGI 映射证券，接入公开价格、因子和市值代理，生成主动权重及 CPS 风格的隐含信息率
信号，再进行 thesis/placebo 对照、边际贡献消融、参数网格、48/12 月滚动样本外选择和
Deflated Sharpe 检查。现阶段可以确认的是“研究链路已具备可复核性”；尚不能确认的是
“策略存在可发表或可交易的稳定 alpha”。

### 背景与论文基线

13F 每季度披露美国机构经理人的多头证券持仓，但不提供完整空头、现金、衍生品经济敞口，
也不能揭示季度内交易路径。披露通常滞后报告期末，因此严谨复现必须区分
`period_date`、`filing_date`、组合再平衡日和执行收益期，不能按季度末持仓回填未来才公开的信息。

本项目主要检验 [Cohen, Polk and Silli, *Best Ideas*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1364827)
所代表的假设：经理人相对其投资机会集的高确信度主动持仓，平均而言可能优于其余持仓。
论文结果在这里仅作为待复现的经济假设；数据时期、基金样本、基准、成本和可交易性不同，
因此不能把原论文的统计结果直接外推到本项目。

### 数据与点时设计

| 层次 | 当前数据/实现 | 关键约束 |
| --- | --- | --- |
| 13F | SEC filings/datasets；保留报告期、披露日、修订和 accession | 决策时只使用当时已公开的版本 |
| 标识映射 | CUSIP → OpenFIGI → ticker，带缓存和覆盖率诊断 | 映射缺失或公司行动可能造成系统性样本偏差 |
| 收益 | Yahoo/yfinance 月收益与缓存 | 适合基础设施验证，不替代含退市收益的 CRSP/WRDS |
| 主动基准 | 经理持仓子集的市值权重；亦支持历史 SPY 权重或可见 13F 聚合 | Yahoo 历史 shares 可能被修订，当前并非严格 vendor PIT |
| 风险模型 | Fama-French 因子；默认用过去 24 个月 CAPM 残差波动率 | 只使用信号月之前的数据；窗口和截尾为工程假设 |
| 审计 | 映射、覆盖、剔除、再平衡、持仓、成本、配置和 manifest | 缺失数据不填成零，不把无价格证券伪装为可交易 |

ETF/ETN/基金类持仓在当前 live 默认中被排除；若研究经理人的 beta allocation，可显式纳入并
单独报告。公开数据下的 `missing_price_policy="exit"` 只是缓解完整历史样本的幸存者偏差，
不是退市收益模型。

### 工程实现与研究流程

1. 建立并缓存 SEC 持仓版本，记录解析与覆盖诊断。
2. 在每个再平衡月选择当时可见且未过期的 filing/amendment。
3. 过滤经理、证券和数据质量，计算集中度、换手、规模、PUT 权重等特征。
4. 计算 book weight、主动超配和 CPS 风格信号；默认分数为
   `positive_active_overweight × trailing_CAPM_residual_vol`。
5. 每位经理选择 Top 1/3/5 ideas；当前论文式默认使用经理等预算，经理内部按申报 book weight
   分配，再施加单名、issuer 和总名称数约束。
6. 以月收益模拟披露后再平衡，扣除默认单边 15 bps 成本，并输出逐月交易审计。
7. 与宽松 placebo、SPY 和因子模型比较，执行消融、预声明网格、walk-forward 与多重检验修正。
8. 输出图表、交互报告、CSV 审计文件和包含 git SHA/config/input hash 的 manifest。

### 可选参数

主要研究轴包括：

- 经理集合：AUM 区间、持仓数、Top-10 集中度、低换手、PUT 权重、价值倾向、历史长度，
  以及 `all` / `exclude_dirty` / `dedicated_like` 分类模式。
- 信号：`level`、`change`、`initiation`、主动权重系列和 `cps_ir` 系列。
- 组合：每位经理 Top 1/3/5、`manager_equal` 或实验性 score 聚合、最低共识经理数、
  持有/延续季度、最少/最多股票数、单名和同 issuer 上限。
- 数据与基准：价格源、ETF 是否剔除、`manager_held_mcap`、历史 SPY holdings 或
  `visible_13f_aggregate`，以及允许的快照陈旧天数。
- 验证：回测区间、交易成本、48 个月训练/12 个月测试、网格轴与 checkpoint 频率。

CLI 只暴露常用运行开关；完整研究参数由 `BacktestConfig`、`LIVE_CONFIG` 和预声明 sweep axes
控制，并写入每次运行的 manifest，避免只保存“最好的结果”。

### 当前证据与结论（截至 2026-06-20）

一份本地探索性 live 快照（git `ab06d0b9`，2015-01 至 2026-05，137 个月）报告：thesis
年化收益 17.0%、年化波动 23.8%、Sharpe 0.69、相对 SPY 的 IR 0.23，六因子年化 alpha
约 6.3%，但 alpha t 值仅 1.14。该运行有 135 个再平衡月，其中 9 个月因最低持仓数约束无效，
平均单边换手约 21.1%；采用 15 bps 单边成本。

这些数字**不是项目的最终结果**：该快照跳过参数 sweep/DSR，使用旧版 `score` 聚合而非当前
`manager_equal` 默认，且主动基准依赖 `strict_pit_row_fraction=0` 的 Yahoo shares proxy。
因此当前最稳健的结论是：信号在某些设定下有正向点估计，但统计证据不足、对实现版本敏感，
尚不能拒绝“结果来自噪声、数据偏差或参数选择”的解释。代码与审计链路已经达到进一步严谨
复现的基础，但策略有效性仍是开放问题。

### 已知问题与后续研究

- 用 CRSP/WRDS 或等价数据替换 Yahoo，纳入真实退市收益、证券历史和可审计 PIT 市值。
- 修复前期换手可能引用后来 amendment 的问题：按每个再平衡日重新构造前期可见版本。
- 提升 CUSIP 映射和公司行动覆盖，并量化未映射市值是否集中于特定证券类型或时期。
- 引入真实历史 SPY/SPX 成分权重；当前快照不能回填历史。
- 对经理分类加入 Form ADV/外部标签验证；当前 `dedicated_like` 主要是本地行为分类。
- 比较 CAPM 与 FF5+MOM 残差波动率、不同窗口、成本和流动性冲击模型。
- 预留真正未触碰的 holdout，报告子时期、危机期、个股集中度和参数稳定性。
- 重新运行当前 `manager_equal` 默认的完整 live sweep/DSR；在此之前不升级研究结论。

### English abstract

This project asks whether public SEC Form 13F disclosures can identify managers'
high-conviction equity ideas *after the disclosures become available*, and whether
those ideas support a cost-aware, reproducible portfolio. Inspired by Cohen,
Polk and Silli's *Best Ideas*, it treats relative active overweight as a hypothesis
about manager-specific information—not as an established source of alpha.

The repository implements a point-in-time pipeline for SEC filing versions,
CUSIP mapping, public returns and factor data, manager classification, active
weights, CPS-style implied-information-ratio signals, portfolio construction,
backtesting, ablation, parameter sweeps, walk-forward selection, deflated Sharpe
checks, and auditable reporting. Report dates, filing dates, rebalance dates, and
return realization are kept distinct; missing observations are counted rather
than converted into fabricated zero returns.

The current evidence supports an engineering conclusion, not an investment
conclusion: the research chain is reproducible and inspectable, while stable
alpha is not yet established. One exploratory 2015-2026 live snapshot produced
positive return, IR, and factor-alpha point estimates, but its alpha t-statistic
was only 1.14; it skipped the full sweep/DSR, used an older score-weighted
aggregation, and relied on a revisable Yahoo shares proxy. The next acceptance
bar is a full run of the current manager-equal specification with audited PIT
market-cap and delisting-aware return data, a true untouched holdout, realistic
liquidity costs, and robustness across subperiods and parameter neighborhoods.

**Research summary.**

- **Prior research:** *Best Ideas* motivates the hypothesis that a manager's
  largest active overweights contain more information than ordinary holdings.
  This repository tests that hypothesis under its own sample and constraints;
  it does not import the paper's result as an assumption.
- **Data:** SEC filing versions and availability dates, OpenFIGI mappings,
  Yahoo/yfinance monthly returns, Fama-French factors, and either manager-held
  market-cap weights, historical SPY weights, or a visible-13F diagnostic proxy.
- **Implementation:** PIT universe selection, manager classification, CPS-style
  ranking, manager-equal portfolio construction, cost-aware monthly simulation,
  thesis/placebo comparison, ablation, walk-forward evaluation, and manifests.
- **Configurable axes:** manager/AUM filters, signal family, Top 1/3/5 ideas,
  aggregation, consensus, carry horizon, portfolio caps, benchmark source,
  data source, costs, date windows, and sweep design.
- **Current conclusion:** positive exploratory point estimates exist, but they
  are not statistically decisive or yet robust to the current specification,
  strict PIT data, full multiple-testing control, and an untouched holdout.
- **Research agenda:** amendment-safe turnover, delisting-aware returns, audited
  PIT market caps, historical benchmark constituents, validated manager labels,
  liquidity impact, richer residual-risk models, and subperiod stability.

The Chinese sections above are the canonical detailed research summary; the
operational reference below documents files, setup, commands, outputs, and
implementation-specific caveats.

## What It Does

- Builds a rule-based universe from SEC 13F datasets.
- Handles filing-date visibility and amendment versions in the backtest path.
- Maps CUSIPs through OpenFIGI, with cache support and coverage diagnostics.
- Downloads monthly returns from yfinance, with cache support and Yahoo Chart API fallback.
- Supports idea signals such as `level`, `change`, `initiation`, `active_weight`, `active_weight_change`, `active_weight_initiation`, `cps_ir`, `cps_ir_change`, and `cps_ir_initiation`.
- Supports PIT manager-type filtering with `all`, `exclude_dirty`, and `dedicated_like` modes.
- Runs thesis vs placebo backtests, marginal-IR ablations, grid sweeps, walk-forward selection, and deflated Sharpe checks.
- Writes dashboard PNGs, interactive sweep HTML, sweep CSVs, rebalance audit CSVs, rule summaries, and run manifests under `reports/`.

## Main Files

- `build_universe.py` - SEC 13F dataset discovery, parsing, caching, and rule-based universe construction.
- `data_adapters.py` - network-facing adapters for OpenFIGI, yfinance/Yahoo Chart, Fama-French factors, and mapping/price diagnostics.
- `engine.py` - pure-pandas portfolio construction, point-in-time backtest, attribution, rebalance trace, and risk/cost logic.
- `sweep.py` - parameter grid evaluation, walk-forward selection, active-return scoring, and deflated Sharpe.
- `manager_classifier.py` - PIT manager behavior/type classifier for cleaning the idea-generation universe.
- `report.py` - dashboard chart rendering.
- `run_example.py` - runnable synthetic/live research pipeline.
- `data/security_overrides.csv` - issuer-group overrides for multi-class securities such as `GOOG`/`GOOGL`.
- `data/fund_ticker_exclusions.csv` - supplemental ETF/ETN/fund ticker exclusions for equity-only research runs.
- `data/manager_overrides.csv` - optional manager allow/deny overrides for filter-active manager modes.

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

The live thesis default uses `active_benchmark_source="manager_held_mcap"`.
For each visible manager filing, the engine restricts that month's market caps
to the manager's security-filtered common-stock holdings and normalizes them to
one. Active weight is `manager_weight - held_portfolio_market_cap_weight`.
Missing market caps are excluded and audited; they are never filled with zero,
and there is no fallback to the peer-13F aggregate.

The first live run incrementally builds
`data/processed/market_cap_history.parquet` from Yahoo historical shares and
month-end closes. Split events are applied so prices and shares use a
consistent share basis, including split-transition months. Each batch is
checkpointed and subsequent runs reuse the cache. This free source is explicitly
labelled `yahoo_shares_proxy`: Yahoo may revise historical shares, so it is a
research-infrastructure proxy, not strict CRSP/Compustat point-in-time market
capitalization. Replace the cache with an audited long table containing
`month_end,ticker,market_cap,available_date,source,strict_pit` for publishable work.

The live thesis default uses `manager_filter_mode="dedicated_like"`, while
`all` remains the untouched baseline. Manager filtering modes:

- `all` - no manager classifier or override is applied; this is the parity anchor.
- `exclude_dirty` - drops obvious out-of-scope filers and extreme behavior fingerprints.
- `dedicated_like` - keeps low-turnover, concentrated, bounded-breadth managers after calendar-quarter persistence.

The classifier is local/PIT in v1. It does not use Form ADV or external Bushee
labels. `factor_r2` is reported as a diagnostic and is not a default hard filter.

Optionally, you can run active weights against point-in-time SPY/S&P 500 weights
by preparing `data/processed/benchmark_weights_spy.parquet` or passing a path
explicitly:

```powershell
python -B run_example.py --mode live --active-benchmark-source spy_holdings --active-benchmark-weights data/processed/benchmark_weights_spy.parquet
```

The file may be CSV, Parquet, or XLSX, and must contain long-form columns:

```text
month_end,ticker,weight
2020-01-31,AAPL,0.045
2020-01-31,MSFT,0.038
```

Weights can be decimals or percentages. The loader normalizes tickers such as
`BRK.B` to `BRK-B`. The default allows a recent prior-month snapshot for rare
missing months (`active_benchmark_max_stale_days=45`) and fails if coverage is
older than that, so a current SPY snapshot is not silently backfilled into
historical tests.

The repository does not auto-generate historical SPY constituent weights. A
current holdings download cannot be used for past months without look-ahead
bias.

```powershell
python -B run_example.py --mode live --active-benchmark-source visible_13f_aggregate
```

ETF-excluded equity-only live run:

```powershell
python -B run_example.py --mode live --equity-only
```

ETF/fund-like 13F rows are excluded by default in live mode. The `--equity-only`
flag remains as an explicit way to request the same setting.

CPS-style implied-IR signals are available as diagnostics:

- `cps_ir`
- `cps_ir_change`
- `cps_ir_initiation`

They rank positive active overweight by:

```text
cps_score = positive_active_overweight * trailing_CAPM_residual_vol
```

Residual volatility is point-in-time: by default it uses the prior 24 monthly
returns, requires 12 observations, and never uses the as-of month itself. Price
and factor history starts in 2014 as warm-up while reported backtest returns
still start in 2015. The
24-month window plus the 10%/80% floor/cap and 5%/95% winsorization are pragmatic
guardrails, not academically calibrated constants. They are written to the run
manifest and should be treated as research assumptions.

Future v2 data/research items:

- Replace the Yahoo shares proxy with audited PIT market-cap data.
- Add true historical SPY/SPX constituent-weight support.
- Evaluate FF5+MOM residual volatility with longer windows.
- Add ADV/Bushee enrichment for manager-type classification.

The thesis portfolio uses CPS-IR only to rank ideas within each manager. Managers
receive equal portfolio budgets; for Top 3/5, each manager's budget is allocated
across the selected names in proportion to that manager's reported book weights
(`idea_aggregation="manager_equal"`). Repeated selections therefore contribute
multiple manager-budget shares. It does not require manager overlap
(`min_consensus_funds=1`). Score-weighted aggregation remains available only as
an experimental variant and is not the paper-style allocation.

For operational convenience, the thesis currently retains
`max_portfolio_names=30`. This aggregate name limit is a documented replication
deviation rather than part of the paper-style manager-equal construction. The
default formal sweep contains 144 predeclared variants across AUM band, CPS-IR
signal form, Top 1/3/5 ideas, manager-equal vs score-weighted aggregation,
consensus threshold, and 0/1-quarter carry.

The live default uses `--price-source chart` through `LIVE_CONFIG` to avoid
`yfinance` hangs on restricted networks. To compare against yfinance manually:

```powershell
python -B run_example.py --mode live --price-source auto
```

For faster diagnostics before a full run:

```powershell
python -B run_example.py --mode live --equity-only --skip-marginal --skip-sweep
```

To compare manager universe definitions:

```powershell
python -B run_example.py --mode live --manager-filter-mode all --skip-marginal --skip-sweep
python -B run_example.py --mode live --manager-filter-mode exclude_dirty --skip-marginal --skip-sweep
python -B run_example.py --mode live --manager-filter-mode dedicated_like --skip-marginal --skip-sweep
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
- `manager_classification.csv`
- `manager_filter_acceptance.csv`

## 参数（中文） / Parameters (English)

以下先给出完整中文说明，后面保留对应英文版本。参数名、枚举值和代码中的
配置键不翻译，以便直接搜索源码和运行清单。

### 中文：Thesis 策略默认参数

这里必须区分两类默认值：`engine.py` 中 dataclass 的值是通用 API 默认值；
`run_example.py::_default_run_configs()` 会覆盖其中多项，形成实际 thesis
研究组合。下表默认指 thesis 配置。

| 层级 | 参数 | Thesis 默认值 | 可选值 / 范围 | 重要解释 |
|---|---|---:|---|---|
| 管理人池 | `manager_filter_mode` | `dedicated_like` | `all`, `exclude_dirty`, `dedicated_like` | 保留分类为集中选股型的管理人；`exclude_dirty` 只排除明显不合格者，`all` 是不筛选基线。 |
| 管理人池 | `min_aum`, `max_aum` | `$0.1B`, `$10B` | 非负美元边界 | 决策时点的管理人 AUM 区间；live 原始抓取覆盖 `$0.1B`–`$30B`。 |
| 管理人池 | `min_history_quarters` | `4` | 整数 `>=1` | 管理人进入候选池前至少需要的申报历史。 |
| 时效性 | `max_stale_filing_months` | `6` | 正整数或 `None` | 申报版本在决策月过旧时剔除。 |
| 时效性 | `max_stale_period_months` | `6` | 正整数或 `None` | 持仓报告期在决策月过旧时剔除。 |
| 集中度 | `use_concentration` | `True` | `True`, `False` | 是否启用集中度筛选。 |
| 集中度 | `top_n_concentration` | `10` | 正整数 | 计算集中度时使用的最大持仓数量。 |
| 集中度 | `min_top_n_weight` | `50%` | `0`–`1` | 前 N 大持仓合计权重的最低要求。 |
| 集中度 | `max_holdings` | `40` | 正整数 | 长股票持仓数量硬上限；通过 Top-N 集中度测试也不能绕过。 |
| 换手率 | `use_low_turnover` | `True` | `True`, `False` | 是否启用低换手管理人筛选。 |
| 换手率 | `turnover_quantile` | `0.34` | `0`–`1` | 使用横截面分位数保留较低换手的管理人。 |
| 对冲 | `use_hedge_filter` | `True` | `True`, `False` | 是否启用精确申报版本的 PUT 暴露筛选。 |
| 对冲 | `hedge_put_max_weight` | `5%` | `0`–`1` | PUT value 相对申报组合的最大比例；缺少精确版本证据时采用 fail-closed。 |
| 价值倾向 | `use_value_tilt` | `True` | `True`, `False` | 是否启用管理人价值倾向筛选。 |
| 价值倾向 | `value_tilt_min_pctl` | `50%` | `0`–`1` | 管理人价值分数的最低横截面百分位。 |
| Active share | `use_active_share` | `False` | `True`, `False` | 可选的管理人 active-share 筛选；thesis 默认关闭。 |
| Active share | `min_active_share` | `60%` | `0`–`1` | 启用筛选且存在基准权重时的最低 active share。 |
| Idea 排名 | `idea_signal` | `cps_ir` | 见下方信号表 | 只负责每位管理人内部的股票排名，不直接成为最终组合权重。 |
| Idea 选择 | `top_n_ideas` | `3` | 正整数 | 每位合格管理人选择排名最高的 3 个 idea。 |
| Idea 分配 | `idea_aggregation` | `manager_equal` | `manager_equal`, `score`, `manager_count`, `equal_name` | 决定被选 idea 如何聚合成股票目标权重。 |
| 共识 | `min_consensus_funds` | `2` | 正整数 | 股票至少被两位合格管理人选中；重复选择代表独立的管理人票数。 |
| 组合宽度 | `min_portfolio_names` | `10` | 非负整数 | 存活股票少于该值时，该次再平衡无效并持有现金。 |
| 组合宽度 | `max_portfolio_names` | `30` | 正整数或 `None` | 权重上限处理前最多保留 30 只股票；这是 thesis 的便利性约束，也是明确记录的论文复现偏差。 |
| 风险上限 | `max_name_weight` | `10%` | `0`–`1` | 单一 ticker 的最终权重上限。 |
| 风险上限 | `max_issuer_weight` | `15%` | `0`–`1` | 映射到同一发行人的多个 ticker 合计权重上限。 |
| 持有期 | `holding_horizon_q` | `0` | 整数 `>=0` | `0` 表示完全再平衡；`N` 表示目标退出后最多继续持有 N 个季度。 |
| 信号有效性 | `min_active_weight_holdings` | `10` | 正整数 | 使用 active-weight/CPS 信号前要求管理人组合达到的最低宽度。 |
| 交易成本 | `bps_per_side` | `15` bps | 非负数 | 买入和卖出分别计费。 |
| 缺失收益 | `missing_price_policy` | `exit` | `exit`, `zero`, `raise` | `exit` 卖出缺失价格的持仓并分配给其余股票；`zero` 假设收益为零；`raise` 中止运行。 |
| 主动基准 | `active_benchmark_source` | `manager_held_mcap` | `manager_held_mcap`, `visible_13f_aggregate`, `spy_holdings` | 用于 active-weight/CPS 的基准权重；默认方案是公开数据研究代理，不是严格 vendor PIT 数据。 |

`consensus_weight` 是兼容旧接口的开关。显式设置 `idea_aggregation` 后它
不生效；当 aggregation 为 `None` 时，`True` 回退到 `score`，`False`
回退到 `manager_count`。

### 中文：Idea 信号与权重聚合

| `idea_signal` | 排名含义 |
|---|---|
| `level` | 当前管理人组合权重。 |
| `change` | 组合权重的季度变化。 |
| `initiation` | 新建仓股票，按当前组合权重排名。 |
| `active_weight` | 当前组合权重减去 PIT 基准权重。 |
| `active_weight_change` | Active weight 的季度变化。 |
| `active_weight_initiation` | 新建仓股票，按 active weight 排名。 |
| `cps_ir` | Active weight × CAPM 残差波动率，即当前实现的 CPS implied-IR 排名代理。 |
| `cps_ir_change` | CPS implied-IR 代理的季度变化。 |
| `cps_ir_initiation` | 新建仓股票，按 CPS implied-IR 代理排名。 |

- `manager_equal`：每位参与管理人只有一份相同预算；其 Top-N idea 内部按
  申报持仓权重分配。同一股票被多人选中会累积多份管理人预算。
- `score`：跨管理人加总排名分数后归一化。这是实验性 score-weighting，
  不是论文式默认构造。
- `manager_count`：按选择该股票的管理人数加权。
- `equal_name`：所有最终存活股票等权。

### 中文：CPS 残差波动率参数

| 参数 | 默认值 | 解释 |
|---|---:|---|
| `idio_vol_window_months` | `24` | CAPM 残差波动率的滚动月度窗口。 |
| `idio_vol_min_obs` | `12` | 最少有效月度观测数。 |
| `idio_vol_floor` | `10%` | 年化残差波动率下限。 |
| `idio_vol_cap` | `80%` | 年化残差波动率上限。 |
| `idio_vol_winsor_lower`, `idio_vol_winsor_upper` | `5%`, `95%` | 横截面缩尾分位数。 |

这些参数只是稳定 implied-IR 排名输入。高 idiosyncratic volatility 本身
不被视为正向 alpha，原始 CPS 分数也不是 thesis 最终股票权重。

### 中文：默认稳健性 sweep

默认笛卡尔网格共有 72 组：`idea_signal` 取 `cps_ir`、
`cps_ir_change`、`cps_ir_initiation`；`top_n_ideas` 取 `1/3/5`；
`idea_aggregation` 取 `manager_equal/score`；`min_consensus_funds` 取
`1/2`；`holding_horizon_q` 取 `0/1`。

固定项为：`manager_filter_mode=dedicated_like`、AUM `$0.1B`–`$10B`、
`min_portfolio_names=10`、`max_portfolio_names=30`、
`min_active_weight_holdings=10`，并启用集中度、低换手和价值倾向筛选。
Walk-forward 使用 48 个月训练、12 个月测试，以 active Sharpe 选参；
marginal-IR 另行评估单项筛选贡献。

### 中文：Live 数据与运行参数

| 参数 | 默认值 / 可选项 | 用途 |
|---|---|---|
| `identity` | 占位字符串 | SEC 请求身份，live 使用前必须替换为真实姓名和邮箱。 |
| `openfigi_key` | `None` | 可由本地环境变量提供的 OpenFIGI API key。 |
| `sec_history_start` | `2013-10-01` | SEC 申报历史起始日。 |
| `price_history_start` | `2014-01-01` | 价格历史起始日。 |
| `start`, `end` | `2015-01-01`, `2026-05-31` | 默认回测区间。 |
| `benchmark_ticker` | `SPY` | 市场收益基准。 |
| ingest `min_aum`, `max_aum` | `$0.1B`, `$30B` | 原始抓取范围，不等于 thesis 筛选范围。 |
| ingest `max_holdings`, `max_put_weight` | `40`, `10%` | 抓取阶段的宽度和 PUT 边界；thesis 后续使用更严格的 `5%`。 |
| `require_factors` | `False` | 因子缺失是否必须中止运行。 |
| `price_source` | `chart` | 可选 `chart`, `auto`, `yfinance`。 |
| `exclude_fund_like_holdings` | `True` | 排除 ETF/ETN/基金类持仓。 |
| `refresh_openfigi_metadata`, `force_refresh_openfigi` | `False`, `False` | 控制缺失元数据刷新和强制全量重映射。 |
| 基准时效上限 | `45` 天 | Active benchmark 快照最大年龄。 |
| 市值缓存 | 自动下载 `True`，市值过期 `45` 天，股数过期 `550` 天 | 控制 manager-held market-cap 代理数据。 |
| 市值请求 | batch `25`，workers `6`，timeout `20s` | 外部请求批量、并发和超时。 |
| 缓存与 override 路径 | 见 `LIVE_CONFIG` | OpenFIGI、价格、基准、市值、证券组、管理人分类及 idio-vol 文件路径。 |

### 中文：命令行参数

| 选项 | 默认值 | 可选值 / 作用 |
|---|---|---|
| `--mode` | `synthetic` | `synthetic`, `live`, `live-smoke` |
| `--output-root` | `reports` | 输出目录。 |
| `--smoke-cusips`, `--smoke-tickers` | `300`, `200` | Smoke 模式映射和取价数量。 |
| `--skip-marginal`, `--skip-sweep` | 关闭 | 分别跳过 marginal-IR 和参数网格。 |
| `--equity-only` | 关闭 | 显式排除基金类行；当前 live 配置本身已默认排除。 |
| `--refresh-openfigi-metadata` | 关闭 | 刷新缺少分类元数据的缓存映射。 |
| `--price-source` | live 配置默认 | `chart`, `auto`, `yfinance` |
| `--active-benchmark-source` | thesis/live 默认 | `manager_held_mcap`, `visible_13f_aggregate`, `spy_holdings` |
| `--active-benchmark-weights` | 无 | 含 `month_end,ticker,weight` 的 CSV/Parquet/XLSX。 |
| `--active-benchmark-max-stale-days` | live 配置默认 | 覆盖基准快照时效上限。 |
| `--sweep-checkpoint-every` | `5` | 每 N 组保存部分网格结果；`0` 关闭。 |
| `--manager-filter-mode` | thesis 默认 | `all`, `exclude_dirty`, `dedicated_like` |

直接调用库时，未覆盖的 API 基线为
`UniverseConfig(min_aum=$1B, max_aum=$30B, ...)`、
`PortfolioConfig(top_n_ideas=8, idea_signal="level", max_name_weight=5%,
max_issuer_weight=7.5%, ...)` 和
`BacktestConfig(manager_filter_mode="all",
active_benchmark_source="visible_13f_aggregate",
missing_price_policy="exit")`。这些不是 thesis recipe。

### English: parameter reference

There are two distinct kinds of defaults. The dataclass defaults in `engine.py`
are generic API defaults; the normal research run created by
`run_example.py::_default_run_configs()` overrides many of them. The tables below
use the **thesis run** values unless explicitly labelled otherwise.

### Thesis strategy defaults

| Layer | Parameter | Thesis default | Allowed / meaningful values | Meaning |
|---|---|---:|---|---|
| Manager universe | `manager_filter_mode` | `dedicated_like` | `all`, `exclude_dirty`, `dedicated_like` | `dedicated_like` keeps managers classified as concentrated stock pickers; `exclude_dirty` only removes clearly unsuitable managers; `all` is the untouched baseline. |
| Manager universe | `min_aum`, `max_aum` | `$0.1B`, `$10B` | non-negative dollar bounds | Point-in-time manager AUM band. The raw live ingest covers `$0.1B`–`$30B` so alternative bands remain available. |
| Manager universe | `min_history_quarters` | `4` | integer `>= 1` | Minimum filing history before a manager can qualify. |
| Manager universe | `max_stale_filing_months` | `6` | positive integer or `None` | Rejects a filing version that was published too long before the decision month. |
| Manager universe | `max_stale_period_months` | `6` | positive integer or `None` | Rejects holdings whose report period is too old at the decision month. |
| Concentration | `use_concentration` | `True` | `True`, `False` | Enables the concentration screen. |
| Concentration | `top_n_concentration` | `10` | positive integer | Number of largest positions used in the concentration calculation. |
| Concentration | `min_top_n_weight` | `50%` | `0`–`1` | Required combined book weight of the largest `top_n_concentration` holdings. |
| Concentration | `max_holdings` | `40` | positive integer | Hard maximum number of long-equity holdings; passing the Top-N test does not bypass it. |
| Turnover | `use_low_turnover` | `True` | `True`, `False` | Enables the cross-sectional low-turnover screen. |
| Turnover | `turnover_quantile` | `0.34` | `0`–`1` | Keeps the lower-turnover portion of eligible managers using the configured quantile threshold. |
| Hedging | `use_hedge_filter` | `True` | `True`, `False` | Enables the filing-version PUT exposure screen. |
| Hedging | `hedge_put_max_weight` | `5%` | `0`–`1` | Maximum PUT value relative to the filing book. Missing exact-version evidence is handled fail-closed. |
| Value tilt | `use_value_tilt` | `True` | `True`, `False` | Enables the manager value-tilt screen. |
| Value tilt | `value_tilt_min_pctl` | `50%` | `0`–`1` | Minimum cross-sectional percentile of the manager value score. |
| Active share | `use_active_share` | `False` | `True`, `False` | Optional manager active-share screen; disabled in the thesis default. |
| Active share | `min_active_share` | `60%` | `0`–`1` | Minimum true active share when that screen is enabled and benchmark weights exist. |
| Idea ranking | `idea_signal` | `cps_ir` | see signal table below | Ranks securities inside each manager book; it is not the final portfolio weight. |
| Idea selection | `top_n_ideas` | `3` | positive integer | Selects each qualifying manager's top three ranked ideas. |
| Idea allocation | `idea_aggregation` | `manager_equal` | `manager_equal`, `score`, `manager_count`, `equal_name` | Determines how selected manager ideas become aggregate portfolio weights. |
| Consensus | `min_consensus_funds` | `2` | positive integer | A stock must be selected by at least two qualifying managers. Repeated selections count as independent manager votes. |
| Breadth | `min_portfolio_names` | `10` | non-negative integer | If fewer names survive, that rebalance is invalid and held as cash. |
| Breadth | `max_portfolio_names` | `30` | positive integer or `None` | Caps the aggregate target before weight caps. This Top-30 limit is a thesis convenience and an explicit paper-replication deviation. |
| Risk cap | `max_name_weight` | `10%` | `0`–`1` | Maximum final weight in one ticker. |
| Risk cap | `max_issuer_weight` | `15%` | `0`–`1` | Maximum combined weight for tickers mapped to the same issuer group. |
| Holding | `holding_horizon_q` | `0` | integer `>= 0` | `0` fully rebalances; `N` carries a dropped target for up to `N` additional quarters. |
| Signal validity | `min_active_weight_holdings` | `10` | positive integer | Minimum manager-book breadth required before active-weight/CPS signals are considered meaningful. |
| Trading cost | `bps_per_side` | `15` bps | non-negative number | Cost charged separately on purchases and sales. |
| Missing returns | `missing_price_policy` | `exit` | `exit`, `zero`, `raise` | `exit` liquidates an unpriced holding and reallocates to priced survivors; `zero` assumes zero return; `raise` stops the run. |
| Active benchmark | `active_benchmark_source` | `manager_held_mcap` | `manager_held_mcap`, `visible_13f_aggregate`, `spy_holdings` | Benchmark weights used to calculate active-weight and CPS signals. `manager_held_mcap` is a public-data research proxy, not strict vendor PIT data. |

`consensus_weight` is a legacy compatibility switch. When
`idea_aggregation` is explicitly set—as it is in the thesis run—it has no
effect. If aggregation is `None`, `consensus_weight=True` falls back to `score`
and `False` falls back to `manager_count`.

### Idea signals and allocation rules

| `idea_signal` option | Ranking quantity |
|---|---|
| `level` | Current manager book weight. |
| `change` | Quarter-over-quarter change in book weight. |
| `initiation` | Newly initiated positions, ranked by current book weight. |
| `active_weight` | Current book weight minus point-in-time benchmark weight. |
| `active_weight_change` | Quarter-over-quarter change in active weight. |
| `active_weight_initiation` | Newly initiated positions ranked by active weight. |
| `cps_ir` | Current active weight multiplied by CAPM residual volatility. This is the implemented CPS implied-IR ranking proxy. |
| `cps_ir_change` | Quarter-over-quarter change in the CPS implied-IR proxy. |
| `cps_ir_initiation` | Newly initiated positions ranked by the CPS implied-IR proxy. |

Allocation is separate from ranking:

- `manager_equal` gives every contributing manager one equal budget. Within a
  manager's Top-N selections, that budget follows the manager's reported book
  weights. If several managers select the same stock, it receives several
  manager shares.
- `score` sums ranking scores across managers and normalizes them. This is an
  experimental score-weighted construction, not the paper-style default.
- `manager_count` weights by the number of managers selecting each stock.
- `equal_name` gives every surviving selected stock equal weight.

### CPS residual-volatility inputs

| Parameter | Default | Meaning |
|---|---:|---|
| `idio_vol_window_months` | `24` | Trailing monthly window for the CAPM residual-volatility estimate. |
| `idio_vol_min_obs` | `12` | Minimum valid monthly observations. |
| `idio_vol_floor` | `10%` | Lower annualized residual-volatility guardrail. |
| `idio_vol_cap` | `80%` | Upper annualized residual-volatility guardrail. |
| `idio_vol_winsor_lower`, `idio_vol_winsor_upper` | `5%`, `95%` | Cross-sectional winsorization percentiles. |

These volatility controls stabilize the implied-IR ranking input. High
idiosyncratic volatility is not treated as an independent positive-alpha
factor, and the raw CPS score is not the thesis portfolio weight.

### Default robustness sweep

The default Cartesian sweep contains 72 configurations:

- `idea_signal`: `cps_ir`, `cps_ir_change`, `cps_ir_initiation`
- `top_n_ideas`: `1`, `3`, `5`
- `idea_aggregation`: `manager_equal`, `score`
- `min_consensus_funds`: `1`, `2`
- `holding_horizon_q`: `0`, `1`

The sweep fixes `manager_filter_mode=dedicated_like`, AUM at `$0.1B`–`$10B`,
`min_portfolio_names=10`, `max_portfolio_names=30`,
`min_active_weight_holdings=10`, and keeps concentration, low-turnover, and
value-tilt screens enabled. Walk-forward selection uses 48 training months and
12 test months, selecting on active Sharpe. Marginal-IR analysis separately
tests isolated screen contributions.

### Live data and runtime defaults

| Parameter | Default / options | Purpose |
|---|---|---|
| `identity` | placeholder string | SEC-compliant caller identity; replace with a real name and email. |
| `openfigi_key` | `None` | Optional OpenFIGI API key; local environment loading can supply it. |
| `sec_history_start` | `2013-10-01` | Earliest SEC filing-history request date. |
| `price_history_start` | `2014-01-01` | Earliest price-history request date. |
| `start`, `end` | `2015-01-01`, `2026-05-31` | Default backtest window. |
| `benchmark_ticker` | `SPY` | Broad-market return benchmark. |
| ingest `min_aum`, `max_aum` | `$0.1B`, `$30B` | Broad ingestion bounds, distinct from the thesis filter band. |
| ingest `max_holdings` | `40` | Ingestion/universe holding-count ceiling. |
| ingest `max_put_weight` | `10%` | Broad ingestion PUT bound; thesis filtering later uses `5%`. |
| `require_factors` | `False` | Whether missing factor data must abort rather than degrade optional analysis. |
| `price_source` | `chart`; options `chart`, `auto`, `yfinance` | Price downloader selection. |
| `exclude_fund_like_holdings` | `True` | Excludes ETF/ETN/fund-like rows in the configured live pipeline. |
| `refresh_openfigi_metadata`, `force_refresh_openfigi` | `False`, `False` | Control normal metadata refresh and explicit full remapping. |
| benchmark staleness | `45` days | Maximum age for active-benchmark snapshots. |
| market-cap download | auto `True`, stale `45` days, shares stale `550` days | Controls the manager-held market-cap proxy cache. |
| market-cap requests | batch `25`, workers `6`, timeout `20s` | External request batching and concurrency. |
| cache/override paths | values in `LIVE_CONFIG` | Paths for OpenFIGI, prices, benchmark weights, market cap, security groups, manager overrides, classification, and idio-vol caches. |

### Command-line parameters

| Option | Default | Allowed values / effect |
|---|---|---|
| `--mode` | `synthetic` | `synthetic`, `live`, `live-smoke` |
| `--output-root` | `reports` | Output directory. |
| `--smoke-cusips` | `300` | CUSIPs mapped in smoke mode. |
| `--smoke-tickers` | `200` | Tickers priced in smoke mode. |
| `--skip-marginal` | off | Skip marginal-IR ablation. |
| `--skip-sweep` | off | Skip grid and walk-forward sweep. |
| `--equity-only` | off | Explicitly request exclusion of fund-like rows; the current live config already defaults to exclusion. |
| `--refresh-openfigi-metadata` | off | Refresh cached mappings missing classification metadata. |
| `--price-source` | live-config default | `chart`, `auto`, `yfinance` |
| `--active-benchmark-source` | thesis/live-config default | `manager_held_mcap`, `visible_13f_aggregate`, `spy_holdings` |
| `--active-benchmark-weights` | none | CSV/Parquet/XLSX table containing `month_end,ticker,weight`. |
| `--active-benchmark-max-stale-days` | live-config default | Override benchmark snapshot age limit. |
| `--sweep-checkpoint-every` | `5` | Write partial sweep output every N configurations; `0` disables it. |
| `--manager-filter-mode` | thesis default | `all`, `exclude_dirty`, `dedicated_like` |

For direct library use, the untouched dataclass defaults are
`UniverseConfig(min_aum=$1B, max_aum=$30B, ...)`,
`PortfolioConfig(top_n_ideas=8, idea_signal="level", max_name_weight=5%,
max_issuer_weight=7.5%, ...)`, and
`BacktestConfig(manager_filter_mode="all",
active_benchmark_source="visible_13f_aggregate",
missing_price_policy="exit")`. They are API baselines, not the thesis recipe.

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

To rebuild only an existing run's interactive HTML after report-template changes,
without rerunning data ingestion, backtests, or the parameter sweep:

```powershell
python -B report.py refresh reports/<run-directory>
```

This reads `sweep_grid.csv` and `sweep_returns.csv` from the run directory and
overwrites its `interactive_results.html`. Use `--output <path>` to write elsewhere.
