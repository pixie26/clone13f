# clone13f

**用于构建、复现与检验 SEC 13F 跟投策略的点时（point-in-time, PIT）研究框架。**

> **研究状态｜Research status**
>
> 本项目是持续迭代的论文复现与稳健性研究，不是可直接交易的产品，也不代表稳定 alpha 已得到确认。  
> 当前最可信的成果是研究基础设施和审计链路；任何策略结果都必须结合数据可得时间、参数搜索、交易成本和样本外检验解释。

## 摘要

本项目研究一个明确的问题：

> 公开披露且存在时滞的 SEC Form 13F，能否在信息实际可得之后识别机构投资者的高确信度股票，并构造扣除交易成本后仍具有稳定超额收益的组合？

研究动机来自 Cohen、Polk 与 Silli 的 *Best Ideas*：经理人相对基准的高主动超配，可能比普通持仓包含更多经理人特有信息。本项目不把论文结论视为既定事实，而是将其拆解为可审计的研究假设，并工程化为：

- SEC filing/amendment 的点时版本管理；
- CUSIP 映射、公司行动和证券类型过滤；
- 经理人样本筛选与分类；
- book weight、active weight 与 CPS 风格信号；
- manager-level idea 选择与股票级聚合；
- 交易成本、风险约束和月度回测；
- placebo、消融、parameter sweep、walk-forward 与 Deflated Sharpe；
- 逐月持仓、来源经理、筛选结果和运行 manifest。

当前结果支持的是**工程结论**，不是**投资结论**。部分探索性设定产生了较强的收益和 alpha 点估计，但统计显著性、严格 PIT 数据、经理实体清洗、现实流动性成本和 untouched holdout 尚未共同通过。

### English abstract

This project tests whether public SEC Form 13F disclosures can identify managers' high-conviction equity ideas **after the disclosures become available**, and whether those ideas can support a reproducible, cost-aware portfolio.

Inspired by Cohen, Polk and Silli's *Best Ideas*, the repository converts the economic hypotheses into an auditable point-in-time pipeline for filing versions, security mapping, manager selection, active-weight signals, portfolio formation, backtesting and robustness analysis. The current evidence supports a reproducible research process, but does not yet establish statistically reliable or investable alpha.

---

## 1. 文献、研究问题与假设

### 1.1 研究背景

Form 13F 每季度披露符合条件的美国机构投资经理人的部分多头证券持仓，但不完整披露：

- 空头、现金及完整衍生品经济敞口；
- 季度内交易路径和实际建仓价格；
- 报告期末至披露日之间的仓位变化；
- 经理人的完整投资机会集；
- 披露时经理是否仍然持有该证券。

因此，研究必须严格区分：

1. `period_date`：持仓报告期末；
2. `filing_date`：申报公开日期；
3. rebalance date：策略实际形成目标仓位的日期；
4. return period：目标仓位开始获得收益的区间。

本项目禁止将尚未公开的持仓信息回填至历史决策时点。

### 1.2 文献基线

核心参考为 Cohen、Polk 与 Silli 的 *Best Ideas*。其经济直觉是：经理人相对投资机会集的高主动超配，可能反映更强的私人信息或投资确信度。

本项目研究的是**披露后可交易版本**。这与论文中使用报告期末持仓研究经理选股能力并不完全相同。即使经理在持仓形成时具有 alpha，45 天左右的申报时滞及额外执行等待也可能消耗大部分可复制收益。

### 1.3 研究问题

公开且滞后的 13F 持仓，能否在以下约束下提供显著、稳定且可复现的超额收益：

- 仅使用当时已经公开的信息；
- 考虑映射、价格和市值覆盖；
- 扣除交易成本；
- 控制因子暴露；
- 通过样本外和多重检验；
- 对参数扰动和子时期保持稳健？

### 1.4 可检验假设及工程实现

| 假设 | 工程实现 | 主要比较 | 核心假设与局限 |
|---|---|---|---|
| **H1｜Best-idea effect**：经理人的最高确信度持仓在披露后优于普通持仓。 | 在每个 rebalance month，选择经理当时最新可见且未过期的 filing；按 `idea_signal` 在经理内部排名，选取 Top 1/3/5。严格 H1 对应 Top 1；Top 3/5 属于稳健性或可投资性扩展。 | Top 1 vs Top 3/5；高排名 idea vs 更宽松持仓/placebo；披露后收益 vs SPY 和因子调整收益。 | 申报权重或主动超配能够代理“确信度”；报告期末持仓在披露日仍有信息；披露时滞尚未完全耗尽 alpha。13F 无法观察完整空头、现金、对冲和季度内交易。 |
| **H2｜Active-weight effect**：正主动超配比单纯持仓权重包含更多增量信息。 | 计算 `BookWeight - BenchmarkWeight`；比较 `level`、`active_weight` 与 `cps_ir` 系列。CPS 使用正主动超配乘以信号月之前估计的 CAPM 残差波动率。 | `level` vs `active_weight`；`active_weight` vs `cps_ir`；不同 active benchmark。 | Active benchmark 能合理代表经理的投资机会集；历史市值数据不存在前视偏差；高 idiosyncratic volatility 的乘数具有经济含义，而非单纯放大高风险股票。 |
| **H3｜Manager-selection effect**：集中、低换手、偏主动选股的经理人具有更强信号质量。 | 在通用 AUM、历史长度、持仓数、集中度、换手、PUT 暴露等筛选之上，比较 `all`、`exclude_dirty` 和 `dedicated_like` 三种经理分类模式。 | 三种 manager universe 的收益、alpha、IR、持仓重合度和边际贡献。 | 持仓行为能够代理主动基金经理类型；本地分类未必等于法律或经济实体分类，可能混入保险公司、企业战略持股、养老金或其他非传统基金主体。 |
| **H4｜Robustness**：结果能通过现实成本、PIT、参数扰动、样本外和多重检验。 | 保存完整 sweep；执行 48/12 月 walk-forward；计算 DSR；报告无效配置、覆盖、换手、成本和子时期。 | 样本内 vs OOS；单一最优格点 vs 参数邻域；原始 Sharpe vs DSR。 | 当前尝试次数必须完整计入；重复观察结果后再修改参数会增加未记录的 researcher degrees of freedom。 |

### 1.5 研究贡献

本项目的主要贡献不是某一条最高累计收益曲线，而是一套可以复核、反证和重复运行的研究基础设施。每次运行应记录：

- run timestamp、Git SHA 和 config hash；
- 输入数据与缓存哈希；
- filing/accession/amendment 版本；
- 经理和证券逐项筛选结果；
- idea、来源经理、聚合分数和最终权重；
- 交易、成本、持仓数量和风险约束；
- sweep、walk-forward、DSR 和失败配置。

---

## 2. 数据与样本

| 数据层 | 当前实现 | 主要限制 |
|---|---|---|
| 13F 申报 | SEC filings/datasets；保留 `period_date`、`filing_date`、accession 和 amendment | 不包含完整空头、现金、衍生品和季度内交易 |
| 证券映射 | CUSIP → OpenFIGI → ticker；缓存并输出覆盖诊断 | 映射缺失、错误映射和公司行动可能造成系统性样本偏差 |
| 收益数据 | Yahoo/yfinance 月度价格；本地缓存与 Chart API fallback | 不等同于包含退市收益和完整证券历史的 CRSP |
| 主动基准 | `manager_held_mcap`、外部历史 SPY/SPX 权重或可见 13F 聚合 | Yahoo historical shares 可能被修订，当前免费市值代理并非严格 vendor PIT |
| 风险模型 | Fama–French 因子；默认使用历史 CAPM 残差波动率 | 窗口、floor/cap 和 winsorization 属于研究假设 |
| 审计输出 | 映射、覆盖、剔除、经理、idea、持仓、换手、成本、规则和 manifest | 研究有效性仍依赖上游数据质量和同一 run 文件的一致性 |

### 2.1 经理人样本与三种分类模式

AUM、历史长度、持仓数、集中度、换手、PUT 暴露等属于通用 universe 条件；`manager_filter_mode` 决定是否在这些条件之外应用经理类型清洗。

| 模式 | 定义 | 保留范围 | 研究用途 |
|---|---|---|---|
| `all` | 不应用 manager classifier 或 manager allow/deny override。通用 universe 条件仍然生效。 | 所有满足 AUM、历史、持仓数、集中度等硬条件的申报主体。 | 未清洗基线；回答“简单使用全部合格 13F filers 会怎样”。 |
| `exclude_dirty` | 在 `all` 基础上排除明显不适合复制的主体或异常行为指纹。 | 可保留 dedicated、transient 和未分类主体，只要未被标记为 dirty。 | 测试基础数据清洗是否有价值，避免把 manager-selection 与过强风格筛选混在一起。 |
| `dedicated_like` | 先排除 dirty，再要求经理表现为集中、有限持仓宽度、低 ETF 暴露并具有一定持续性的主动选股型主体；同时服从 overrides。 | 行为上接近 dedicated stock picker 的经理。 | H3 的主要实验组。它是行为分类，不是经过 Form ADV 验证的法律实体分类。 |

`dirty` 的典型原因包括银行经营账户、broker/dealer 或 market maker、央行/主权主体、慈善/捐赠账户、ETF sponsor、养老金、异常宽度和高 ETF 暴露等。具体原因以 `manager_classification.csv` 和运行时配置为准。

> **重要限制**
>
> `dedicated_like` 不是“纯基金经理名单”。在接入 Form ADV、Bushee 或外部实体标签之前，分类结果可能仍包含保险公司、企业控股主体、大学基金或其他行为上类似集中经理的申报者。

### 2.2 证券样本

当前 live 默认排除 ETF、ETN 和其他 fund-like 持仓，并保留排除原因和价值覆盖诊断。证券缺少映射、市值或价格时，不填充为零：

- 缺少 CUSIP 映射：从可交易证券池剔除并记录；
- 缺少 active benchmark 市值：不把市值当成零，缩小 benchmark covered set 并记录覆盖率；
- 缺少价格：按 `missing_price_policy` 处理；
- 多类别股票：通过 `issuer_group` 合并计算 issuer exposure。

### 2.3 点时样本构造

每个再平衡月依次执行：

1. 仅保留 `filing_date <= rebalance_month` 的申报版本；
2. 为每位经理选择当时最新可见且未超过 filing/period staleness 上限的版本；
3. 应用经理 hard filters 和 `manager_filter_mode`；
4. 映射并过滤证券，计算 raw/investable book diagnostics；
5. 构造当月 active benchmark 和历史风险输入；
6. 计算经理内部信号并选择 Top-N ideas；
7. 进行跨经理聚合、共识过滤、持仓数量和风险上限处理；
8. 在月末设定下一期目标权重并记录交易成本。

---

## 3. 研究方法

### 3.1 Book weight 与 `manager_held_mcap`

对经理 $m$、股票 $i$ 和决策月 $t$，申报组合权重为：

```math
w^{book}_{i,m,t} = \frac{\text{reported value}_{i,m,t}}{\sum_{j \in S_{m,t}}\text{reported value}_{j,m,t}}
```

其中 $S_{m,t}$ 是该经理在当月可用于研究的长股票持仓集合。

`manager_held_mcap` 是一个**经理特定的 held-universe benchmark**：

1. 取该经理当期持有、通过证券过滤且存在当月市值的股票集合 $C_{m,t}$；
2. 对每只股票取得当月市场价值 $MC_{i,t}$；
3. 只在该经理自己的 covered holdings 内归一化：

```math
w^{mcap}_{i,m,t} = \frac{MC_{i,t}}{\sum_{j \in C_{m,t}}MC_{j,t}}
```

随后：

```math
\text{ActiveWeight}_{i,m,t} = w^{book}_{i,m,t} - w^{mcap}_{i,m,t}
```

因此，`manager_held_mcap`：

- 不是全市场市值指数；
- 不是 SPY/SPX 权重；
- 不是其他经理的 13F 聚合；
- 只回答：**在该经理实际持有的股票集合中，相对于按市值持有，他对某只股票超配了多少。**

缺少市值的股票不填成零；系统会审计 covered names、book coverage 和缺失规模。

该定义包含三个重要假设：

1. 经理当前持有的股票集合可以代理其投资机会集；
2. held-universe 市值权重是合理的中性配置；
3. 历史 shares 与价格具有足够的 PIT 质量。

当前 `yahoo_shares_proxy` 适合基础设施研究，但不等同于严格 PIT 的 CRSP/Compustat 市值。

### 3.2 Idea 信号

| `idea_signal` | 经理内部排名含义 |
|---|---|
| `level` | 当前申报 book weight |
| `change` | book weight 的季度变化 |
| `initiation` | 新建仓股票，按当前 book weight 排名 |
| `active_weight` | 当前 book weight 减 active benchmark weight |
| `active_weight_change` | active weight 的季度变化 |
| `active_weight_initiation` | 新建仓股票，按 active weight 排名 |
| `cps_ir` | 正 active weight × 历史 CAPM 残差波动率 |
| `cps_ir_change` | CPS 代理的季度变化 |
| `cps_ir_initiation` | 新建仓股票，按 CPS 代理排名 |

当前 CPS 风格代理为：

```math
\text{CPS}_{i,m,t} = \max(\text{ActiveWeight}_{i,m,t},0) \times \widehat{\sigma}^{idio}_{i,t}
```

$\widehat{\sigma}^{idio}_{i,t}$ 仅使用 $t$ 之前的月度收益估计。该公式并非论文参数的机械复刻；24 个月窗口、最少观测、floor/cap 和 winsorization 都属于工程假设。

### 3.3 Idea 选择与聚合

每位合格经理先按 `idea_signal` 排名并贡献 Top-N ideas。`min_consensus_funds` 随后要求一只股票至少被指定数量的**不同经理**选中。

#### `manager_equal`：每个经理人一票

设当月有 $M_t$ 位实际贡献经理。每位经理获得相同总预算：

```math
B_{m,t} = \frac{1}{M_t}
```

经理 $m$ 的 Top-K idea 集合为 $I_{m,t}$。其经理内部预算按这些股票的申报 book weight 归一化：

```math
q_{i,m,t} = \frac{w^{book}_{i,m,t}}{\sum_{j \in I_{m,t}}w^{book}_{j,m,t}}
```

股票 $i$ 的 cap 前聚合权重为：

```math
W^{precap}_{i,t} = \sum_{m:i\in I_{m,t}} \frac{1}{M_t}q_{i,m,t}
```

直观上：

- 每位经理只有**一票、同一份预算**；
- Top-K 只是该经理如何拆分这一票；
- 同一股票被三位经理选中，就获得三份经理预算的贡献；
- 经理的 AUM 大小和 CPS 分数绝对值不会直接放大其投票权；
- `min_consensus_funds=2` 表示至少需要两张独立经理票。

这更接近“汇总经理 best ideas”的论文启发式组合思想，但不是对原论文权重方法的严格复刻。

#### 其他聚合方式

- `score`：跨经理加总 signal score 后归一化。高 active weight、高 idio-vol 或多经理重叠都可能显著放大股票权重；属于实验性高确信度/高风险版本。
- `manager_count`：按选择该股票的不同经理人数加权，不考虑经理内 book weight。
- `equal_name`：所有通过筛选的最终股票等权。

排名、聚合和风险约束是三个独立步骤。`cps_ir` 可以仅用于排名，也可以在 `score` 模式下进一步影响最终权重；两者经济含义不同，必须分别报告。

### 3.4 组合约束

组合层支持：

- `min_consensus_funds`：最低独立经理票数；
- `min_portfolio_names` / `max_portfolio_names`：最少和最多股票数；
- `max_name_weight`：单 ticker 上限；
- `max_issuer_weight`：同 issuer 多类别股票合计上限；
- `holding_horizon_q`：退出目标后允许延续持有的季度数；
- `min_active_weight_holdings`：使用 active/CPS 信号前的经理 book 最低宽度；
- `missing_price_policy`：缺失价格处理；
- `bps_per_side`：单边交易成本。

### 3.5 回测与执行假设

1. 当月现有持仓先获得当月收益；
2. 当月月底根据当时可见 filing 计算新目标权重；
3. 新目标从之后的月份开始获得收益；
4. 交易成本按 `one_way_turnover × bps_per_side` 扣除；
5. 与 placebo、SPY 和多因子模型比较；
6. 保存逐月经理、idea、holding sources、持仓和交易审计。

该执行方式比按 `period_date` 立即交易更接近公开数据可实现性，但可能额外引入 filing date 到月底的等待。要分离 13F 原始信息和披露时滞，应使用日频价格构建 alpha decay curve。

### 3.6 评价指标

主要指标包括：

- 年化收益、波动率和 Sharpe；
- 相对 SPY 的 active return 与 information ratio；
- 当前可用因子列上的回归 alpha 与 t 值；live 数据完整时通常为 FF5+MOM 六因子；
- 最大回撤；
- 换手率和模型内交易成本；真实流动性与 capacity 尚未建模；
- 有效再平衡比例、持仓数量和 effective number；
- sweep 网格及相邻参数表现（需从完整 grid 联合分析）；
- walk-forward OOS active Sharpe；
- Deflated Sharpe Ratio（DSR）。

累计收益图只用于展示，不作为独立的策略有效性证据。

---

## 4. 实证结果

### 4.1 主结果：最多30只、10%/15%风险上限

主结果采用 `reports/20260620T163324Z`（git `02ec3472`），覆盖 2015-01 至 2026-05。这里的“30只”是**组合上限**，不是每月固定持有30只；受共识和最低持仓规则影响，实际平均约19.3只。

| 层级 | 参数 | 含义 |
|---|---|---|
| 经理样本 | AUM `$0.1B–$10B`，`dedicated_like` | 集中、低换手、行为上接近主动选股的经理 |
| 经理 hard filters | holdings ≤40；Top-10 ≥50%；低换手分位；PUT ≤5%；历史 ≥4Q | 提高经理信号的集中度和可解释性 |
| 主动基准 | `manager_held_mcap` | 在经理自己的持仓子集中，以市值权重作为中性配置 |
| Idea 信号 | `cps_ir`；每位经理 Top 3 | 正主动超配 × 历史 CAPM 残差波动率 |
| 共识与聚合 | 至少2位经理；`score` | 跨经理加总 CPS score |
| 组合宽度 | 最少10只、最多30只 | 相对分散的主研究组合 |
| 风险上限 | 单名10%；issuer 15% | 限制单一股票及多类别证券集中度 |
| Carry / 成本 | 0季度；单边15 bps | 每次重建目标，使用固定线性成本 |

| 指标 | 主结果 |
|---|---:|
| 年化收益 / 波动率 | 约 **17.0% / 19.4%** |
| Sharpe / 相对 SPY IR | 约 **0.79 / 0.29** |
| 六因子年化 alpha / t 值 | 约 **4.0% / 1.25** |
| 平均实际股票数 / effective number | 约 **19.3 / 14.9** |
| 平均单边月换手 | 约 **26.2%** |
| 平均最大单名权重 | 约 **9.0%** |
| 有效再平衡月 | **122/135**（约90.4%） |

该 run 使用 `--skip-sweep`，因此没有可报告的 walk-forward OOS 或 DSR；不能把参数表中的72个候选试验误写成已经完成的多重检验。

### 4.2 补充：5–10只高集中测试

随后运行的 `reports/20260620T180809Z` 将 AUM 上限收窄至 `$5B`，组合改为最少5只、最多10只，单名/issuer 上限提高至20%/25%，其他核心信号仍为 Top-3 `cps_ir`、至少2位经理共识和 `score` 聚合。

该高集中测试的年化收益约19.6%、波动率21.3%、Sharpe 0.83、相对 SPY IR 0.42、六因子 alpha 6.7%（t=1.54）；84个月 OOS active Sharpe 约0.58，DSR 约0.50。平均实际持股9.1只、effective number 7.5、平均单边月换手28.3%。点估计更高，但集中度和参数选择风险也更高，因此只作为补充敏感性测试，不作为项目主结果。

### 4.3 结果解释与组合差异

两次探索性运行都给出正向点估计，但仍不能确认稳定 alpha：主结果的 alpha t 值仅1.25且未完成 sweep/DSR；高集中补充测试的 t 值仅1.54、DSR 仅0.50。两者还共同依赖非严格 PIT 的 Yahoo shares proxy、行为式经理分类和固定15 bps成本。

| 维度 | 论文启发式经理投票 | 30只上限主结果 | 10只上限补充测试 |
|---|---|---|---|
| 经理 ideas | 严格 Best Idea 或较窄集合 | Top 3 CPS | Top 3 CPS |
| 聚合 | `manager_equal`，每位经理一票 | `score` | `score` |
| 组合上限 | 由经理集合自然形成 | 30只 | 10只 |
| 单名 / issuer 上限 | 不依赖极端 CPS magnitude | 10% / 15% | 20% / 25% |
| 研究用途 | 检验普遍 best-idea 效应 | 主研究组合 | 高集中敏感性测试 |

因此当前结论是：特定筛选和 score weighting 可产生值得继续验证的历史信号，但尚未通过统计显著性、严格 PIT、多重检验、现实成本和独立 holdout 的共同标准。

---

## 5. 稳健性与验收标准

### 5.1 H1：Best-idea effect

至少报告：

- Top 1、Top 3、Top 5；
- manager-equal 下的结果；
- Top ideas 对普通持仓/placebo 的增量；
- filing date 后不同等待期的 alpha decay；
- 子时期和危机期。

严格 H1 的主结果应来自 Top 1，而不是事后选择表现更好的 Top 3/5。

### 5.2 H2：Active-weight effect

至少比较：

- `level`；
- `active_weight`；
- `cps_ir`；
- `manager_held_mcap`、历史 SPY/SPX 和可见 13F 聚合基准；
- idio-vol power、窗口及风险乘数的消融。

若 `score` 版本有效而 manager-equal 版本无效，应明确区分“选股信息”和“高风险集中配置”的贡献。

### 5.3 H3：Manager-selection effect

固定同一信号和组合构造，比较：

- `all`；
- `exclude_dirty`；
- `dedicated_like`。

报告：

- 收益、alpha 和 OOS；
- 经理数量和有效贡献比例；
- 最新持仓重合度；
- 非传统实体占比；
- 清洗条件的 marginal IR。

### 5.4 Walk-forward 与多重检验

- 默认使用 48 个月训练、12 个月测试；
- 训练期只用于选参，测试期不回看；
- DSR 必须区分 generated、feasible 和 effective trials；
- 不应让结构性无效配置进入模型选择；
- 当前项目建议将 `DSR > 0.95` 作为强验收线，而不是以累计净值替代。

### 5.5 数据与交易验收

正式研究至少需要：

- 严格 PIT market cap；
- 退市收益和 survivorship-aware returns；
- amendment-safe 历史重建；
- Form ADV 或外部实体标签；
- ADV、bid–ask spread 和 market impact；
- 同一 run 的文件 provenance 检查；
- 真正未触碰的 holdout。

---

## 6. 后续研究方向

1. 构建日频 alpha-decay curve：报告期末、T+15、T+30、filing date、filing 后月底及再延迟一个月。
2. 使用 CRSP/WRDS 或同等级数据替代 Yahoo，加入退市收益和严格 PIT 市值。
3. 接入 Form ADV、Bushee 或外部实体标签，区分传统主动基金、保险账户、企业战略持股和养老金。
4. 对 `manager_held_mcap`、历史 SPY/SPX 和其他合理机会集基准做并列检验。
5. 将 active-weight 选股信息与 score-based 风险预算完全分离。
6. 研究 idio-vol exponent、score damping 和风险标准化，而不是默认线性放大高波动股票。
7. 加入真实流动性、capacity 和披露后拥挤交易成本。
8. 报告参数邻域、子时期、危机期和经理贡献集中度。
9. 固定当前候选策略，不再调参，在未来数据或真正 untouched holdout 上验证。
10. 将 13F 定位为慢速研究特征，与估值、盈利修正、价格动量和拥挤度联合使用。

---

## 7. 结论

本项目已经建立一套较完整的 13F 点时研究框架，能够区分报告日、披露日、再平衡日和收益实现期，并审计经理筛选、idea 来源、聚合、持仓、换手、成本和参数搜索。

当前仍无法确认公开 13F 本身具有稳定、可交易的 alpha。更准确的研究结论是：

> 13F 可能保留经理选股能力的弱痕迹，但公开披露的延迟、信息残缺和拥挤交易会显著减少可复制收益。30只上限主结果及更高集中的补充测试均产生了正向历史点估计，但尚未通过统计显著性、严格 PIT、多重检验、现实成本和独立留出样本的共同标准。

---

# 操作指南

## 项目功能

- 构建规则化 SEC 13F 经理人样本；
- 处理 filing-date 可见性与 amendment 版本；
- 通过 OpenFIGI 映射 CUSIP，并输出覆盖诊断；
- 下载和缓存月度价格、因子及市值代理；
- 计算 book weight、active weight 和 CPS 风格信号；
- 支持 `all`、`exclude_dirty`、`dedicated_like` manager universe；
- 运行 thesis/placebo、消融、parameter sweep、walk-forward 和 DSR；
- 输出 dashboard、交互报告、CSV 审计文件及 manifest。

## Quick start

### 环境

建议使用 Python 3.11+。

```powershell
python -m pip install pandas numpy scipy statsmodels matplotlib requests yfinance pandas_datareader pytest pyarrow
python -m pip install edgartools
```

创建本地 `.env`，不要提交到 Git：

```text
OPENFIGI_API_KEY=your_openfigi_key
```

Live SEC 下载前，将 `run_example.py` 中的 `LIVE_CONFIG["identity"]` 替换为真实姓名和邮箱。

### 运行

```powershell
# 离线 synthetic smoke test
python -B run_example.py --mode synthetic

# Live 数据链 smoke test
python -B run_example.py --mode live-smoke --smoke-cusips 300 --smoke-tickers 200

# 完整 live 研究
python -B run_example.py --mode live

# 快速诊断：跳过 marginal 与 sweep
python -B run_example.py --mode live --skip-marginal --skip-sweep

# 比较经理分类模式
python -B run_example.py --mode live --manager-filter-mode all --skip-marginal --skip-sweep
python -B run_example.py --mode live --manager-filter-mode exclude_dirty --skip-marginal --skip-sweep
python -B run_example.py --mode live --manager-filter-mode dedicated_like --skip-marginal --skip-sweep
```

测试：

```powershell
python -B -m pytest tests
```

## 主要文件

| 文件 | 作用 |
|---|---|
| `build_universe.py` | SEC 数据发现、解析、缓存和经理人样本构建 |
| `data_adapters.py` | OpenFIGI、Yahoo/yfinance、Fama–French 和覆盖诊断 |
| `engine.py` | 组合构造、PIT 回测、归因、成本和风险逻辑 |
| `sweep.py` | 参数网格、walk-forward 和 Deflated Sharpe |
| `manager_classifier.py` | PIT 经理行为分类 |
| `run_diagnostics.py` | Manager、value unit 和筛选诊断 |
| `report.py` | 静态 dashboard 和交互报告 |
| `run_example.py` | synthetic、smoke 和 live 主流程 |
| `data/security_overrides.csv` | 多类别证券的 issuer group override |
| `data/fund_ticker_exclusions.csv` | ETF/ETN/基金类 ticker 补充排除表 |
| `data/manager_overrides.csv` | 经理人 allow/deny override |

## 主要输出

运行在 `reports/` 下生成带时间戳的目录。以下文件按运行模式和开关生成；例如 `--skip-sweep` 不会产生完整 sweep 文件：

| 文件 | 内容 |
|---|---|
| `strategy_dashboard.png` | 策略总览 |
| `interactive_results.html` | 交互式 sweep 和逐月持仓报告 |
| `sweep_grid.csv` | 参数组合和统计结果 |
| `sweep_returns.csv` | 各配置逐月收益序列 |
| `manifest.json` | Git SHA、配置、输入哈希、覆盖和运行元数据 |
| `rebalance_summary_thesis.csv` | 再平衡级别摘要 |
| `rebalance_holdings_thesis.csv` | 股票级最终持仓 |
| `rebalance_holding_sources_thesis.csv` | 最终股票的来源经理和贡献 |
| `rebalance_ideas_thesis.csv` | 经理级 idea 与 signal 明细 |
| `rebalance_managers_thesis.csv` | 经理资格和贡献摘要 |
| `rebalance_manager_candidates_audit.csv` | 经理候选值、cutoff 和逐项 pass/fail |
| `rebalance_rules_thesis.json` | 运行规则和 active filter 状态 |
| `manager_characteristics_raw_investable.csv` | Raw 与 investable book 特征对照 |
| `manager_classification.csv` | 经理分类和 dirty 原因 |
| `manager_filter_acceptance.csv` | 经理清洗对组合和绩效的影响 |
| `active_benchmark_coverage_by_month.csv` | Active benchmark 月度覆盖 |
| `value_unit_diagnostics.csv` | AUM/value 单位连续性诊断 |

仅重建已有运行的 HTML：

```powershell
python -B report.py refresh reports/<run-directory>
```

## 配置原则

项目存在三类数值：

1. `engine.py` dataclass：通用 API 默认；
2. `run_example.py::_default_run_configs()`：当前 thesis/placebo recipe；
3. `manifest.json`：某一次运行真正使用的最终配置。

**任何结果均以对应 run 的 `manifest.json` 和 `rebalance_rules_thesis.json` 为准。** README 中的数值只用于解释和示例。

<details>
<summary><strong>30只上限主结果参数</strong></summary>

| 层级 | 参数 | 主结果值 | 解释 |
|---|---|---:|---|
| 经理分类 | `manager_filter_mode` | `dedicated_like` | 行为上接近集中主动选股型经理 |
| AUM | `min_aum`, `max_aum` | `$0.1B`, `$10B` | 决策时点经理 AUM 区间 |
| 历史 | `min_history_quarters` | `4` | 最少申报历史 |
| 时效 | `max_stale_filing_months` | `6` | Filing 公开时间最大陈旧月数 |
| 时效 | `max_stale_period_months` | `6` | 持仓报告期最大陈旧月数 |
| 集中度 | `use_concentration` | `True` | 启用集中度筛选 |
| 集中度 | `top_n_concentration` | `10` | 计算前十大权重 |
| 集中度 | `min_top_n_weight` | `50%` | Top-10 最低合计权重 |
| 持仓数 | `max_holdings` | `40` | 经理长股票持仓硬上限 |
| 换手率 | `use_low_turnover` | `True` | 启用低换手筛选 |
| 换手率 | `turnover_quantile` | `0.34` | 保留较低换手横截面 |
| 对冲 | `use_hedge_filter` | `True` | 启用 PUT 暴露筛选 |
| 对冲 | `hedge_put_max_weight` | `5%` | Filing PUT value 最大比例 |
| Value tilt | `use_value_tilt` | 配置可开 | 只有存在 PIT value scores 时才真正生效；运行规则应同时检查 `value_tilt_active` |
| Active share | `use_active_share` | `False` | Thesis 默认不作为 hard filter |
| Idea 信号 | `idea_signal` | `cps_ir` | 经理内部排名信号 |
| Idea 数量 | `top_n_ideas` | `3` | 每位经理选择 Top 3 |
| 聚合 | `idea_aggregation` | `score` | 跨经理加总 signal score |
| 共识 | `min_consensus_funds` | `2` | 至少两位不同经理选择 |
| 组合宽度 | `min_portfolio_names` | `10` | 少于10只则当月组合无效 |
| 组合宽度 | `max_portfolio_names` | `30` | 最多保留30只股票 |
| 单名上限 | `max_name_weight` | `10%` | 单 ticker 最终上限 |
| Issuer 上限 | `max_issuer_weight` | `15%` | 同 issuer 合计上限 |
| Carry | `holding_horizon_q` | `0` | 不额外保留已退出目标 |
| 信号有效性 | `min_active_weight_holdings` | `10` | Active/CPS 计算的经理 book 最低宽度 |
| 成本 | `bps_per_side` | `15 bps` | 单边线性交易成本 |
| 缺失价格 | `missing_price_policy` | `exit` | 退出无价格证券 |
| 主动基准 | `active_benchmark_source` | `manager_held_mcap` | 经理持仓子集市值权重 |

这些数值描述 `20260620T163324Z` 主结果，不是已经验证的永久默认。当前工作树中的 `_default_run_configs()` 已指向5–10只高集中补充测试，因此任何复现都应显式锁定配置并以 run manifest 为准。论文式对照应使用 `manager_equal`，并单独报告严格 Top-1 结果。

</details>

<details>
<summary><strong>Idea 信号与聚合方式</strong></summary>

| `idea_signal` | 排名含义 |
|---|---|
| `level` | 当前经理 book weight |
| `change` | book weight 的季度变化 |
| `initiation` | 新建仓，按当前 book weight 排名 |
| `active_weight` | book weight 减 active benchmark weight |
| `active_weight_change` | active weight 的季度变化 |
| `active_weight_initiation` | 新建仓，按 active weight 排名 |
| `cps_ir` | 正 active weight × CAPM 残差波动率 |
| `cps_ir_change` | CPS 代理的季度变化 |
| `cps_ir_initiation` | 新建仓，按 CPS 代理排名 |

| `idea_aggregation` | 权重含义 |
|---|---|
| `manager_equal` | 每位经理一票、同一总预算；经理内部按入选股票的申报 book weight 分配 |
| `score` | 跨经理加总 signal score 后归一化 |
| `manager_count` | 按选择该股票的不同经理人数加权 |
| `equal_name` | 最终存活股票等权 |

`consensus_weight` 是旧接口兼容项。显式设置 `idea_aggregation` 后，应以 `idea_aggregation` 为准。

</details>

<details>
<summary><strong>CPS 残差波动率参数</strong></summary>

| 参数 | 参考值 | 解释 |
|---|---:|---|
| `idio_vol_window_months` | `24` | CAPM 残差波动率滚动窗口（月） |
| `idio_vol_min_obs` | `12` | 最少有效月度观测 |
| `idio_vol_floor` | `10%` | 年化残差波动率下限 |
| `idio_vol_cap` | `80%` | 年化残差波动率上限 |
| `idio_vol_winsor_lower` | `5%` | 横截面下缩尾点 |
| `idio_vol_winsor_upper` | `95%` | 横截面上缩尾点 |

这些参数只用于稳定 CPS 排名输入，不是经过学术校准的常数。缓存或 manifest 内部可能简写为 `window_months`、`min_obs`；运行层配置以 `LIVE_CONFIG` 中的字段名为准。

</details>

<details>
<summary><strong>5–10只高集中补充测试 sweep</strong></summary>

最新高集中补充测试使用64组笛卡尔网格；它不是上述30只上限主结果的 sweep：

- `universe.aum_band`：`0.1–1B`、`0.1–5B`
- `portfolio.idea_signal`：`cps_ir`、`cps_ir_change`
- `portfolio.top_n_ideas`：`3`、`5`
- `portfolio.idea_aggregation`：`manager_equal`、`score`
- `portfolio.min_consensus_funds`：`1`、`2`
- `portfolio.holding_horizon_q`：`0`、`1`

固定项包括：

- `manager_filter_mode=dedicated_like`
- `min_portfolio_names=5`
- `max_portfolio_names=10`
- `min_active_weight_holdings=10`
- concentration 和 low-turnover filters 开启

Walk-forward 使用 48 个月训练、12 个月测试，以 active Sharpe 选参。生成配置数、可行配置数和有效独立试验数应分别报告。

</details>

<details>
<summary><strong>Live 数据与运行参数</strong></summary>

| 参数 | 默认/参考值 | 用途 |
|---|---:|---|
| `identity` | 占位字符串 | SEC 请求身份；live 前必须替换为真实姓名和邮箱 |
| `openfigi_key` | `None` | 可由本地环境变量提供 |
| `sec_history_start` | `2013-10-01` | SEC 申报历史起始日 |
| `price_history_start` | `2014-01-01` | 价格和因子 warm-up 起始日 |
| `start`, `end` | `2015-01-01`, `2026-05-31` | 默认研究区间 |
| `benchmark_ticker` | `SPY` | 市场收益基准 |
| ingest `min_aum`, `max_aum` | `$0.1B`, `$30B` | 原始抓取范围，与 thesis AUM 筛选不同 |
| ingest `max_holdings` | `40` | 抓取/基础 universe 持仓数上限 |
| ingest `max_put_weight` | `10%` | 宽松抓取边界；thesis 可使用更严格的5% |
| `require_factors` | `False` | 因子缺失是否必须中止 |
| `price_source` | `chart` | 可选 `chart`、`auto`、`yfinance` |
| `exclude_fund_like_holdings` | `True` | Live 默认排除 ETF/ETN/基金类行 |
| benchmark stale limit | `45 days` | Active benchmark 快照最大年龄 |
| market-cap shares stale limit | `550 days` | 免费 shares proxy 的最大允许陈旧期 |
| market-cap batch/workers/timeout | `25 / 6 / 20s` | 市值请求批量、并发和超时 |
| 缓存和 override 路径 | `LIVE_CONFIG` | OpenFIGI、价格、市值、经理分类和风险缓存 |

</details>

<details>
<summary><strong>主要命令行参数</strong></summary>

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--mode` | `synthetic` | `synthetic`、`live`、`live-smoke` |
| `--output-root` | `reports` | 输出目录 |
| `--smoke-cusips` | `300` | Smoke 模式映射 CUSIP 数量 |
| `--smoke-tickers` | `200` | Smoke 模式取价 ticker 数量 |
| `--skip-marginal` | off | 跳过 marginal-IR 消融 |
| `--skip-sweep` | off | 跳过参数网格和 walk-forward |
| `--equity-only` | off | 显式要求排除 ETF/ETN/基金类持仓；当前 live 配置默认已排除 |
| `--refresh-openfigi-metadata` | off | 刷新缺少分类元数据的映射 |
| `--price-source` | live config | `chart`、`auto`、`yfinance` |
| `--active-benchmark-source` | thesis/live config | `manager_held_mcap`、`visible_13f_aggregate`、`spy_holdings` |
| `--active-benchmark-weights` | none | 外部 benchmark 长表 |
| `--active-benchmark-max-stale-days` | live config | Benchmark 快照最大年龄 |
| `--manager-filter-mode` | thesis config | `all`、`exclude_dirty`、`dedicated_like` |
| `--sweep-checkpoint-every` | `5` | 每 N 组保存 sweep checkpoint；`0` 关闭 |

Fund-like 证券在当前 live 配置中默认排除；`--equity-only` 仍保留为显式开关和兼容接口。是否实际排除以对应 run 的 manifest/rules 为准。

</details>

## 数据源说明

### `manager_held_mcap`

Live 流程可根据历史 shares 和月末价格构建：

```text
data/processed/market_cap_history.parquet
```

免费来源标记为 `yahoo_shares_proxy`。更严格的 PIT 市值表应至少包含：

```text
month_end,ticker,market_cap,available_date,source,strict_pit
```

不得用当前 shares 或当前指数成分回填历史月份。

### 历史 SPY/SPX 权重

外部 benchmark 权重文件可使用 CSV、Parquet 或 XLSX：

```text
month_end,ticker,weight
2020-01-31,AAPL,0.045
2020-01-31,MSFT,0.038
```

运行示例：

```powershell
python -B run_example.py --mode live `
  --active-benchmark-source spy_holdings `
  --active-benchmark-weights data/processed/benchmark_weights_spy.parquet
```

## 已知限制

- Yahoo/yfinance 不适合发表级、退市敏感的最终研究；
- CUSIP/OpenFIGI 覆盖不完整，可能改变 AUM、持仓数和集中度；
- `missing_price_policy="exit"` 不是退市收益模型；
- Yahoo historical shares 可能被修订，`manager_held_mcap` 目前不是严格 PIT；
- 13F 不披露完整空头、现金、对冲和季度内交易；
- 经理分类尚未接入 Form ADV 或可靠外部实体标签；
- 固定15 bps成本未建模 bid–ask spread、ADV 和 market impact；
- Value tilt 只有在存在 PIT value score 时才真正生效；
- 多轮观察结果后的参数修改会增加真实多重检验负担；
- 高累计收益不能替代 alpha 显著性、DSR 和 untouched holdout；
- 同一研究包中的 CSV、dashboard 和 manifest 必须来自同一 run。

## Git hygiene

以下本地数据、密钥和生成文件默认不提交：

```text
.env
13f_cache/
reports/
artifacts/
openfigi_cache.parquet
yfinance_close_cache.parquet
yfinance_close_cache_coverage.parquet
```

## 参考文献

Cohen, R. B., Polk, C., and Silli, B., *Best Ideas*. SSRN abstract 1364827.
