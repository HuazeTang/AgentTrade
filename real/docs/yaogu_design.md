# 妖股检测系统设计文档

## 1. 概述

两阶段级联检测架构：**Stage 1 LR Screener（高召回）→ Stage 2 DualTowerModel（高精度）**。

```
原始 OHLCV → 衍生特征 → Winsorize(1%/99%) → CS Z-Score(按日截面)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    Stage 1: Logistic Reg          Stage 2: DualTowerModel
    (Recall ≥ 95%)                 (Precision ≥ 30%)
              │                               │
              ▼                               ▼
       Candidate Pool                Final Yaogu Scores
       (~20-40% 全市场)
```

**设计目标**：海量负样本中精确定位妖股。Stage 1 用简单模型快速过滤，保证不丢正例；Stage 2 用深度学习在候选池内精判。

---

## 2. 数据 Pipeline

### 2.1 数据源

统一使用 **Tushare Pro** API（token 在 `.env` 的 `TUNSHARE_TOKEN`）：

| API | 用途 | 频率 |
|-----|------|------|
| `pro.daily(trade_date, adj=None)` | 未复权 OHLCV | 按日期批量，全市场 ~5500 条/日 |
| `pro.daily_basic(trade_date)` | 换手率、总市值 | 同上 |
| `pro.adj_factor(trade_date)` | 复权因子 | 同上 |
| `pro.stock_basic()` | 股票列表（含上市/退市日期） | 一次 |

### 2.2 缓存分层

```
data/cache/
├── daily_raw/          # 未复权原始数据 (源)
│   └── year=YYYY/month=MM/YYYYMM.parquet
├── daily_badj/         # 后复权 (raw × adj_factor)
│   └── year=YYYY/month=MM/YYYYMM.parquet
├── daily_qfq/          # 前复权 (raw / adj_factor)
│   └── year=YYYY/month=MM/YYYYMM.parquet
├── adj_factor.parquet  # 复权因子参考表 (ts_code, trade_date, adj_factor, symbol)
├── stock_list.parquet  # 股票基本信息 (symbol, name, board, list_date)
└── daily -> daily_badj # 软链接，训练/推理默认读取
```

### 2.3 复权公式

- **后复权**（默认使用）：`badj_price = raw_price × adj_factor`
- **前复权**：`qfq_price = raw_price / adj_factor`

复权因子来自 `adj_factor.parquet`，跨除权日收益率连续。5月25日除权高峰已验证。

### 2.4 关键模块

| 文件 | 功能 |
|------|------|
| `data/sources/tushare.py` | TushareSource 数据源，实现 DataSource ABC |
| `data/adjust.py` | `raw_to_badj()`, `raw_to_qfq()`, `rebuild_adjusted()` |
| `data/cache.py` | `read_daily()`, `write_daily(merge=True/False)` |
| `data/pipeline.py` | `ingest_daily()`, `ensure_data()` 编排层 |

---

## 3. 特征工程

### 3.1 衍生特征（14 维）

所有特征基于**后复权价格**计算，使用 `shift(1)` 避免未来信息泄露。特征值在 T 日反映的是 T-1 日及之前的信息。

| 特征 | 计算方式 | 类型 |
|------|---------|------|
| `ret_1d` | close.pct_change(1).shift(1) | 动量 |
| `ret_5d` | close.pct_change(5).shift(1) | 动量 |
| `ret_20d` | close.pct_change(20).shift(1) | 动量 |
| `vol_ratio_5d` | 5日均量 / 20日均量 | 量能 |
| `turnover_ratio` | 5日换手 / 20日换手 | 活跃度 |
| `amplitude_20d` | (20日最高 - 20日最低) / 20日均价 | 波动 |
| `close_position_5d` | mean((close-low)/(high-low)), 5日 | 价格位置 |
| `up_days_ratio_5d` | 5日上涨天数占比 | 趋势 |
| `vol_surge_5d` | 5日放量天数占比 (量 > 20日均量×1.5) | 异动 |
| `overnight_gap_5d` | mean((close(t-1) - pre_close) / pre_close), 5日 | 跳空 |
| `hl_ratio` | (high - low) / close | 日内振幅 |
| `amihud` | \|ret\| / amount × 10^6 | 流动性 |
| `vol_20d` | 20日收益率标准差 | 波动率 |
| `ret_vol_ratio_20d` | ret_20d / vol_20d | 风险调整收益 |

### 3.2 标准化 Pipeline

```
compute_derived_features(data)
  → winsorize_cross_sectional(df, lo=0.01, hi=0.99)   # 按日截面缩尾
  → cs_zscore_features(df)                              # 按日截面标准化: (x-daily_mean)/daily_std
  → fillna(0.0)
```

所有特征经过标准化后均值为 0、标准差为 1（每日截面），过滤日股票数 < 30 的交易日。

---

## 4. Stage 1: LR Screener（高召回筛选器）

### 4.1 模型

- **算法**：`sklearn.linear_model.LogisticRegression`
- **参数**：`solver="lbfgs"`, `C=1.0`, `class_weight="balanced"`, `max_iter=2000`
- **输入**：单日衍生特征向量（14 维），无序列信息
- **输出**：妖股概率 score ∈ [0, 1]

### 4.2 阈值调优

扫描 97 个阈值（0.02 ~ 0.98，步长 0.01），**从高到低**选择第一个满足 `recall ≥ 0.95` 的阈值。这样在保证召回的前提下，最小化候选池。

- 默认阈值：0.26
- 训练集召回：95.6%
- 验证集召回：93.2%
- 候选率：~80%（800 只股票 → ~640 只候选）

### 4.3 训练数据

`ScreenerDataset`：每条样本为单个 (trade_date, symbol)，标签定义为：

```python
y = 1 当且仅当:
  forward 10日最大累计收益 ≥ 30%  (min_cum_ret=0.30)
  AND 出现 ≥ 2 个连续涨停      (min_limit_up=2)
```

训练窗口 60 天历史 + 10 天前向标签窗口。

### 4.4 持久化

- `save(path)` / `load(path)` 通过 `joblib`
- 保存文件 ~1.3KB，包含 `LogisticRegression` 模型 + 阈值

---

## 5. Stage 2: DualTowerModel（高精度判别器）

### 5.1 架构

```
Input: (B, 60, 14)  # 60天序列，14维特征
       │
       ├──→ CNN Tower ──────────────→ 128-dim
       │    ├─ Conv1d(14→64, k=3, d=1)
       │    ├─ Conv1d(64→128, k=3, d=2)
       │    ├─ Conv1d(128→128, k=3, d=4)
       │    └─ AdaptiveAvgPool1d(1) → squeeze
       │
       └──→ Transformer Tower ──────→ 64-dim
            ├─ d_model=64, nhead=4, num_layers=2
            ├─ dim_feedforward=128, dropout=0.2
            ├─ GELU activation
            └─ Mean Pooling over sequence
       
       Concat → 192-dim → MLP Head → 1-dim (logit)
                          ├─ Linear(192, 128) → ReLU → Dropout(0.3)
                          ├─ Linear(128, 64)  → ReLU → Dropout(0.3)
                          └─ Linear(64, 1) → sigmoid
```

**设计思路**：
- **CNN Tower**：捕获局部价格形态（连续涨停、放量突破等），膨胀卷积增大感受野
- **Transformer Tower**：建模长程时序依赖（趋势、周期、量价背离）
- **双塔融合**：局部 + 全局特征互补，拼接后由 MLP 非线性组合

### 5.2 损失函数

**Focal Loss**（`dl/__init__.py:17`）：

```python
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

α = 0.85   # 正样本权重 85%，负样本 15%
γ = 2.0    # 聚焦因子，降低易分类样本的梯度贡献
```

**设计考量**：正负样本极度不平衡（~5% 正例）。Focal Loss 双重处理：
- α=0.85：训练时正样本 loss 放大 5.7x（0.85/0.15），强制模型关注妖股
- γ=2.0：预测置信度高的负样本（p≈0）loss 接近 0，模型不被海量易分类负样本淹没

### 5.3 训练配置

| 参数 | 值 | 说明 |
|------|------|------|
| 优化器 | AdamW | weight_decay=1e-4 |
| 学习率 | 1e-3 | ReduceLROnPlateau, factor=0.5, patience=10 |
| Batch Size | 2048 | |
| 梯度裁剪 | max_norm=1.0 | |
| 序列长度 | 60 交易日 | ~3 个月 |
| 前向窗口 | 10 交易日 | 标签计算窗口 |
| 负样本比例 | max 20:1 | train 和 val 均限制 |
| Early Stopping | patience=20 | 监控 val_precision |
| 保存指标 | val_precision | 非 val_f1，优先保证精度 |

### 5.4 标签定义（与 Screener 一致）

```python
y = 1 当且仅当:
  forward 10日最大累计收益 ≥ 30%
  AND 出现 ≥ 2 个连续涨停
```

### 5.5 阈值校准

`find_precision_threshold`：扫描 95 个阈值（0.05~0.99），选取 precision ≥ target_precision(0.3) 且 tp ≥ min_tp(5) 的最高阈值。避免 fp 过多导致实际使用效果差。

---

## 6. 两阶段级联推理

### 6.1 流程

```
1. 全市场股票 (target_date)
       │
2. 构建特征缓存 (build_normalized_feature_cache)
       │
3. Stage 1: Screener.score_day(feature_cache, target_date, all_symbols)
       │  返回每个股票的 LR 概率
       │  过滤: score ≥ screener.threshold (0.26)
       ▼
4. Candidate Pool (~640 stocks, 80% market)
       │
5. Stage 2: 对每个候选股票提取 60 天特征序列
       │  DualTowerModel.predict_proba(sequence)
       │  过滤: prob ≥ dl_threshold (calibrated)
       ▼
6. Final Yaogu Scores (precision ≥ 30%)
```

### 6.2 因子注册

`TwoStageYaoguFactor`（`dl/two_stage_factor.py`）实现 `Factor` 基类，注册为 `yaogu_two_stage`，可直接用于回测系统。

---

## 7. 文件结构

```
dl/
├── __init__.py           # DualTowerModel, FocalLoss, CNNTower, TransformerTower
├── derived_features.py   # 衍生特征计算 + 标准化 pipeline
├── dataset.py            # YaoguDataset (序列样本), build_dataloaders
├── screener.py           # YaoguScreener (LR 筛选器)
├── screener_dataset.py   # ScreenerDataset (单日样本)
├── train.py              # train(), train_two_stage() 训练入口
├── train_screener.py     # train_screener() 独立训练入口
├── run.py                # CLI 入口 (python -m dl.run)
├── factor.py             # 单阶段因子
└── two_stage_factor.py   # 两阶段因子

data/
├── cache.py              # read_daily(), write_daily()
├── pipeline.py           # ingest_daily(), ensure_data()
├── adjust.py             # raw_to_badj(), raw_to_qfq(), rebuild_adjusted()
└── sources/
    ├── base.py           # DataSource ABC
    ├── tushare.py        # TushareSource (统一数据源)
    ├── baostock.py       # BaoStockSource (备用)
    └── akshare.py        # AkshareSource (备用)

data/models/
├── yaogu_screener.joblib    # Stage 1 (旧)
├── yaogu_screener_v2.joblib # Stage 1 (新数据)
├── yaogu_best.pt            # Stage 2 best (旧)
├── yaogu_v2_best.pt         # Stage 2 best (新数据, 训练中)
└── yaogu_{run_id}_ep{N}.pt # 每轮 checkpoint
```

---

## 8. 使用方式

### 训练

```bash
# 两阶段训练（推荐）
python -m dl.run \
  --mode two-stage \
  --train-start 2020-01-01 --train-end 2023-12-31 \
  --val-start 2024-01-01 --val-end 2024-12-31 \
  --epochs 100 --batch-size 2048 \
  --target-precision 0.3 \
  --save data/models/yaogu_v2_best.pt \
  --screener-save data/models/yaogu_screener_v2.joblib

# 仅训练 Screener
python -m dl.run --mode screener-only --recall-target 0.95

# 单阶段（不使用 Screener）
python -m dl.run --mode single --epochs 100
```

### 下载数据

```python
from data.sources.tushare import TushareSource
from datetime import date

source = TushareSource()
raw = source.fetch_daily(symbols, date(2026, 5, 1), date(2026, 5, 28))
# raw 为未复权数据，通过 adjust.py 转换为后复权
```

### 推理

```python
from dl.two_stage_factor import TwoStageYaoguFactor

factor = TwoStageYaoguFactor(
    screener_path="data/models/yaogu_screener_v2.joblib",
    dl_path="data/models/yaogu_v2_best.pt",
)
scores = factor.compute(daily_cache)
```

---

## 9. 已知问题与改进方向

1. **Screener 候选率偏高**（80%）：当前 LR 模型区分度有限，仅过滤 20% 股票。可尝试 XGBoost 或更多特征。
2. **Recall-Precision 权衡**：训练中 precision 提升伴随 recall 骤降（tp 从 188 → 15）。可能需要调整 Focal Loss 的 α 或尝试 label smoothing。
3. **标签定义**：30%/2连板 可能漏掉慢牛型妖股（一个月涨 50% 但无连续涨停）。可考虑多标签方案。
4. **ST 股票**：当前未排除 ST，*ST 股票涨跌幅 5% 不同于正常的 10%，涨停检测需要区分。
5. **复权数据维护**：每次下载新数据后需同步更新 `adj_factor.parquet`，确保后复权收益连续。
