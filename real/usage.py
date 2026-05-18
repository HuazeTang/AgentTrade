#!/usr/bin/env python3
"""
AgentTrade 量化交易系统 — 使用指南

本文件列出了所有可用命令及参数说明，可直接复制执行。
前置条件：data/cache/daily/ 下有股票日线数据（baostock前复权），
          data/results/gp_factors.json 存在（用于 --load-gp）。

核心脚本: run_agent_simulation.py
辅助脚本: compare_weekdays.py, run_agent_report.py, run_backtest.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 每日选股推荐（明天买什么）
# ═══════════════════════════════════════════════════════════════════════════════
"""
# 基础推荐（Top 15，10万资金）
python run_agent_simulation.py --recommend --recommend-top 15 --recommend-cash 100000

# 加载 GP 因子（推荐）
python run_agent_simulation.py --recommend --recommend-top 15 --recommend-cash 100000 --load-gp

# 输出: 排名、代码、综合得分、最新收盘价、涨跌幅、建议股数、估算成本、top因子
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 单只股票深度分析
# ═══════════════════════════════════════════════════════════════════════════════
"""
# 生成 K 线图 + 因子走势 + 综合得分
python run_agent_simulation.py --analyze 603629

# 带 GP 因子
python run_agent_simulation.py --analyze 603629 --load-gp

# 输出: 多面板 PNG 图表 (data/results/sim_*/analysis_*.png)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 回测 / 仿真
# ═══════════════════════════════════════════════════════════════════════════════
"""
# --- 因子驱动回测（默认区间 2025-10-01 → 2026-04-30）---
python run_agent_simulation.py --mode factor

# 同上 + 加载已保存的 GP 因子
python run_agent_simulation.py --mode factor --load-gp

# 自定义区间 + 资金
python run_agent_simulation.py --mode factor --load-gp \
    --start 2025-10-01 --end 2026-05-18 --cash 100000

# --- GP 因子发现 + 回测（从头跑遗传规划，耗时长）---
python run_agent_simulation.py --mode factor --use-gp \
    --gp-population 200 --gp-generations 25

# --- LLM Agent 回测（需 Qwen API Key）---
python run_agent_simulation.py --mode llm --model qwen-max

# --- 对比模式: LLM vs Factor 同屏对比 ---
python run_agent_simulation.py --mode compare --model qwen-max

# 输出:
#   逐日权益、交易记录
#   最终: 累计收益率、年化收益、夏普比率、最大回撤、胜率
#   ablation: 基准 vs +GP 因子对比
#   K线图: data/results/sim_*/charts/
#   权益曲线: data/results/sim_*/equity_position.png
#   报告: data/results/sim_*/report.md
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 调仓日对比（周一~周五哪个最好）
# ═══════════════════════════════════════════════════════════════════════════════
"""
python compare_weekdays.py

# 输出: 每个周日的收益率、最大回撤、夏普、调仓次数对比表
# 结论: A股周三调仓最优（"黑色星期四"效应）
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 完整研报生成（含 GP 因子发现）
# ═══════════════════════════════════════════════════════════════════════════════
"""
# 跑完整流程: 数据加载 → 基准因子 → GP发现 → 验证 → 回测对比 → 图表 → 研报
python run_agent_report.py

# 输出: data/results/report_*/report.md + 4张图表
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 6. 数据下载（批量缓存全A股日线）
# ═══════════════════════════════════════════════════════════════════════════════
"""
# 默认下载全部A股日线到 data/cache/daily/
python download_all_main.py

# 输出: data/cache/daily/year=YYYY/month=MM/*.parquet (前复权日线)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 常用参数汇总
# ═══════════════════════════════════════════════════════════════════════════════
"""
  --start DATE           回测开始日期 (默认 2025-10-01)
  --end DATE             回测结束日期 (默认 2026-04-30)
  --cash FLOAT           初始资金 (默认 10,000)
  --mode {llm,factor,compare}
                         决策模式: llm=AI代理, factor=因子驱动, compare=两者对比
  --model MODEL          LLM模型名 (如 qwen-max)
  --use-gp               从头跑GP因子发现 (耗时长，约30分钟)
  --load-gp              加载已保存的GP因子 (推荐，秒级)
  --gp-population N      GP种群大小 (默认200)
  --gp-generations N     GP最大代数 (默认25)
  --recommend            选股推荐模式 (基于最新数据，假定空仓)
  --recommend-cash FLOAT 推荐模式可用资金 (默认10,000)
  --recommend-top N      推荐前N只 (默认3)
  --analyze SYMBOL       单只股票深度分析 (如 002384)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 关键配置 (run_agent_simulation.py 顶部)
# ═══════════════════════════════════════════════════════════════════════════════
"""
  SYMBOL_COUNT       = 800    # 股票池大小
  REBALANCE_FREQ     = "weekly"  # 调仓频率: daily/weekly/monthly
  INITIAL_CASH       = 10_000    # 默认初始资金
  MAX_POSITIONS      = 1         # 最大持仓数
  STOP_LOSS_PCT      = None      # 止损线 (未启用)
  TAKE_PROFIT_PCT    = 0.25      # 止盈线 25%

  周调仓 = 每周三 (calendar dayofweek==2)
  股票池 = 行业分层流动性选股 (非ST, 非北交所)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 项目文件结构
# ═══════════════════════════════════════════════════════════════════════════════
"""
  run_agent_simulation.py  核心: 仿真/选股/分析 (主要使用)
  compare_weekdays.py      调仓日对比 (周三最优验证)
  run_agent_report.py      完整研报生成 (含GP发现)
  run_backtest.py          端到端回测 (旧版，独立使用)
  download_all_main.py     批量下载A股日线
  data/cache/daily/        日线缓存 (parquet, 前复权)
  data/results/            回测/推荐/报告输出
  data/results/gp_factors.json  已保存的GP因子
  discovery/gp.py          遗传规划因子发现引擎
  factor/engine.py         因子计算引擎
  config/chart_style.py    图表样式 (中文字体 + 颜色)
"""

if __name__ == "__main__":
    print(__doc__)
