# AgentTrade 模拟撮合 + 链上结算示例

一个极简的「链下撮合、链上结算（模拟）」交易演示。适合科研仿真：几百到几万个 Agent 可以通过 HTTP 提交订单，撮合在内存完成，结算以伪 tx 记录。

## 快速启动

```bash
cd /Users/tanghuaze/projects/AgentTrade
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

浏览器打开 <http://127.0.0.1:8000/> 即可看到前端。

如果只想查看盘口/成交（无本地绘图），可以在新终端：

```bash
cd /Users/tanghuaze/projects/AgentTrade/frontend
python -m http.server 8001
```

然后访问 <http://127.0.0.1:8001/book_viz.html> ，默认从 <http://127.0.0.1:8000> 拉取数据。
成交回放：访问 <http://127.0.0.1:8001/book_replay.html> ，可加载 trades.json（时间轴回放），可选加载 book.json（盘口快照）和 agents.json（账户分布/极值）。
可视化概览（盘口 Top5、账户极值、分布）：打开 <http://127.0.0.1:8000/dashboard.html> （或用上面的 http server 访问 /dashboard.html）。

## 主要接口（链下撮合）

- `POST /agents`：注册 Agent，可指定初始余额/持仓。
- `POST /agents/{id}/deposit`：给 Agent 充值。
- `POST /orders`：下单 `{agent, side, price, quantity}`，`quantity` 必须为整数且 >=1；返回撮合出的成交及伪 tx，会校验余额/持仓。
- `GET /agents`：列出所有 Agent 余额/持仓（用于可视化、统计）。
- `GET /book?depth=5`：查看前 N 档买卖盘。
- `GET /trades?limit=20`：最近成交。
- `GET /agents/{id}`：查看某个 Agent 的现金/持仓。
- `GET /health`：健康检查。

撮合规则：价格-时间优先，买入撮合卖一（及以下），卖出撮合买一（及以上），成交价取对手与本方报价格中值；未成交部分挂簿等待。买单需有足够余额（按报单价*数量），卖单需有足够持仓。

## 前端

`frontend/index.html` 是一个纯静态页面，通过 `fetch` 调用上述接口，下单后每秒刷新订单簿、成交、账户状态。FastAPI 直接将该目录作为静态资源挂载。

## 模拟「链上结算」

真实链上交互未实现，仅用 `tx00000001` 这样的自增哈希表示链上结算记录，保留 cash/position 账本。若要替换为真实链上调用，可在 `backend/main.py` 的 `_settle` 中接入区块链 SDK。

## 压力与并发

该示例使用内存锁保证线程安全，适合单进程演示与本地压测（几百到几万并发请求）。生产或分布式环境需改为外部撮合服务、队列、持久化及真实链上结算。
