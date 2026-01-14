"""
Simple simulation driver:
- Registers N agents with equal initial cash/position
- Lets them trade around a target mid price (default 100)
- Keeps local logs of all submitted orders and server trade history

Prereq: start the API locally:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
Then run:
    python simulate_agents.py
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import requests

API = "http://127.0.0.1:8000"
LLM_API_KEY = ""  # Placeholder for Tongyi/Qwen; left empty for fallback heuristic


@dataclass
class OrderDecision:
    side: Literal["buy", "sell"]
    price: float
    quantity: int


def llm_or_heuristic_decision(agent: str, mid: float = 100.0) -> OrderDecision:
    """
    Placeholder for LLM-driven strategy. If LLM_API_KEY is set, replace this body
    with a call to Tongyi/Qwen to decide side/price/qty. For now, use a simple
    stochastic strategy around the target mid price.
    """
    # Heuristic: small random skew around mid, alternate sides
    side = random.choice(["buy", "sell"])
    price = round(random.normalvariate(mid, 1.5), 2)
    price = max(0.1, price)
    qty = random.randint(1, 10)
    return OrderDecision(side=side, price=price, quantity=qty)


def register_agent(agent: str, cash: float, position: float) -> None:
    resp = requests.post(
        f"{API}/agents",
        json={"agent": agent, "initial_cash": cash, "initial_position": position},
        timeout=10,
    )
    if resp.status_code not in (200, 400):
        raise RuntimeError(f"register {agent} failed: {resp.text}")
    if resp.status_code == 400 and "already exists" not in resp.text:
        raise RuntimeError(f"register {agent} failed: {resp.text}")


def place_order(agent: str, decision: OrderDecision) -> dict:
    payload = {
        "agent": agent,
        "side": decision.side,
        "price": decision.price,
        "quantity": decision.quantity,
    }
    resp = requests.post(f"{API}/orders", json=payload, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"order failed for {agent}: {resp.text}")
    return resp.json()


def fetch_book(depth: int = 8) -> dict:
    return requests.get(f"{API}/book", params={"depth": depth}, timeout=10).json()


def fetch_trades(limit: int = 100000) -> dict:
    return requests.get(f"{API}/trades", params={"limit": limit}, timeout=10).json()


def fetch_agent(agent: str) -> dict:
    return requests.get(f"{API}/agents/{agent}", timeout=10).json()


def fetch_agents_all() -> dict:
    return requests.get(f"{API}/agents", timeout=10).json()


def run_simulation(
    agent_count: int = 100,
    initial_cash: float = 100_000.0,
    initial_position: float = 1_000.0,
    duration_sec: float = 60.0,
    target_mid: float = 100.0,
    orders_per_sec: float = 20.0,
    log_dir: str = "sim_logs",
    sample_agent: str = "agent-001",
    snapshot_interval: float = 0.0,  # 0 表示每次下单后都抓快照
):
    print(f"Registering {agent_count} agents...")
    for i in range(agent_count):
        aid = f"agent-{i+1:03d}"
        register_agent(aid, initial_cash, initial_position)
    print("Registration done.")

    orders_log: List[dict] = []
    gap = 1.0 / orders_per_sec if orders_per_sec > 0 else 0.0
    book_series: List[dict] = []
    agents_series: List[dict] = []
    last_snapshot = 0.0
    last_trade_ts = None

    print(f"Running for {duration_sec}s at ~{orders_per_sec} orders/s around mid={target_mid} ...")
    start = time.time()
    n = 0
    while time.time() - start < duration_sec:
        aid = f"agent-{(n % agent_count) + 1:03d}"
        decision = llm_or_heuristic_decision(aid, mid=target_mid)
        trade_ts = None
        try:
            res = place_order(aid, decision)
            # capture server trade timestamp if present (better alignment than client time)
            if res.get("trades"):
                trade_ts = max(t.get("ts", 0) for t in res["trades"])
            orders_log.append(
                {
                    "agent": aid,
                    "decision": decision.__dict__,
                    "response": res,
                    "ts": time.time(),
                }
            )
        except Exception as exc:  # keep going even if some orders reject (e.g., balance)
            orders_log.append(
                {
                    "agent": aid,
                    "decision": decision.__dict__,
                    "error": str(exc),
                    "ts": time.time(),
                }
            )
        n += 1

        now = time.time()
        need_snap = snapshot_interval == 0 or now - last_snapshot >= snapshot_interval
        if need_snap:
            # capture book and agents with timestamp; if本次有成交，优先用本次成交 ts，否则用当前时间
            snap_ts = trade_ts or now
            try:
                book_snap = fetch_book(depth=10)
                book_series.append({"ts": snap_ts, "bids": book_snap.get("bids", []), "asks": book_snap.get("asks", [])})
            except Exception as exc:
                orders_log.append({"agent": aid, "decision": decision.__dict__, "error": f"book snapshot failed: {exc}", "ts": now})
            try:
                agents_snap = fetch_agents_all()
                agents_series.append({"ts": snap_ts, "agents": agents_snap.get("agents", [])})
            except Exception as exc:
                orders_log.append({"agent": aid, "decision": decision.__dict__, "error": f"agents snapshot failed: {exc}", "ts": now})
            last_snapshot = now

        if gap:
            time.sleep(gap)

    print("Simulation window completed. Fetching final state...")
    trades = fetch_trades(limit=50000)

    sample_state = fetch_agent(sample_agent)

    print("\n=== Summary ===")
    print(f"Total orders sent: {len(orders_log)}")
    print(f"Total trades on server: {len(trades.get('trades', []))}")
    print(f"Sample agent {sample_agent} state: {sample_state}")
    if trades.get("trades"):
        print("Last 3 trades:")
        for t in trades["trades"][-3:]:
            print(t)

    out_dir = Path(log_dir)
    out_dir.mkdir(exist_ok=True)
    Path(out_dir / "orders.json").write_text(json.dumps(orders_log, indent=2))
    Path(out_dir / "trades.json").write_text(json.dumps(trades, indent=2))
    Path(out_dir / "book_series.json").write_text(json.dumps({"book_series": book_series}, indent=2))
    Path(out_dir / "agents.json").write_text(json.dumps({"agents_series": agents_series}, indent=2))
    print(f"\nLogs written to {out_dir}/orders.json, trades.json, book_series.json, agents.json")


def parse_args():
    p = argparse.ArgumentParser(description="Continuous agent trading simulation.")
    p.add_argument("--agents", type=int, default=100, help="Number of agents to register")
    p.add_argument("--initial-cash", type=float, default=100_000.0, help="Initial cash per agent")
    p.add_argument("--initial-pos", type=float, default=1_000.0, help="Initial position per agent")
    p.add_argument("--duration", type=float, default=60.0, help="Duration in seconds to run")
    p.add_argument("--ops", type=float, default=20.0, help="Orders per second target")
    p.add_argument("--mid", type=float, default=100.0, help="Target mid price to trade around")
    p.add_argument("--log-dir", type=str, default="sim_logs", help="Directory to store logs")
    p.add_argument("--sample-agent", type=str, default="agent-001", help="Agent id to sample state")
    return p.parse_args()


def main():
    args = parse_args()
    run_simulation(
        agent_count=args.agents,
        initial_cash=args.initial_cash,
        initial_position=args.initial_pos,
        duration_sec=args.duration,
        target_mid=args.mid,
        orders_per_sec=args.ops,
        log_dir=args.log_dir,
        sample_agent=args.sample_agent,
    )


if __name__ == "__main__":
    main()

