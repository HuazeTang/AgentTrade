"""
Visualize order book depth from live API or a saved book.json.

Usage examples:
  # Live view, refresh every 1s, depth 20
  python viz_book.py --live --interval 1 --depth 20

  # Visualize saved sim_logs/book.json
  python viz_book.py --file sim_logs/book.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import requests

API = "http://127.0.0.1:8000"


def fetch_book(depth: int = 20) -> Dict:
    resp = requests.get(f"{API}/book", params={"depth": depth}, timeout=5)
    resp.raise_for_status()
    return resp.json()


def load_book_file(path: str) -> Dict:
    data = json.loads(Path(path).read_text())
    # In case file structure nests under "book"
    if "bids" not in data and "book" in data:
        data = data["book"]
    return data


def render_book(ax, book: Dict):
    ax.clear()
    bids = sorted(book.get("bids", []), key=lambda x: x["price"], reverse=True)
    asks = sorted(book.get("asks", []), key=lambda x: x["price"])

    bid_prices = [b["price"] for b in bids]
    bid_qtys = [b["quantity"] for b in bids]
    ask_prices = [a["price"] for a in asks]
    ask_qtys = [a["quantity"] for a in asks]

    if bid_prices:
        ax.barh(bid_prices, bid_qtys, color="#3ccf91", alpha=0.7, label="Bids")
    if ask_prices:
        ax.barh(ask_prices, ask_qtys, color="#ff6b6b", alpha=0.7, label="Asks")

    mid = None
    if bid_prices and ask_prices:
        mid = (bid_prices[0] + ask_prices[0]) / 2
        ax.axhline(mid, color="#7cb7ff", linestyle="--", linewidth=1, label=f"Mid ~ {mid:.2f}")

    ax.set_xlabel("Quantity")
    ax.set_ylabel("Price")
    ax.set_title("Order Book Depth")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.invert_yaxis()  # higher prices at top


def live_view(depth: int, interval: float):
    fig, ax = plt.subplots(figsize=(7, 6))
    plt.ion()
    while True:
        try:
            book = fetch_book(depth=depth)
            render_book(ax, book)
            plt.pause(interval)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"[warn] fetch/render failed: {exc}")
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Visualize order book depth.")
    parser.add_argument("--live", action="store_true", help="Fetch from API continuously.")
    parser.add_argument("--file", type=str, help="Path to saved book JSON.")
    parser.add_argument("--depth", type=int, default=20, help="Depth to fetch/display.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between refresh (live).")
    args = parser.parse_args()

    if not args.live and not args.file:
        parser.error("must specify --live or --file")

    if args.live:
        live_view(depth=args.depth, interval=args.interval)
    else:
        book = load_book_file(args.file)
        fig, ax = plt.subplots(figsize=(7, 6))
        render_book(ax, book)
        plt.show()


if __name__ == "__main__":
    main()

