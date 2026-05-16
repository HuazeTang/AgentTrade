"""Download all main-board non-ST A-share data to cache.

Usage: python download_all_main.py
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download")

START_DATE = date(2024, 6, 1)
END_DATE = date(2026, 5, 14)
BATCH_SIZE = 100  # write to cache after each batch for fault tolerance

print("=" * 60)
print("  全主板 A 股数据下载")
print(f"  区间: {START_DATE} ~ {END_DATE}")
print("=" * 60)

from data.sources.baostock import BaoStockSource
from data.pipeline import ensure_data
from data.cache import read_daily

source = BaoStockSource(rate_limit=0.05)
all_stocks = source.list_stocks()

# Filter: main board, non-ST
main = all_stocks[(all_stocks["board"] == "main_board") & (~all_stocks["is_st"])]
syms = sorted(main["symbol"].tolist())
print(f"\n主板非ST: {len(syms)} 只")
print(f"(全A共 {len(all_stocks)} 只)")

# Check cache
cached_df = read_daily(START_DATE, END_DATE)
cached_syms = set()
if not cached_df.empty:
    cached_syms = set(cached_df.index.get_level_values("symbol").unique())
to_fetch = [s for s in syms if s not in cached_syms]
print(f"已缓存: {len(cached_syms)} 只, 待下载: {len(to_fetch)} 只")

if not to_fetch:
    print("\n全部已缓存，无需下载。")
    source.close()
    exit(0)

total_elapsed = 0.0
total_done = 0
total_failed = 0

for i in range(0, len(to_fetch), BATCH_SIZE):
    batch = to_fetch[i : i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    total_batches = (len(to_fetch) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n--- 批次 {batch_num}/{total_batches}: {len(batch)} 只 ({i+1}-{min(i+BATCH_SIZE, len(to_fetch))}/{len(to_fetch)}) ---")
    t0 = time.time()

    try:
        result = ensure_data(batch, START_DATE, END_DATE, source=source)
        if not result.empty:
            batch_syms = result.index.get_level_values("symbol").nunique()
            batch_rows = len(result)
            total_done += len(batch)
            print(f"  完成: {batch_syms} 只, {batch_rows} 行")
        else:
            total_failed += len(batch)
            print(f"  警告: 批次返回空数据")
    except Exception as e:
        total_failed += len(batch)
        print(f"  错误: {e}")
        # Re-login on failure (connection may have died)
        try:
            source._logged_in = False
            source._login()
        except Exception:
            pass

    elapsed = time.time() - t0
    total_elapsed += elapsed

    # Update progress
    completed = len(cached_syms) + total_done
    pct = completed / len(syms) * 100
    remaining = len(to_fetch) - i - len(batch)
    if total_elapsed > 0 and total_done > 0:
        rate = total_done / total_elapsed
        eta = remaining / rate if rate > 0 else 0
        print(f"  耗时: {elapsed:.0f}s | 累计: {total_elapsed/60:.1f}min | 进度: {completed}/{len(syms)} ({pct:.1f}%) | 失败: {total_failed} | 预计剩余: {eta/60:.0f}min")

source.close()

# Final stats
print("\n" + "=" * 60)
print(f"下载完成!")
print(f"成功: {total_done} 只, 失败: {total_failed} 只")
print(f"总耗时: {total_elapsed/60:.1f} 分钟")

# Disk usage
cache_dir = Path("real/data/cache/daily")
total_mb = sum(f.stat().st_size for f in cache_dir.rglob("*.parquet")) / 1024 / 1024
print(f"缓存总大小: {total_mb:.1f} MB")
print("=" * 60)
