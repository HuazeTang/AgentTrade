"""Shenwan (申万) L1 industry classification for A-shares.

Uses akshare's stock_industry_clf_hist_sw() which downloads the official
Shenwan industry Excel. One HTTP call covers all stocks — no per-stock API.
First download is cached to disk; subsequent calls use the cache.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).resolve().parent / "_industry_cache.json"

# Shenwan L1 code (first 2 digits of 6-digit industry code) → name.
# Source: akshare stock_industry_category_cninfo(symbol="申银万国行业分类标准")
_L1_CODE_TO_NAME: dict[str, str] = {
    "11": "农林牧渔",
    "21": "采掘",
    "22": "基础化工",
    "23": "钢铁",
    "24": "有色金属",
    "25": "建筑材料",
    "26": "机械设备",
    "27": "电子",
    "28": "汽车",
    "31": "休闲服务",
    "32": "电气设备",
    "33": "家用电器",
    "34": "食品饮料",
    "35": "纺织服饰",
    "36": "轻工制造",
    "37": "医药生物",
    "41": "公用事业",
    "42": "交通运输",
    "43": "房地产",
    "45": "商贸零售",
    "46": "社会服务",
    "47": "计算机",
    "48": "银行",
    "49": "非银金融",
    "51": "综合",
    "61": "建筑材料",
    "62": "建筑装饰",
    "63": "电力设备",
    "64": "机械设备",
    "65": "国防军工",
    "71": "计算机",
    "72": "传媒",
    "73": "通信",
    "74": "煤炭",
    "75": "石油石化",
    "76": "环保",
    "77": "美容护理",
}


def fetch_shenwan_industry(as_of: date | None = None) -> pd.DataFrame:
    """Fetch latest Shenwan L1 industry for every A-share.

    Downloads the official Shenwan industry classification Excel via akshare
    and selects the most recent classification per stock.
    First call is cached to disk; subsequent calls return cached data.

    Args:
        as_of: If given, uses classification valid on or before this date.

    Returns:
        DataFrame with columns: symbol, industry_code, industry_name, l1_code.
        ``symbol`` is the raw 6-digit code (e.g. '600519'), no exchange prefix.
    """
    # ── Return from disk cache if available ─────────────────────────────
    if _CACHE_PATH.exists():
        try:
            cached = json.loads(_CACHE_PATH.read_text())
            df = pd.DataFrame(cached)
            if as_of is not None:
                as_of_ts = pd.Timestamp(as_of)
                df = df[df["start_date"] <= str(as_of_ts)].copy()
            logger.info("Industry map loaded from disk cache (%d stocks)", len(df))
            return df
        except Exception:
            _CACHE_PATH.unlink(missing_ok=True)

    import akshare as ak

    raw = ak.stock_industry_clf_hist_sw()
    # Columns: symbol, start_date, industry_code, update_time

    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        raw = raw[raw["start_date"] <= as_of_ts].copy()

    # Keep latest classification per symbol (by update_time then start_date)
    df = raw.sort_values(["update_time", "start_date"]).groupby("symbol").last()
    df = df.reset_index()

    # Map L1 code → name
    df["l1_code"] = df["industry_code"].str[:2]
    df["industry_name"] = df["l1_code"].map(_L1_CODE_TO_NAME).fillna("综合")

    result = df[["symbol", "industry_code", "industry_name", "l1_code"]]

    # ── Persist to disk cache ────────────────────────────────────────────
    try:
        _CACHE_PATH.write_text(result.to_json(orient="records", force_ascii=False))
        logger.info("Industry map cached to disk (%d stocks)", len(result))
    except Exception:
        pass

    return result


def build_industry_map(symbols: list[str] | None = None) -> dict[str, str]:
    """Return {symbol: industry_name} mapping for the given symbols.

    Symbols can be with or without exchange prefix (e.g. 'sh.600519' or '600519').

    If symbols is None, returns the mapping for the entire A-share market.
    """
    df = fetch_shenwan_industry()
    symbol_map: dict[str, str] = {}
    for _, row in df.iterrows():
        symbol_map[row["symbol"]] = row["industry_name"]

    if symbols is not None:
        # Normalise symbols: strip prefix for lookup
        stripped = {s.split(".")[-1] if "." in s else s: s for s in symbols}
        result: dict[str, str] = {}
        for raw_code, original in stripped.items():
            name = symbol_map.get(raw_code, "综合")
            result[original] = name
        return result

    return symbol_map
