"""Extract positive (妖股) and negative (假启动) matched pairs from historical data.

Core idea: for each stock that launched (妖股), find stocks on the same day with
similar pre-launch patterns that did NOT launch. This creates contrastive pairs
for the LLM to analyze.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Config ──
FORWARD_WINDOW = 10           # N days forward to check for launch
MIN_CUMULATIVE_RET = 0.30     # 30% minimum run-up
MIN_LIMIT_UP_DAYS = 2         # at least 2 consecutive limit-up days
PRE_LAUNCH_WINDOW = 20        # lookback days for pre-launch patterns
NEGATIVES_PER_POSITIVE = 3    # matched negative samples per positive
MAX_NEG_FORWARD_RET = 0.10    # negatives must have < 10% forward return
STOCK_POOL_BEFORE_LAUNCH = 5  # skip first N days (need pre-launch history)


@dataclass
class CasePair:
    """A matched pair: one 妖股 and one similar-looking stock that didn't launch."""
    symbol_pos: str
    symbol_neg: str
    launch_date: date
    pos_metrics: dict
    neg_metrics: dict
    pos_forward_ret: float
    neg_forward_ret: float
    diff_metrics: dict  # pos - neg for key metrics


class YaoguCaseExtractor:
    """Extract 妖股 cases and matched 假启动 controls from daily cache."""

    def __init__(
        self,
        forward_window: int = FORWARD_WINDOW,
        min_cum_ret: float = MIN_CUMULATIVE_RET,
        min_limit_up: int = MIN_LIMIT_UP_DAYS,
        pre_window: int = PRE_LAUNCH_WINDOW,
        negatives_per: int = NEGATIVES_PER_POSITIVE,
        max_neg_ret: float = MAX_NEG_FORWARD_RET,
        limit_up_threshold: float = 0.095,  # A-share 10% limit (ST is 5%)
    ):
        self.forward_window = forward_window
        self.min_cum_ret = min_cum_ret
        self.min_limit_up = min_limit_up
        self.pre_window = pre_window
        self.negatives_per = negatives_per
        self.max_neg_ret = max_neg_ret
        self.limit_up_threshold = limit_up_threshold

    # ── Public API ─────────────────────────────────────────────────────────

    def extract(
        self, daily_cache: pd.DataFrame, symbols: list[str],
    ) -> list[CasePair]:
        """Extract matched pairs from daily OHLCV data.

        Args:
            daily_cache: MultiIndex (trade_date, symbol) DataFrame with
                         columns: open, high, low, close, pre_close, volume,
                         turnover, amount.
            symbols: List of symbols to consider.

        Returns:
            List of CasePair objects for LLM analysis.
        """
        logger.info("Extracting 妖股 cases and matched controls...")

        close = daily_cache["close"].unstack()
        high = daily_cache["high"].unstack()
        low = daily_cache["low"].unstack()
        volume = daily_cache["volume"].unstack()
        turnover = daily_cache["turnover"].unstack() if "turnover" in daily_cache.columns else None
        pre_close = daily_cache["pre_close"].unstack()

        dates = sorted(close.index)
        valid_symbols = [s for s in symbols if s in close.columns]

        # Skip early dates (need pre-launch history)
        start_idx = self.pre_window + STOCK_POOL_BEFORE_LAUNCH
        end_idx = len(dates) - self.forward_window - 1

        logger.info("  Computing pre-launch metrics (vectorized)...")
        metrics_cache = self._compute_all_metrics(
            close, high, low, volume, turnover, pre_close, dates, valid_symbols
        )

        logger.info("  Computing forward returns...")
        fwd_max_ret = self._compute_forward_max_return(close, dates, self.forward_window, valid_symbols)

        logger.info("  Detecting limit-up days...")
        limit_up_mask = self._detect_limit_ups(close, pre_close, dates, valid_symbols)

        logger.info("  Finding positive cases and matching negatives...")
        pairs: list[CasePair] = []
        pos_count = 0

        for i in range(start_idx, end_idx):
            td = dates[i]

            # Find positive cases on this day
            for sym in valid_symbols:
                fwd_ret = fwd_max_ret.loc[td, sym]
                if pd.isna(fwd_ret) or fwd_ret < self.min_cum_ret:
                    continue

                # Check for consecutive limit-up days in forward window
                fwd_end = min(i + self.forward_window, len(dates) - 1)
                lu_streak = self._max_consecutive_limit_up(
                    limit_up_mask, dates, i + 1, fwd_end, sym
                )
                if lu_streak < self.min_limit_up:
                    continue

                # Positive case found
                pos_count += 1
                pos_metrics = self._extract_metrics_at(
                    metrics_cache, dates, i, sym
                )

                # Find negatives: similar pre-launch, no launch
                neg_matches = self._find_negatives(
                    metrics_cache, dates, i, sym, fwd_max_ret, limit_up_mask,
                    valid_symbols, dates
                )

                for neg_sym, neg_metrics in neg_matches:
                    diff = {
                        k: pos_metrics.get(k, 0) - neg_metrics.get(k, 0)
                        for k in pos_metrics
                    }
                    pairs.append(CasePair(
                        symbol_pos=sym,
                        symbol_neg=neg_sym,
                        launch_date=td.date(),
                        pos_metrics=pos_metrics,
                        neg_metrics=neg_metrics,
                        pos_forward_ret=fwd_ret,
                        neg_forward_ret=fwd_max_ret.loc[td, neg_sym]
                        if neg_sym in fwd_max_ret.columns else 0.0,
                        diff_metrics=diff,
                    ))

                if pos_count % 50 == 0:
                    logger.info("    Found %d positive cases, %d pairs so far...",
                                pos_count, len(pairs))

        logger.info("  Total: %d positive cases, %d matched pairs", pos_count, len(pairs))
        return pairs

    # ── Vectorized computations ────────────────────────────────────────────

    def _compute_all_metrics(
        self, close, high, low, volume, turnover, pre_close, dates, symbols,
    ) -> pd.DataFrame:
        """Compute pre-launch metrics for all dates and symbols.

        Returns DataFrame indexed by (date, symbol) with metric columns.
        Each row at date D uses data from D-pre_window to D-1.
        """
        syms = [s for s in symbols if s in close.columns]
        records = []

        # Rolling metrics over pre_window for context
        ret_5d = close.pct_change(5).shift(1)
        ret_20d = close.pct_change(20).shift(1)
        vol_5d = volume.rolling(5, min_periods=3).mean().shift(1)
        vol_20d = volume.rolling(20, min_periods=10).mean().shift(1)

        # Amplitude: (max_high - min_low) / mean_close over pre_window
        rolling_high = high.rolling(self.pre_window, min_periods=10).max().shift(1)
        rolling_low = low.rolling(self.pre_window, min_periods=10).min().shift(1)
        rolling_mean_close = close.rolling(self.pre_window, min_periods=10).mean().shift(1)

        # Close position in daily range
        daily_range = high - low
        close_position = ((close - low) / daily_range.clip(lower=1e-8))
        close_position_5d = close_position.rolling(5, min_periods=3).mean().shift(1)

        # Up days ratio
        up_days = (close > pre_close.shift(1)).astype(float)
        up_days_5d = up_days.rolling(5, min_periods=3).mean().shift(1)

        # Volume surge consistency: fraction of days where vol > 1.5x 20d avg
        vol_surge = (volume > vol_20d * 1.5).astype(float)
        vol_surge_5d = vol_surge.rolling(5, min_periods=3).mean().shift(1)

        # Gap opening strength
        overnight_gap = (close.shift(1) - pre_close) / pre_close.clip(lower=1e-8)
        overnight_gap_5d = overnight_gap.rolling(5, min_periods=3).mean().shift(1)

        start_idx = max(self.pre_window, 5)
        for i in range(start_idx, len(dates)):
            td = dates[i]
            row = {
                "trade_date": td,
                "ret_5d": ret_5d.loc[td, syms].values if td in ret_5d.index else np.nan,
                "ret_20d": ret_20d.loc[td, syms].values if td in ret_20d.index else np.nan,
                "vol_5d_vs_20d": (vol_5d.loc[td, syms] / vol_20d.loc[td, syms].clip(lower=1e-8)).values
                if td in vol_5d.index else np.nan,
                "amplitude_20d": ((rolling_high.loc[td, syms] - rolling_low.loc[td, syms])
                                   / rolling_mean_close.loc[td, syms].clip(lower=1e-8)).values
                if td in rolling_high.index else np.nan,
                "close_position_5d": close_position_5d.loc[td, syms].values
                if td in close_position_5d.index else np.nan,
                "up_days_ratio_5d": up_days_5d.loc[td, syms].values
                if td in up_days_5d.index else np.nan,
                "vol_surge_5d": vol_surge_5d.loc[td, syms].values
                if td in vol_surge_5d.index else np.nan,
                "overnight_gap_5d": overnight_gap_5d.loc[td, syms].values
                if td in overnight_gap_5d.index else np.nan,
            }
            # Add turnover if available
            if turnover is not None:
                t_5d = turnover.rolling(5, min_periods=3).mean().shift(1)
                t_20d = turnover.rolling(20, min_periods=10).mean().shift(1)
                row["turnover_5d"] = t_5d.loc[td, syms].values if td in t_5d.index else np.nan
                row["turnover_5d_vs_20d"] = (t_5d.loc[td, syms] / t_20d.loc[td, syms].clip(lower=1e-8)).values if td in t_5d.index else np.nan

            records.append(row)

        # Build multi-index DataFrame
        result_rows = []
        metric_names = [k for k in records[0] if k != "trade_date"]
        for row in records:
            td = row["trade_date"]
            for j, sym in enumerate(syms):
                entry = {"trade_date": td, "symbol": sym}
                for m in metric_names:
                    val = row[m]
                    entry[m] = float(val[j]) if isinstance(val, np.ndarray) and j < len(val) else np.nan
                result_rows.append(entry)

        df = pd.DataFrame(result_rows)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index(["trade_date", "symbol"]).sort_index()
        return df

    def _compute_forward_max_return(
        self, close, dates, window, symbols,
    ) -> pd.DataFrame:
        """Compute maximum cumulative return over forward N-day window."""
        syms = [s for s in symbols if s in close.columns]
        # For each date, compute the max cumulative return achievable in [T+1, T+window]
        result = pd.DataFrame(index=dates, columns=syms, dtype=float)

        for i in range(len(dates) - window - 1):
            td = dates[i]
            future_close = close.iloc[i + 1 : i + window + 1][syms]
            if future_close.empty:
                continue
            start_price = close.iloc[i][syms]
            cum_rets = future_close.values / start_price.values.clip(1e-8) - 1
            result.loc[td, syms] = np.max(cum_rets, axis=0)

        return result.astype(float)

    def _detect_limit_ups(
        self, close, pre_close, dates, symbols,
    ) -> pd.DataFrame:
        """Detect limit-up days (>= 9.5% daily return)."""
        syms = [s for s in symbols if s in close.columns and s in pre_close.columns]
        daily_ret = close[syms] / pre_close[syms].shift(1).clip(lower=1e-8) - 1
        mask = daily_ret >= self.limit_up_threshold
        return mask.astype(bool)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _max_consecutive_limit_up(
        self, limit_up_mask, dates, start_i, end_i, symbol,
    ) -> int:
        """Max consecutive limit-up days for a symbol in [start_i, end_i]."""
        max_streak = 0
        streak = 0
        for i in range(start_i, end_i + 1):
            td = dates[i]
            if symbol in limit_up_mask.columns and td in limit_up_mask.index:
                if limit_up_mask.loc[td, symbol]:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
        return max_streak

    def _extract_metrics_at(
        self, metrics_cache, dates, idx, symbol,
    ) -> dict[str, float]:
        """Extract pre-launch metrics for a specific date and symbol."""
        td = dates[idx]
        td_pd = pd.Timestamp(td)
        try:
            row = metrics_cache.xs(td_pd, level="trade_date")
            if symbol in row.index:
                vals = row.loc[symbol]
                return {k: float(v) if not pd.isna(v) else 0.0
                        for k, v in vals.items()}
        except (KeyError, TypeError):
            pass
        return {}

    def _find_negatives(
        self, metrics_cache, dates, idx, pos_symbol,
        fwd_max_ret, limit_up_mask, symbols, all_dates,
    ) -> list[tuple[str, dict[str, float]]]:
        """Find stocks with similar pre-launch but no launch."""
        td = dates[idx]
        td_pd = pd.Timestamp(td)
        pos_metrics = self._extract_metrics_at(metrics_cache, dates, idx, pos_symbol)
        if not pos_metrics:
            return []

        # Get all metrics for this date
        try:
            day_metrics = metrics_cache.xs(td_pd, level="trade_date")
        except KeyError:
            return []

        # Normalize metrics for distance computation
        feature_keys = ["amplitude_20d", "vol_5d_vs_20d", "close_position_5d",
                        "ret_5d", "up_days_ratio_5d", "vol_surge_5d"]
        feature_keys = [k for k in feature_keys if k in day_metrics.columns]

        candidates = []
        for sym in symbols:
            if sym == pos_symbol:
                continue
            if sym not in day_metrics.index:
                continue

            # Forward return check: must NOT have launched
            fwd_ret = fwd_max_ret.loc[td, sym] if sym in fwd_max_ret.columns else 0
            if pd.isna(fwd_ret) or fwd_ret >= self.max_neg_ret:
                continue

            # Check no consecutive limit-ups
            fwd_end = min(idx + self.forward_window, len(all_dates) - 1)
            lu_streak = self._max_consecutive_limit_up(
                limit_up_mask, all_dates, idx + 1, fwd_end, sym
            )
            if lu_streak >= self.min_limit_up:
                continue

            # Compute distance to positive
            dist = 0.0
            for fk in feature_keys:
                pv = pos_metrics.get(fk, 0) or 0
                nv = day_metrics.loc[sym, fk] if fk in day_metrics.columns else 0
                if pd.isna(nv):
                    nv = 0
                dist += (float(pv) - float(nv)) ** 2

            candidates.append((sym, dist, day_metrics))

        # Sort by distance, take top N
        candidates.sort(key=lambda x: x[1])
        results = []
        for sym, dist, day_met in candidates[:self.negatives_per]:
            met = {k: float(day_met.loc[sym, k]) if k in day_met.columns and not pd.isna(day_met.loc[sym, k]) else 0.0
                   for k in pos_metrics}
            results.append((sym, met))

        return results

    # ── Summary stats for LLM ──────────────────────────────────────────────

    @staticmethod
    def compute_summary(pairs: list[CasePair]) -> dict:
        """Compute aggregate statistics across all pairs for LLM prompt."""
        if not pairs:
            return {"n_pairs": 0}

        pos_avg = {}
        neg_avg = {}
        diff_avg = {}
        metric_keys = list(pairs[0].pos_metrics.keys())

        for k in metric_keys:
            pos_vals = [p.pos_metrics.get(k, 0) or 0 for p in pairs]
            neg_vals = [p.neg_metrics.get(k, 0) or 0 for p in pairs]
            pos_avg[k] = float(np.mean(pos_vals))
            neg_avg[k] = float(np.mean(neg_vals))
            diff_avg[k] = pos_avg[k] - neg_avg[k]

        # Top discriminating features
        sorted_diffs = sorted(diff_avg.items(), key=lambda x: -abs(x[1]))

        pos_fwd = [p.pos_forward_ret for p in pairs]
        neg_fwd = [p.neg_forward_ret for p in pairs]

        return {
            "n_positives": len(set(p.symbol_pos for p in pairs)),
            "n_pairs": len(pairs),
            "pos_avg_metrics": pos_avg,
            "neg_avg_metrics": neg_avg,
            "top_discriminating": [(k, diff_avg[k]) for k, _ in sorted_diffs[:8]],
            "pos_avg_forward_ret": float(np.mean(pos_fwd)),
            "neg_avg_forward_ret": float(np.mean(neg_fwd)),
        }

    @staticmethod
    def top_pairs(pairs: list[CasePair], n: int = 5) -> list[CasePair]:
        """Return top-N pairs with largest metric differences (most informative)."""
        def diff_score(p: CasePair) -> float:
            return sum(abs(v) for v in p.diff_metrics.values())
        sorted_pairs = sorted(pairs, key=diff_score, reverse=True)
        # Deduplicate by pos_symbol
        seen = set()
        result = []
        for p in sorted_pairs:
            if p.symbol_pos not in seen:
                seen.add(p.symbol_pos)
                result.append(p)
            if len(result) >= n:
                break
        return result
