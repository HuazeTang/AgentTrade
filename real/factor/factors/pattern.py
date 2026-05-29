"""XGBoost-learned pattern matching factor for extreme upside detection.

Plan C v3: full feature set (34 factors) + ADASYN adaptive oversampling
+ sample weights based on return extremity. ADASYN generates more
synthetic samples in low-density regions of the minority class, creating
a more robust decision boundary.

Evolution:
  v1: 34 features, 20d>20%, no balancing → zero IC weight
  v2: 18 features, 10d>10%, SMOTE 0.33 → min weight (0.0057)
  v3: 34 features, 10d>10%, ADASYN 0.5 + return-rank sample weights
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from factor.base import Factor, FactorMeta
from factor.registry import register_factor
from data.calendar import get_trading_days

logger = logging.getLogger(__name__)


# Full feature set — all enabled baseline factors.
# ADASYN handles class imbalance, so we can use richer features.
FEATURE_FACTOR_NAMES: list[str] = [
    "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m1m",
    "reversal_5d", "reversal_10d",
    "volatility_20d", "volatility_60d",
    "beta_60d",
    "overnight_gap_5d", "gap_strength_5d",
    "turnover_20d",
    "trend_efficiency_20d", "ma_trend_5_20", "donchian_pct_20d",
    "up_days_ratio_20d", "ma_cross_5_20",
    "limit_up_freq_20d", "relative_strength_10d", "volume_surge_5d",
    "close_position_5d",
    "vol_weighted_mom_5d", "money_flow_ratio_20d", "vwap_delta_5d",
    "vol_price_div_5d",
    "risk_adj_mom_20d", "dd_recovery_5d",
    "market_dd_beta_20d",
    "decline_intensity_10d", "bounce_strength_5d", "volume_climax_ratio",
    "oversold_reversal_5d", "price_position_10d",
    "momentum_deceleration_5d", "signed_volume_climax_5d", "reversal_initiation_3d",
]

TRAIN_START = date(2024, 8, 30)
TRAIN_END = date(2025, 10, 16)
FORWARD_DAYS = 10
RETURN_THRESHOLD = 0.10
NEG_POS_RATIO = 8            # subsample negatives to this ratio before ADASYN
SMOTE_TARGET_RATIO = 0.5     # minority = 50% of majority after ADASYN
MIN_TRAIN_SAMPLES = 200


@register_factor
class PatternScoreXGB(Factor):
    """XGBoost probability of >10% return in 10 days (full features + ADASYN)."""

    meta = FactorMeta(
        name="pattern_score_xgb",
        category="pattern",
        description="XGBoost v3: >10% in 10d, 34 features, ADASYN + return-rank weights",
        lookback_days=FORWARD_DAYS,
    )

    @property
    def dependencies(self) -> list[str]:
        return FEATURE_FACTOR_NAMES

    @property
    def required_fields(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.Series:
        import xgboost as xgb

        result = pd.Series(np.nan, index=data.index, name=self.meta.name)

        feature_cols = [c for c in FEATURE_FACTOR_NAMES if c in data.columns]
        if len(feature_cols) < 10:
            logger.warning("pattern_score_xgb: only %d features, skipping", len(feature_cols))
            return result

        # Labels: 10-day forward return
        close = data["close"].unstack()
        fwd_ret = close.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS).stack()

        # Binary label for ADASYN
        label: pd.Series = (fwd_ret > RETURN_THRESHOLD).astype(int)
        label.name = "_target"

        # Sample weight: cross-sectional rank percentile of forward return.
        # Gives higher weight to stocks with more extreme positive returns.
        fwd_rank = fwd_ret.groupby(level="trade_date").rank(pct=True)
        sample_weight: pd.Series = fwd_rank.clip(lower=0.0, upper=1.0)
        sample_weight.name = "_weight"

        combined = data[feature_cols].join(label, how="left")
        combined["_weight"] = sample_weight
        label_vals = combined["_target"]
        weight_vals = combined["_weight"]
        X_all = combined[feature_cols]

        # Train / predict split
        train_dates = get_trading_days(TRAIN_START, TRAIN_END)
        date_idx = combined.index.get_level_values("trade_date")
        is_train = date_idx.isin(train_dates)

        train_ok = is_train & X_all.notna().all(axis=1) & label_vals.notna()
        n_train = train_ok.sum()
        if n_train < MIN_TRAIN_SAMPLES:
            logger.warning("pattern_score_xgb: only %d trainable rows, skipping", n_train)
            return result

        X_train_full = X_all[train_ok].astype(float)
        y_train_full = label_vals[train_ok].astype(int)
        w_train_full = weight_vals[train_ok].astype(float)
        pos_frac = y_train_full.mean()
        logger.info("pattern_score_xgb v3: %d train rows, %.2f%% positive (10d>10%%)",
                     n_train, pos_frac * 100)

        if pos_frac < 0.002:
            logger.warning("pattern_score_xgb: positive rate too low (%.3f%%), skipping",
                           pos_frac * 100)
            return result

        # Time-based train/val split
        full_dates_sorted = sorted(X_train_full.index.get_level_values("trade_date").unique())
        split_date = full_dates_sorted[int(len(full_dates_sorted) * 0.8)]
        val_mask = X_train_full.index.get_level_values("trade_date") >= split_date
        tr_mask = ~val_mask

        X_tr_raw = X_train_full[tr_mask]
        y_tr_raw = y_train_full[tr_mask]
        X_val = X_train_full[val_mask]
        y_val = y_train_full[val_mask]
        w_val = w_train_full[val_mask]

        # Step 1: subsample negatives
        pos_idx_tr = y_tr_raw[y_tr_raw == 1].index
        neg_idx_tr = y_tr_raw[y_tr_raw == 0].index
        n_pos_tr = len(pos_idx_tr)

        max_neg = n_pos_tr * NEG_POS_RATIO
        rng = np.random.default_rng(42)
        if len(neg_idx_tr) > max_neg:
            neg_idx_tr = neg_idx_tr[rng.choice(len(neg_idx_tr), size=max_neg, replace=False)]

        keep_idx_tr = pos_idx_tr.union(neg_idx_tr)
        X_tr_subsampled = X_tr_raw.loc[keep_idx_tr]
        y_tr_subsampled = y_tr_raw.loc[keep_idx_tr]

        # Step 2: ADASYN — adaptive synthetic sampling, focuses on harder cases
        try:
            from imblearn.over_sampling import ADASYN
            adasyn = ADASYN(
                sampling_strategy=SMOTE_TARGET_RATIO,
                random_state=42,
                n_neighbors=min(5, n_pos_tr - 1),
                n_jobs=1,
            )
            X_tr, y_tr = adasyn.fit_resample(X_tr_subsampled, y_tr_subsampled)
            logger.info("pattern_score_xgb: ADASYN %d → %d (ratio %.2f)",
                         len(y_tr_subsampled), len(y_tr), y_tr.mean())
        except Exception as e:
            logger.warning("pattern_score_xgb: ADASYN failed (%s), trying SMOTE", e)
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(
                    sampling_strategy=SMOTE_TARGET_RATIO,
                    random_state=42,
                    k_neighbors=min(5, n_pos_tr - 1),
                    n_jobs=1,
                )
                X_tr, y_tr = smote.fit_resample(X_tr_subsampled, y_tr_subsampled)
                logger.info("pattern_score_xgb: SMOTE fallback %d → %d (ratio %.2f)",
                             len(y_tr_subsampled), len(y_tr), y_tr.mean())
            except Exception as e2:
                logger.warning("pattern_score_xgb: all oversampling failed (%s)", e2)
                X_tr, y_tr = X_tr_subsampled, y_tr_subsampled

        has_val = len(y_val) >= 50
        # Only use sample weights for validation (training set has synthetic samples)
        val_sample_weight = w_val.loc[val_mask[val_mask].index] if has_val else None

        model = xgb.XGBClassifier(
            max_depth=4,                    # slightly deeper for richer feature set
            learning_rate=0.05,
            n_estimators=300,
            objective="binary:logistic",
            eval_metric="auc",
            early_stopping_rounds=30 if has_val else None,
            random_state=42,
            n_jobs=1,
        )

        if has_val:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                sample_weight_eval_set=[val_sample_weight],
                verbose=False,
            )
        else:
            model.fit(X_tr, y_tr, verbose=False)

        # Feature importance
        importance = model.feature_importances_
        top_idx = np.argsort(importance)[-8:][::-1]
        top_features = [(feature_cols[i], importance[i]) for i in top_idx]
        logger.info("pattern_score_xgb v3: top features: %s",
                     ", ".join(f"{n}={v:.3f}" for n, v in top_features))

        # Predict on all dates
        pred_ok = X_all.notna().all(axis=1)
        if pred_ok.sum() == 0:
            return result

        X_pred = X_all[pred_ok].astype(float)
        proba = model.predict_proba(X_pred)[:, 1]
        result.loc[pred_ok] = proba
        logger.info("pattern_score_xgb v3: predicted %d rows, mean=%.4f", pred_ok.sum(), proba.mean())

        return result
