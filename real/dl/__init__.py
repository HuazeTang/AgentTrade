"""Dual-tower deep learning model for yaogu (妖股) detection.

Architecture: 1D CNN (local temporal patterns) + Transformer (long-range
dependencies), fused with MLP head. Optimized for precision via Focal Loss.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Focal Loss ──────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Down-weights easy examples, focuses training on hard examples.
    Optimized for precision over recall when α > 0.5 for positive class.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)  # p_t = exp(-BCE)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return focal_weight.mean() * bce.mean()  # scale-invariant


# ── Smooth AP Loss ───────────────────────────────────────────────────────

class SmoothAPLoss(nn.Module):
    """Differentiable Average Precision loss.

    Replaces the discrete ranking in AP with sigmoid-smoothed indicators.
    Returns (1 - AP) so minimizing the loss maximizes AP.

    Reference: Brown et al., "Smooth-AP: Smoothing the Path Towards
    Large-Scale Image Retrieval", ECCV 2020.
    """

    def __init__(self, tau: float = 0.01):
        super().__init__()
        self.tau = tau

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_mask = targets == 1
        n_pos = int(pos_mask.sum().item())
        if n_pos == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        scores = torch.sigmoid(logits)
        pos_scores = scores[pos_mask]                          # (P,)

        # Smoothed indicator: I(s_j > s_i) ≈ sigmoid((s_j - s_i) / tau)
        diff = scores.unsqueeze(0) - pos_scores.unsqueeze(1)   # (P, B)
        smoothed_gt = torch.sigmoid(diff / self.tau)           # (P, B)

        # Rank of each positive (subtract self-contribution ~0.5)
        ranks = 1.0 + smoothed_gt.sum(dim=1) - 0.5             # (P,)

        # Precision at rank position k: k / rank_k
        _, sort_idx = torch.sort(ranks)
        precisions = torch.arange(1, n_pos + 1, device=logits.device).float()
        precisions = precisions / ranks[sort_idx]

        ap = precisions.mean()
        return 1.0 - ap


# ── Hard AP (evaluation) ─────────────────────────────────────────────────

def compute_average_precision(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Non-differentiable AP score for validation (hard ranking)."""
    logits = logits.cpu()
    targets = targets.cpu()
    scores = torch.sigmoid(logits)
    sorted_idx = torch.argsort(scores, descending=True)
    sorted_targets = targets[sorted_idx].long()
    n_pos = int(sorted_targets.sum().item())
    if n_pos == 0:
        return 0.0
    tp_cumsum = torch.cumsum(sorted_targets, dim=0).float()
    precision_at_k = tp_cumsum / torch.arange(1, len(sorted_targets) + 1)
    pos_mask = sorted_targets == 1
    return float(precision_at_k[pos_mask].mean().item())


# ── Gradient Reversal Layer ──────────────────────────────────────────────

class _GradientReversal(torch.autograd.Function):
    """Reverse gradient sign during backward pass, scaled by lambda."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, lambda_)


def compute_grl_lambda(epoch: int, total_epochs: int,
                       max_lambda: float = 0.1, gamma: float = 10.0) -> float:
    """Sigmoidal GRL schedule from Ganin et al. (2016).

    Starts near 0, rises to max_lambda. gamma controls steepness.
    """
    import math as _math
    p = epoch / max(total_epochs - 1, 1)
    return max_lambda * (2.0 / (1.0 + _math.exp(-gamma * p)) - 1.0)


# ── Positional Encoding ─────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer input."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


# ── CNN Tower ───────────────────────────────────────────────────────────

class CNNTower(nn.Module):
    """1D CNN with dilated convolutions for multi-scale temporal pattern extraction."""

    def __init__(self, in_channels: int = 8, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, dilation=1, padding="same")
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, dilation=2, padding="same")
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, dilation=4, padding="same")
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → permute to (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)  # (B, 128)
        return x


# ── Transformer Tower ───────────────────────────────────────────────────

class TransformerTower(nn.Module):
    """Lightweight Transformer encoder for long-range temporal dependencies."""

    def __init__(self, in_features: int = 8, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=100, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x)  # (B, T, d_model)
        x = x.mean(dim=1)  # mean pooling → (B, d_model)
        return x


# ── Dual-Tower Model ────────────────────────────────────────────────────

class DualTowerModel(nn.Module):
    """Dual-tower CNN + Transformer for yaogu detection.

    CNN captures local multi-scale temporal patterns (3/6/12-day).
    Transformer captures long-range dependencies across the 60-day window.
    Fusion MLP head produces a probability score.
    """

    def __init__(
        self,
        in_features: int = 8,
        cnn_dropout: float = 0.2,
        trans_d_model: int = 64,
        trans_nhead: int = 4,
        trans_num_layers: int = 2,
        trans_dim_ff: int = 128,
        trans_dropout: float = 0.2,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.cnn = CNNTower(in_channels=in_features, dropout=cnn_dropout)
        self.transformer = TransformerTower(
            in_features=in_features, d_model=trans_d_model, nhead=trans_nhead,
            num_layers=trans_num_layers, dim_feedforward=trans_dim_ff,
            dropout=trans_dropout,
        )
        cnn_out = 128
        trans_out = trans_d_model
        fusion_dim = cnn_out + trans_out

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(x)
        trans_out = self.transformer(x)
        fused = torch.cat([cnn_out, trans_out], dim=-1)
        return self.head(fused).squeeze(-1)  # logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability scores (0-1)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


# ── Threshold Tuning ────────────────────────────────────────────────────

def find_precision_threshold(
    model: DualTowerModel,
    val_loader: torch.utils.data.DataLoader,
    target_precision: float = 0.3,
    device: str = "cpu",
    min_tp: int = 5,
) -> tuple[float, dict]:
    """Find probability threshold that achieves target precision on validation set.

    Scans thresholds and finds the one with highest precision among those
    meeting precision >= target_precision AND tp >= min_tp.

    Returns (best_threshold, metrics_dict).
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            probs = model.predict_proba(x_batch)
            all_probs.append(probs.cpu())
            all_labels.append(y_batch)

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    best_threshold = 0.5
    best_precision = 0.0
    best_metrics = {}

    thresholds = torch.linspace(0.05, 0.99, 95).numpy()
    for t in thresholds:
        preds = (probs >= t).astype(float)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Highest precision among thresholds meeting precision target + min_tp
        if precision >= target_precision and tp >= min_tp and precision > best_precision:
            best_precision = precision
            best_threshold = float(t)
            best_metrics = {
                "threshold": float(t), "precision": precision, "recall": recall,
                "tp": int(tp), "fp": int(fp), "fn": int(fn),
                "f1": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
            }

    # Fallback: if no threshold meets both criteria, relax min_tp to 1
    if best_precision == 0:
        for t in thresholds:
            preds = (probs >= t).astype(float)
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            if p >= target_precision and p > best_precision:
                best_precision = p
                best_threshold = float(t)
                best_metrics = {
                    "threshold": float(t), "precision": p, "recall": r,
                    "tp": int(tp), "fp": int(fp), "fn": int(fn),
                    "f1": 2 * p * r / (p + r) if (p + r) > 0 else 0,
                }

    # Last resort: just highest precision overall
    if best_precision == 0:
        best_idx = 0
        for i, t in enumerate(thresholds):
            preds = (probs >= t).astype(float)
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            if p > best_precision:
                best_precision = p
                best_idx = i
        t = float(thresholds[best_idx])
        preds = (probs >= t).astype(float)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        best_threshold = t
        best_metrics = {
            "threshold": t, "precision": best_precision,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "f1": 0,
        }

    return best_threshold, best_metrics


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Adversarial Dual-Tower Model ─────────────────────────────────────────

class AdversarialDualTowerModel(nn.Module):
    """DualTowerModel with domain-adversarial year classifier.

    Wraps a DualTowerModel and adds a Gradient Reversal Layer + year
    classifier on the fused CNN+Transformer features. This forces the
    model to learn year-invariant representations.

    During training: forward() returns (yaogu_logits, year_logits).
    During inference: predict_proba() delegates to base model.
    Checkpoints save only the base model state_dict for compatibility.
    """

    def __init__(self, base_model: DualTowerModel, n_years: int,
                 grl_lambda: float = 0.1, year_cls_dropout: float = 0.2):
        super().__init__()
        self.base = base_model
        self._grl_lambda = grl_lambda
        fusion_dim = 192  # 128 (CNN) + 64 (Transformer)
        self.year_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(year_cls_dropout),
            nn.Linear(64, n_years),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cnn_out = self.base.cnn(x)
        trans_out = self.base.transformer(x)
        fused = torch.cat([cnn_out, trans_out], dim=-1)
        yaogu_logits = self.base.head(fused).squeeze(-1)

        reversed_f = grad_reverse(fused, self._grl_lambda)
        year_logits = self.year_classifier(reversed_f)

        return yaogu_logits, year_logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.predict_proba(x)

    def set_grl_lambda(self, value: float) -> None:
        self._grl_lambda = value
