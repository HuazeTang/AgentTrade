"""LLM-assisted model architecture suggestions."""

from __future__ import annotations

import logging

from agent.llm_client import LLMClient, create_default_client

logger = logging.getLogger(__name__)


class ModelSuggestionAssistant:
    """Uses LLM to suggest model architectures and hyperparameters.

    Used during development only. The actual backtest uses trained,
    serialized models with no LLM dependency.
    """

    SYSTEM_PROMPT = """You are a machine learning researcher specializing in
financial time series. Given a description of the prediction problem,
dataset characteristics, and constraints, suggest model architectures
and training approaches. Be concrete about hyperparameters, regularization
strategies, and validation methodology."""

    def __init__(self, client: LLMClient | None = None):
        self._client = client or create_default_client()

    def suggest_hyperparameters(
        self,
        model_type: str,
        n_features: int,
        n_samples: int,
        target_description: str,
    ) -> str:
        """Suggest hyperparameters for a given model type."""
        if not self._client.configured:
            return "LLM not configured. Suggestion unavailable."

        prompt = f"""Model type: {model_type}
Features: {n_features}
Samples: {n_samples}
Target: {target_description}

Suggest optimal hyperparameters and training configuration.
Include: learning rate, tree depth / layer sizes, regularization,
early stopping criteria, and expected training time."""

        try:
            return self._client.chat(prompt, system_prompt=self.SYSTEM_PROMPT).text
        except Exception as e:
            logger.error("LLM hyperparameter suggestion failed: %s", e)
            return f"Suggestion failed: {e}"

    def diagnose_overfitting(
        self,
        train_ic: float,
        val_ic: float,
        test_ic: float,
        n_features: int,
        n_samples: int,
    ) -> str:
        """Diagnose potential overfitting from IC gap."""
        if not self._client.configured:
            return "LLM not configured. Diagnosis unavailable."

        prompt = f"""Model performance:
- Train IC: {train_ic:.4f}
- Validation IC: {val_ic:.4f}
- Test IC: {test_ic:.4f}
- {n_features} features, {n_samples} samples

Is there evidence of overfitting? What specific adjustments would you recommend?"""

        try:
            return self._client.chat(prompt, system_prompt=self.SYSTEM_PROMPT).text
        except Exception as e:
            logger.error("LLM overfitting diagnosis failed: %s", e)
            return f"Diagnosis failed: {e}"

    def suggest_feature_engineering(
        self,
        factor_names: list[str],
        target_description: str,
    ) -> str:
        """Suggest feature engineering transformations."""
        if not self._client.configured:
            return "LLM not configured. Suggestion unavailable."

        prompt = f"""Available factors: {', '.join(factor_names)}
Target: {target_description}

Suggest feature engineering approaches:
1. Which factors to combine/interact
2. Nonlinear transformations to try
3. Dimensionality reduction considerations
4. Sector/market cap neutralization strategy"""

        try:
            return self._client.chat(prompt, system_prompt=self.SYSTEM_PROMPT).text
        except Exception as e:
            logger.error("LLM feature engineering suggestion failed: %s", e)
            return f"Suggestion failed: {e}"
