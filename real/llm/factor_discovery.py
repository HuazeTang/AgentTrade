"""LLM-assisted factor discovery.

Uses an LLM to generate factor code from natural language descriptions
of economic intuition. Only used during research/development, never at runtime.
"""

from __future__ import annotations

import logging
from typing import Optional

from agent.llm_client import LLMClient, create_default_client

logger = logging.getLogger(__name__)


class FactorDiscoveryAssistant:
    """Generates factor code from natural language descriptions using an LLM.

    Usage:
        assistant = FactorDiscoveryAssistant()
        code = assistant.generate_factor(
            "Stocks with high R&D spending should outperform in the long run"
        )
        # Code is Python source conforming to the Factor interface.
    """

    SYSTEM_PROMPT = """You are a quantitative researcher specializing in equity factor discovery.
Given an economic intuition, generate Python code implementing a Factor class.

The Factor interface:
```python
from factor.base import Factor, FactorMeta
from factor.registry import register_factor
import pandas as pd

@register_factor
class MyFactor(Factor):
    meta = FactorMeta(
        name="my_factor_name",
        category="momentum|value|quality|volatility|size|liquidity|growth",
        description="One line describing what this factor captures",
        lookback_days=20,  # days of lookback needed
    )

    @property
    def required_fields(self) -> list[str]:
        return ["close", "volume"]  # raw data fields needed

    def compute(self, data: pd.DataFrame) -> pd.Series:
        # data has multi-index (trade_date, symbol)
        # Return a Series with the same index
        ...
```

Requirements:
- Use only pandas/numpy operations, vectorized on the multi-index
- Handle edge cases (zero volume, NaN prices)
- The compute() method must work on the full multi-index DataFrame
- category must be one of: momentum, value, quality, volatility, size, liquidity, growth
- name must be snake_case
- Include proper lookback_days estimation
"""

    def __init__(self, client: LLMClient | None = None):
        self._client = client or create_default_client()

    def generate_factor(
        self,
        description: str,
        existing_factors: list[str] | None = None,
    ) -> str:
        """Generate factor implementation from a natural language description.

        Returns the Python source code or an error message.
        """
        if not self._client.configured:
            return self._no_api_fallback()

        existing_str = "\n".join(f"- {n}" for n in (existing_factors or []))
        prompt = f"""Economic intuition: {description}

Existing factors in the system:
{existing_str or '(none)'}

Generate a complete Factor class implementation that captures this intuition.
Output ONLY the Python code, no explanation."""

        try:
            response = self._client.chat(prompt, system_prompt=self.SYSTEM_PROMPT)
            text = response.text
            # Strip markdown fences if present
            return _extract_code(text) or text
        except Exception as e:
            logger.error("LLM factor generation failed: %s", e)
            return f"# LLM call failed: {e}\n# Falling back to manual implementation."

    def analyze_effectiveness(
        self,
        factor_name: str,
        ic_summary: dict,
        quantile_returns: dict,
    ) -> str:
        """Analyze factor validation results and suggest improvements."""
        if not self._client.configured:
            return "LLM not configured. Analysis unavailable."

        prompt = f"""Factor '{factor_name}' validation results:
- IC mean: {ic_summary.get('mean', 'N/A')}
- IC std: {ic_summary.get('std', 'N/A')}
- IC IR: {ic_summary.get('ir', 'N/A')}
- Hit rate: {ic_summary.get('hit_rate', 'N/A')}

Quantile returns:
{quantile_returns}

Analyze these results. Is this factor promising? How could it be improved?
Consider: sector neutralization, outlier handling, different lookback windows,
combining with other factors."""

        try:
            return self._client.chat(prompt, system_prompt=self.SYSTEM_PROMPT).text
        except Exception as e:
            logger.error("LLM effectiveness analysis failed: %s", e)
            return f"Analysis failed: {e}"

    def suggest_model_architecture(
        self,
        n_features: int,
        n_samples: int,
        problem_description: str,
    ) -> str:
        """Suggest an ML model architecture given the problem context."""
        if not self._client.configured:
            return "LLM not configured. Suggestion unavailable."

        prompt = f"""Design a machine learning model for equity alpha prediction:

Problem: {problem_description}
- {n_features} features
- {n_samples} samples
- Time-series panel data (samples are not i.i.d.)

Suggest:
1. Model type (GBDT, neural net, ensemble)
2. Key hyperparameters
3. Feature engineering approach
4. Validation strategy
5. Expected IC range for a good model"""

        try:
            return self._client.chat(prompt, system_prompt=self.SYSTEM_PROMPT).text
        except Exception as e:
            logger.error("LLM architecture suggestion failed: %s", e)
            return f"Suggestion failed: {e}"

    def _no_api_fallback(self) -> str:
        return (
            "No API key configured. Set QWEN_API_KEY or OPENAI_API_KEY "
            "environment variable.\n\n"
            "LLM-assisted factor discovery is optional. You can implement "
            "factors manually by extending the Factor base class."
        )


def _extract_code(text: str) -> str | None:
    """Extract Python code from markdown fences."""
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start) if "```" in text[start:] else len(text)
        return text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start) if "```" in text[start:] else len(text)
        return text[start:end].strip()
    return None
