"""Unified LLM client for the agent system.

Supports DeepSeek (primary, via OpenAI-compatible chat API), Qwen (DashScope),
and OpenAI as fallback.

All calls go through this module so rate-limiting, retry, structured output
parsing, and fallback logic are centralized.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

# Auto-load .env from project root (if present)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_PATH)
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
QWEN_MODELS = ["qwen-plus", "qwen-max", "qwen-turbo"]
OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-reasoner", "deepseek-v4-pro", "deepseek-v4-flash"]

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.3  # low for structured outputs


@dataclass
class LLMResponse:
    """Normalized response from any LLM backend."""
    text: str
    model: str
    backend: str  # "qwen", "openai", or "deepseek"
    latency_ms: float
    usage: dict = field(default_factory=dict)


class LLMClient:
    """Unified client that auto-selects backend based on available API keys.

    Priority: DEEPSEEK_API_KEY > QWEN_API_KEY > OPENAI_API_KEY

    DeepSeek and OpenAI share the same chat-completions calling pattern.
    Qwen uses DashScope's text-generation endpoint.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        # Resolve API key and backend
        self._api_key = api_key
        self._backend: str | None = None

        if self._api_key:
            self._backend = self._detect_backend(self.model)
        else:
            self._api_key, self._backend = self._resolve_from_env(self.model)

        self._call_count = 0
        self._last_call_time = 0.0
        self._min_interval = 0.1  # rate limit: 10 calls/sec max

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def configured(self) -> bool:
        """Whether an API key is available."""
        return self._api_key is not None and self._backend is not None

    @property
    def backend(self) -> str | None:
        return self._backend

    def chat(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """Send a prompt and get a text response.

        Args:
            prompt: The main user prompt.
            system_prompt: Optional system-level instructions.

        Returns:
            LLMResponse with the text output.
        """
        if not self.configured:
            return LLMResponse(
                text=self._no_api_fallback(prompt),
                model="fallback",
                backend="none",
                latency_ms=0,
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self._call_with_retry(messages)

    def chat_json(
        self,
        prompt: str,
        system_prompt: str = "",
        expected_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send a prompt and parse the response as JSON.

        Args:
            prompt: The main user prompt.
            system_prompt: Optional system-level instructions.
            expected_keys: If provided, validate these keys exist in the response.

        Returns:
            Parsed JSON dict, or {"error": "...", "raw_text": "..."} on failure.
        """
        # Add JSON instruction to prompt if not already present
        if "JSON" not in prompt and "json" not in prompt:
            json_instruction = "\n\nRespond ONLY with valid JSON. No markdown fences, no explanation."
            full_prompt = prompt + json_instruction
        else:
            full_prompt = prompt

        response = self.chat(full_prompt, system_prompt)
        text = response.text

        # Try to extract JSON from markdown fences if present
        text = _extract_json(text)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response: %s", text[:200])
            return {"error": "JSON parse failed", "raw_text": text}

        if expected_keys:
            missing = [k for k in expected_keys if k not in parsed]
            if missing:
                logger.warning("LLM JSON missing keys: %s", missing)

        return parsed

    # ── Convenience methods for agent pipeline ───────────────────────────────

    def generate_factor_ideas(
        self,
        diagnosis: dict,
        existing_factors: list[str],
        n_ideas: int = 5,
    ) -> list[dict]:
        """Given a performance diagnosis, suggest factor ideas.

        Returns a list of dicts with keys: name, intuition, category, expression_hint.
        """
        prompt = f"""You are a quantitative researcher analyzing factor performance decay.

Current diagnosis:
{json.dumps(diagnosis, indent=2, ensure_ascii=False)}

Existing factors: {', '.join(existing_factors) if existing_factors else '(none)'}

Generate {n_ideas} new factor ideas that could address the weaknesses identified.
Each idea must include:
- name: snake_case identifier
- intuition: economic reasoning (1-2 sentences)
- category: momentum|value|quality|volatility|size|liquidity|growth|composite
- expression_hint: mathematical formula sketch using close/volume/amount/turnover/high/low/open

Output a JSON array of objects with these exact keys."""

        result = self.chat_json(prompt, expected_keys=None)
        if isinstance(result, list):
            return result
        if "error" in result:
            return []
        # Sometimes LLM wraps in a key like "ideas"
        for key in ("ideas", "factors", "suggestions"):
            if key in result and isinstance(result[key], list):
                return result[key]
        return [result] if result else []

    def diagnose_regime(
        self,
        ic_history: dict[str, list[float]],
        factor_names: list[str],
        recent_performance: dict,
    ) -> dict:
        """Analyze regime shift from IC history and recent performance.

        Returns dict with: regime_description, decay_detected, affected_factors,
        suggested_actions.
        """
        prompt = f"""You are analyzing factor performance for regime change detection.

Factor IC history (by quarter):
{json.dumps(ic_history, indent=2)}

Recent performance summary:
{json.dumps(recent_performance, indent=2, ensure_ascii=False)}

Analyze:
1. Is there a regime shift? Describe the nature of the change.
2. Which factors are decaying vs stable vs improving?
3. What market conditions could explain these changes?
4. What actions should we take? (retire, adjust, replace)

Output JSON with keys:
- regime_description: string
- decay_detected: boolean
- affected_factors: list of factor names showing decay
- stable_factors: list of factor names holding up
- suggested_actions: list of strings (specific actions)
- confidence: float 0-1"""

        return self.chat_json(prompt, expected_keys=["regime_description", "decay_detected"])

    def suggest_strategy_params(
        self,
        strategy_name: str,
        current_params: dict,
        performance: dict,
        market_context: str,
    ) -> dict:
        """Suggest parameter adjustments for a strategy.

        Returns dict with: adjusted_params, rationale, expected_impact.
        """
        prompt = f"""You are optimizing a quantitative trading strategy.

Strategy: {strategy_name}
Current parameters: {json.dumps(current_params)}
Recent performance: {json.dumps(performance, ensure_ascii=False)}
Market context: {market_context}

Suggest parameter adjustments. Consider:
- Position sizing (wider stops in high vol, tighter in low vol)
- Lookback windows (shorter in fast regimes, longer in slow)
- Factor weights (reduce decaying factors, boost stable ones)

Output JSON with:
- adjusted_params: object with parameter name -> new value
- rationale: string explaining each change
- expected_impact: string describing expected effect"""

        return self.chat_json(prompt, expected_keys=["adjusted_params", "rationale"])

    # ── Internal ────────────────────────────────────────────────────────────

    def _call_with_retry(self, messages: list[dict]) -> LLMResponse:
        """Call the backend with retry on transient failures."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._do_call(messages)
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.debug("LLM call failed (attempt %d/%d), retrying in %ds: %s",
                                 attempt + 1, self.max_retries, wait, e)
                    time.sleep(wait)
            except requests.HTTPError as e:
                resp = e.response if hasattr(e, 'response') else None
                status = resp.status_code if resp else None
                if status == 429 and attempt < self.max_retries:
                    wait = 2 ** attempt * 2
                    logger.debug("Rate limited, retrying in %ds", wait)
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_error}")

    def _do_call(self, messages: list[dict]) -> LLMResponse:
        """Execute a single API call to the selected backend."""
        # Rate limit
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        t0 = time.time()
        if self._backend == "qwen":
            resp = self._call_qwen(messages)
        elif self._backend in ("openai", "deepseek"):
            base_url = DEEPSEEK_BASE_URL if self._backend == "deepseek" else OPENAI_BASE_URL
            resp = self._call_openai_compatible(messages, base_url)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

        self._call_count += 1
        self._last_call_time = time.time()
        return resp

    def _call_qwen(self, messages: list[dict]) -> LLMResponse:
        """Call Qwen via DashScope API."""
        # Convert chat messages to a single prompt (DashScope text generation)
        prompt_parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                prompt_parts.append(f"Instructions: {content}\n\n")
            else:
                prompt_parts.append(content)
        prompt = "\n".join(prompt_parts)

        t0 = time.time()
        resp = requests.post(
            QWEN_BASE_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            json={
                "model": self.model,
                "input": {"prompt": prompt},
                "parameters": {
                    "temperature": self.temperature,
                    "result_format": "text",
                },
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # DashScope response shape varies by model version
        text = (
            data.get("output", {}).get("text")
            or data.get("choices", [{}])[0].get("message", {}).get("content")
            or ""
        )
        latency = (time.time() - t0) * 1000

        usage = data.get("usage", {})
        return LLMResponse(
            text=text.strip(),
            model=self.model,
            backend="qwen",
            latency_ms=latency,
            usage=usage,
        )

    def _call_openai_compatible(self, messages: list[dict], base_url: str) -> LLMResponse:
        """Call OpenAI-compatible chat completions API (works for OpenAI and DeepSeek)."""
        t0 = time.time()
        resp = requests.post(
            base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"]
        latency = (time.time() - t0) * 1000

        usage = data.get("usage", {})
        return LLMResponse(
            text=text.strip(),
            model=self.model,
            backend=self._backend or "openai",
            latency_ms=latency,
            usage=usage,
        )

    @staticmethod
    def _detect_backend(model: str) -> str:
        if model in QWEN_MODELS or model.startswith("qwen"):
            return "qwen"
        if model in DEEPSEEK_MODELS or model.startswith("deepseek"):
            return "deepseek"
        return "openai"

    @staticmethod
    def _resolve_from_env(model: str) -> tuple[str | None, str | None]:
        """Find API key from environment variables.

        Priority: DEEPSEEK_API_KEY > QWEN_API_KEY > OPENAI_API_KEY
        """
        # DeepSeek
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            backend = "deepseek" if model in DEEPSEEK_MODELS or model.startswith("deepseek") else "openai"
            return deepseek_key, backend

        # Qwen
        qwen_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if qwen_key and (model in QWEN_MODELS or model.startswith("qwen")):
            return qwen_key, "qwen"

        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            return openai_key, "openai"

        # Cross-compatible fallbacks
        if qwen_key:
            return qwen_key, "qwen"
        if openai_key:
            return openai_key, "openai"
        if deepseek_key:
            return deepseek_key, "deepseek"

        return None, None

    @staticmethod
    def _no_api_fallback(prompt: str) -> str:
        """Return a fallback message when no API is configured."""
        return (
            "LLM not configured. Set DEEPSEEK_API_KEY, QWEN_API_KEY, or OPENAI_API_KEY "
            "environment variable.\n\n"
            "LLM-assisted research is optional. The agent will fall back to "
            "heuristic methods for factor discovery and parameter optimization."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> str:
    """Extract JSON from markdown code fences or surrounding text."""
    text = text.strip()
    # Remove markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening fence (```json or ```)
        lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # Try to find JSON object/array boundaries
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return text
    # Find first { or [
    for start_char in ("{", "["):
        idx = text.find(start_char)
        if idx >= 0:
            return text[idx:]
    return text


def create_default_client() -> LLMClient:
    """Create an LLMClient with auto-detected configuration.

    Auto-selects backend based on available API keys:
    DEEPSEEK_API_KEY → deepseek (deepseek-chat)
    QWEN_API_KEY → qwen (qwen-plus)
    OPENAI_API_KEY → openai (gpt-4o-mini)
    """
    return LLMClient()
