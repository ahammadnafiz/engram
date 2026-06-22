#!/usr/bin/env python3
"""Real token-counting and end-to-end cost accounting for the benchmarks.

Two distinct measurements, deliberately kept separate:

1. REAL BILLED COST -- summed from each LLM response's ``usage`` field
   (Anthropic returns exact ``input_tokens`` / ``output_tokens``). This is what
   the provider actually charged for the composer + judge calls. Never
   estimated.

2. CONTEXT-COMPRESSION BASELINE -- token counts via ``tiktoken`` for the
   evidence block we actually sent vs the *full conversation* a naive
   "stuff everything into context" agent would send. Those full-context calls
   are never made, so there is no real ``usage`` to read -- tiktoken is the
   only available estimate. ``tiktoken`` has no official Claude encoding;
   ``o200k_base`` is used as a close, widely-cited proxy.

Pricing below is base input / output USD per million tokens, transcribed from
Anthropic's public pricing table. Cache rates are recorded for completeness but
the benchmarks do not use prompt caching, so cost uses base input + output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tiktoken

# --------------------------------------------------------------------------
# Pricing: USD per 1,000,000 tokens. (base input, output, 5m cache write,
# 1h cache write, cache hit/refresh). Source: Anthropic model pricing table.
# --------------------------------------------------------------------------
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-fable-5": {"input": 10.0, "output": 50.0, "cache_5m": 12.50, "cache_1h": 20.0, "cache_hit": 1.0},
    "claude-mythos-5": {"input": 10.0, "output": 50.0, "cache_5m": 12.50, "cache_1h": 20.0, "cache_hit": 1.0},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0, "cache_5m": 6.25, "cache_1h": 10.0, "cache_hit": 0.50},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0, "cache_5m": 6.25, "cache_1h": 10.0, "cache_hit": 0.50},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0, "cache_5m": 6.25, "cache_1h": 10.0, "cache_hit": 0.50},
    "claude-opus-4-5": {"input": 5.0, "output": 25.0, "cache_5m": 6.25, "cache_1h": 10.0, "cache_hit": 0.50},
    "claude-opus-4-1": {"input": 15.0, "output": 75.0, "cache_5m": 18.75, "cache_1h": 30.0, "cache_hit": 1.50},
    "claude-opus-4": {"input": 15.0, "output": 75.0, "cache_5m": 18.75, "cache_1h": 30.0, "cache_hit": 1.50},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_5m": 3.75, "cache_1h": 6.0, "cache_hit": 0.30},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0, "cache_5m": 3.75, "cache_1h": 6.0, "cache_hit": 0.30},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0, "cache_5m": 3.75, "cache_1h": 6.0, "cache_hit": 0.30},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_5m": 1.25, "cache_1h": 2.0, "cache_hit": 0.10},
    "claude-haiku-3-5": {"input": 0.80, "output": 4.0, "cache_5m": 1.0, "cache_1h": 1.60, "cache_hit": 0.08},
}

_FALLBACK_MODEL = "claude-sonnet-4-6"
_TIKTOKEN_ENCODING = "o200k_base"

_enc = None


def _encoder() -> tiktoken.Encoding:
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding(_TIKTOKEN_ENCODING)
    return _enc


def count_tokens(text: str) -> int:
    """Real token count via tiktoken (o200k_base proxy for Claude)."""
    if not text:
        return 0
    return len(_encoder().encode(text, disallowed_special=()))


def _rates(model: str) -> dict[str, float]:
    """Resolve a model id (possibly with a date suffix) to its pricing row."""
    m = (model or "").lower()
    # Longest key first so "claude-opus-4-8" wins over "claude-opus-4".
    for key in sorted(MODEL_PRICING, key=len, reverse=True):
        if m.startswith(key) or key in m:
            return MODEL_PRICING[key]
    return MODEL_PRICING[_FALLBACK_MODEL]


def normalize_usage(usage: dict[str, int] | None) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from a provider usage dict.

    Anthropic uses input_tokens/output_tokens; OpenAI uses
    prompt_tokens/completion_tokens. Returns (0, 0) when usage is absent.
    """
    if not usage:
        return 0, 0
    inp = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    out = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    return int(inp), int(out)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Billed USD for a single call given real input/output token counts."""
    r = _rates(model)
    return input_tokens / 1e6 * r["input"] + output_tokens / 1e6 * r["output"]


@dataclass
class CostAccumulator:
    """Aggregates real billed usage across every LLM call in a run.

    ``add()`` is called once per composer / judge response with its real
    ``usage`` dict. ``summary()`` returns a JSON-able dict with API-call
    counts, real token totals, real cost, and -- using the tiktoken
    full-context baseline -- the projected cost of a naive full-context agent
    and the compression savings.
    """

    composer_calls: int = 0
    judge_calls: int = 0
    composer_in: int = 0
    composer_out: int = 0
    judge_in: int = 0
    judge_out: int = 0
    composer_cost: float = 0.0
    judge_cost: float = 0.0
    composer_model: str = _FALLBACK_MODEL
    judge_model: str = _FALLBACK_MODEL
    # Per-question tiktoken measurements for the compression baseline.
    evidence_tokens: list[int] = field(default_factory=list)
    full_context_tokens: list[int] = field(default_factory=list)
    # Projected full-context (naive) input tokens per composer call:
    # real_input - evidence + full_conversation. Captured per question so the
    # baseline reuses the real system+question overhead.
    baseline_input_tokens: list[int] = field(default_factory=list)

    def add_composer(self, model: str, usage: dict[str, int] | None) -> None:
        inp, out = normalize_usage(usage)
        self.composer_calls += 1
        self.composer_in += inp
        self.composer_out += out
        self.composer_cost += cost_usd(model, inp, out)
        if model:
            self.composer_model = model

    def add_judge(self, model: str, usage: dict[str, int] | None) -> None:
        inp, out = normalize_usage(usage)
        self.judge_calls += 1
        self.judge_in += inp
        self.judge_out += out
        self.judge_cost += cost_usd(model, inp, out)
        if model:
            self.judge_model = model

    def add_compression(
        self,
        *,
        evidence_text: str,
        real_input_tokens: int,
        full_context_text: str = "",
        full_context_tokens: int | None = None,
    ) -> None:
        """Record one question's compression measurement (tiktoken).

        Pass ``full_context_tokens`` when the full-conversation token count is
        precomputed (e.g. once per conversation) to avoid re-tokenizing the
        whole history per question; otherwise it is counted from
        ``full_context_text``.
        """
        ev = count_tokens(evidence_text)
        fc = full_context_tokens if full_context_tokens is not None else count_tokens(full_context_text)
        self.evidence_tokens.append(ev)
        self.full_context_tokens.append(fc)
        # Naive baseline input = real prompt with the evidence block swapped
        # for the entire conversation: real_input - evidence + full_context.
        self.baseline_input_tokens.append(max(real_input_tokens - ev + fc, fc))

    def summary(self) -> dict[str, object]:
        n = max(self.composer_calls, 1)
        real_cost = self.composer_cost + self.judge_cost
        in_rate = _rates(self.composer_model)["input"]
        out_rate = _rates(self.composer_model)["output"]
        # Projected naive full-context composer cost: same per-question output,
        # but the whole conversation in the input every time.
        baseline_in = sum(self.baseline_input_tokens)
        baseline_cost = baseline_in / 1e6 * in_rate + self.composer_out / 1e6 * out_rate
        avg_ev = round(sum(self.evidence_tokens) / len(self.evidence_tokens)) if self.evidence_tokens else 0
        avg_fc = round(sum(self.full_context_tokens) / len(self.full_context_tokens)) if self.full_context_tokens else 0
        return {
            "pricing_source": "Anthropic public pricing (base input/output USD per MTok)",
            "token_counter": f"tiktoken/{_TIKTOKEN_ENCODING} (compression baseline); provider usage (billed)",
            "api_calls": {
                "composer": self.composer_calls,
                "judge": self.judge_calls,
                "total": self.composer_calls + self.judge_calls,
            },
            "billed_tokens": {
                "composer_input": self.composer_in,
                "composer_output": self.composer_out,
                "judge_input": self.judge_in,
                "judge_output": self.judge_out,
                "total": self.composer_in + self.composer_out + self.judge_in + self.judge_out,
            },
            "real_cost_usd": {
                "composer": round(self.composer_cost, 4),
                "judge": round(self.judge_cost, 4),
                "total": round(real_cost, 4),
                "per_question": round(real_cost / n, 6),
            },
            "context_compression": {
                "avg_evidence_tokens": avg_ev,
                "avg_full_context_tokens": avg_fc,
                "compression_pct": round((1 - avg_ev / avg_fc) * 100, 1) if avg_fc else 0.0,
            },
            "full_context_baseline": {
                "note": "Projected cost if the entire conversation were sent as "
                "context for every question (no retrieval), composer only.",
                "projected_composer_cost_usd": round(baseline_cost, 4),
                "our_composer_cost_usd": round(self.composer_cost, 4),
                "savings_pct": round((1 - self.composer_cost / baseline_cost) * 100, 1)
                if baseline_cost
                else 0.0,
                "cost_multiplier": round(baseline_cost / self.composer_cost, 1)
                if self.composer_cost
                else 0.0,
            },
        }
