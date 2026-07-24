"""Provider-agnostic LLM client — one interface, swappable providers.

The chat/answer and eval/judge layers depend only on `complete()`. Providers plug in
behind it; credentials come from env, never committed. Every call records full telemetry
(prompt, response, model, token counts, estimated cost, latency) onto the active trace so
LLM usage is observable.

Providers:
  anthropic  — Claude (default). claude-opus-4-8 for judge/hard, claude-haiku-4-5 routine.
               No sampling params are sent (Opus 4.8 rejects temperature/top_p with a 400).
  openai     — optional, if OPENAI_API_KEY + openai installed.
  echo       — deterministic offline stub (NO API key). Lets the whole pipeline + eval run
               and be tested without network/keys; it extractively answers from the given
               context so groundedness plumbing is exercisable. Not a real model.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

from ..config import settings
from ..observability import logging as slog
from ..observability.cost import estimate_cost
from ..observability.tracing import LLMCall, Trace


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _rough_tokens(s: str) -> int:
    # fallback estimate only (~4 chars/token) when a provider doesn't report usage
    return max(1, len(s) // 4)


class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        self._client = None

    # ---- providers -------------------------------------------------------------

    def _anthropic(self, system: str, user: str, model: str, max_tokens: int):
        import anthropic
        if self._client is None:
            # prefer the key loaded from .env/config; fall back to the SDK's env lookup
            key = settings.anthropic_api_key or None
            self._client = anthropic.Anthropic(api_key=key)
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            text = "[model refused to answer]"
        else:
            text = "".join(b.text for b in resp.content if b.type == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens

    def _openai(self, system: str, user: str, model: str, max_tokens: int):
        from openai import OpenAI
        if self._client is None:
            self._client = OpenAI()
        resp = self._client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=settings.llm_temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        u = resp.usage
        return resp.choices[0].message.content, u.prompt_tokens, u.completion_tokens

    def _echo(self, system: str, user: str, model: str, max_tokens: int):
        """Deterministic offline answerer. Extracts the best-matching context line and
        returns it with its citation id, so grounded-answer plumbing works keyless."""
        cites = re.findall(r"\[(PID_[AB]:[0-9a-f]+|DELTA:[0-9a-z]+)\]", user)
        # naive: echo the first context block's citation + a canned grounded sentence
        first_cite = cites[0] if cites else "NONE"
        # pull the question (last line after 'Question:')
        m = re.search(r"Question:\s*(.*)", user)
        q = (m.group(1).strip() if m else "")[:120]
        body = (f"Based on the retrieved context, here is the grounded answer to "
                f"\"{q}\". See the cited sources. [{first_cite}]" if cites
                else "I don't have enough grounded context to answer that. [no-citation]")
        return body, _rough_tokens(system + user), _rough_tokens(body)

    # ---- public ---------------------------------------------------------------

    def complete(self, system: str, user: str, *, purpose: str = "",
                 model: Optional[str] = None, max_tokens: Optional[int] = None,
                 trace: Optional[Trace] = None) -> LLMResult:
        model = model or self.model
        max_tokens = max_tokens or settings.llm_max_tokens
        t0 = time.perf_counter()
        try:
            if self.provider == "anthropic":
                text, itok, otok = self._anthropic(system, user, model, max_tokens)
            elif self.provider == "openai":
                text, itok, otok = self._openai(system, user, model, max_tokens)
            elif self.provider == "echo":
                model = "echo"
                text, itok, otok = self._echo(system, user, model, max_tokens)
            else:
                raise ValueError(f"unknown provider {self.provider}")
        except Exception as e:
            slog.error("llm.error", provider=self.provider, model=model, error=str(e))
            raise
        dt = round((time.perf_counter() - t0) * 1000, 2)
        cost, known = estimate_cost(model, itok, otok)
        if trace is not None:
            trace.record_llm(LLMCall(model=model, prompt=(system + "\n\n" + user),
                                     response=text, input_tokens=itok, output_tokens=otok,
                                     cost_usd=cost, cost_known=known, duration_ms=dt,
                                     purpose=purpose))
        return LLMResult(text=text, model=model, input_tokens=itok,
                         output_tokens=otok, cost_usd=cost)
