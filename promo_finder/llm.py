"""One tiny interface for every model: `LLM(provider, model).json(system, user) -> dict`.

Swapping models = changing two strings (CLI flags, env vars, or the app sidebar).
Responses are cached on disk, so re-runs are free and an interrupted run resumes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",  # free tier: 10 RPM / 500 RPD (3.8 Flash is only 20 RPD)
    "openai": "gpt-4o-mini",           # any OpenAI-compatible endpoint (set OPENAI_BASE_URL)
    "groq": "llama-3.3-70b-versatile",
    "anthropic": "claude-sonnet-4-5",
    "heuristic": "keyword-baseline",   # offline, no API key; used as a baseline + for tests
}

# Paid-tier list prices (USD per 1M tokens, input/output) for cost estimates only; free tier is $0.
# Per-model first (checked 2026-11-08 at aistudio.google.com/docs/pricing), then a per-provider fallback.
MODEL_PRICES = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),   # through 2026-12-31; $1.50 / $7.50 from 2027-01-01
}
PRICES = {
    "gemini": (0.30, 2.50),
    "openai": (0.15, 0.60),
    "groq": (0.59, 0.79),
    "anthropic": (3.00, 15.00),
    "heuristic": (0.0, 0.0),
}


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group())
        raise


class LLM:
    def __init__(self, provider: str = "gemini", model: str | None = None,
                 api_key: str | None = None, cache_dir: str | Path = ".cache",
                 temperature: float = 0.3, min_interval: float = 0.0, base_url: str | None = None):
        self.provider = provider.lower()
        self.base_url = base_url             # openai provider: any OpenAI-compatible endpoint
        self.model = model or DEFAULT_MODELS[self.provider]
        self.api_key = api_key
        self.temperature = temperature
        self.min_interval = min_interval     # seconds between calls (free-tier RPM limits)
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.usage = {"calls": 0, "cached": 0, "in_tokens": 0, "out_tokens": 0, "seconds": 0.0}
        self._last = 0.0
        self._lock = threading.Lock()
        self._client = None

    # ---------- public ----------
    def json(self, system: str, user: str, tag: str = "") -> dict:
        key = hashlib.sha256(f"{self.provider}|{self.model}|{self.temperature}|{system}|{user}".encode()).hexdigest()[:24]
        f = self.cache / f"{tag + '-' if tag else ''}{key}.json"
        if f.exists():
            self.usage["cached"] += 1
            return json.loads(f.read_text(encoding="utf-8"))["parsed"]

        last_err = None
        for attempt in range(5):
            try:
                with self._lock:  # space out calls across threads (free-tier RPM limits)
                    slot = max(time.time(), self._last + self.min_interval)
                    self._last = slot
                time.sleep(max(0.0, slot - time.time()))
                t0 = time.time()
                text, tin, tout = self._call(system, user)
                parsed = _extract_json(text)
                with self._lock:
                    self.usage["calls"] += 1
                    self.usage["in_tokens"] += tin
                    self.usage["out_tokens"] += tout
                    self.usage["seconds"] += time.time() - t0
                f.write_text(json.dumps({"model": self.model, "raw": text, "parsed": parsed}, indent=1), encoding="utf-8")
                return parsed
            except Exception as e:  # rate limits, transient errors, bad JSON
                last_err = e
                msg = str(e).lower()
                if any(k in msg for k in ("api key", "api_key", "401", "403", "permission", "not found", "404")):
                    raise  # bad key / bad model name: retrying won't help
                if "per day" in msg or "perday" in msg or "requests_per_day" in msg:
                    raise  # daily quota gone: retrying for minutes won't help; the cache lets you resume tomorrow
                backoff = 20 * (attempt + 1) if ("429" in msg or "quota" in msg or "rate" in msg) else 2 ** attempt
                time.sleep(backoff)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    def cost_usd(self) -> float:
        pin, pout = MODEL_PRICES.get(self.model) or PRICES.get(self.provider, (0, 0))
        return self.usage["in_tokens"] / 1e6 * pin + self.usage["out_tokens"] / 1e6 * pout

    # ---------- providers ----------
    def _call(self, system: str, user: str) -> tuple[str, int, int]:
        p = self.provider
        if p == "gemini":
            from google import genai
            from google.genai import types
            if self._client is None:
                self._client = genai.Client(api_key=self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
            r = self._client.models.generate_content(
                model=self.model, contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system, temperature=self.temperature,
                    response_mime_type="application/json"))
            u = r.usage_metadata
            return r.text, (u.prompt_token_count or 0), (u.candidates_token_count or 0) + (getattr(u, 'thoughts_token_count', 0) or 0)

        if p in ("openai", "groq"):
            from openai import OpenAI
            if self._client is None:
                if p == "groq":
                    self._client = OpenAI(api_key=self.api_key or os.getenv("GROQ_API_KEY"),
                                          base_url="https://api.groq.com/openai/v1")
                else:
                    self._client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
                                          base_url=self.base_url or os.getenv("OPENAI_BASE_URL") or None)
            r = self._client.chat.completions.create(
                model=self.model, temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens

        if p == "anthropic":
            import anthropic
            if self._client is None:
                self._client = anthropic.Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
            r = self._client.messages.create(
                model=self.model, max_tokens=4096, temperature=self.temperature, system=system,
                messages=[{"role": "user", "content": user + "\n\nReturn only JSON."}])
            return r.content[0].text, r.usage.input_tokens, r.usage.output_tokens

        if p == "heuristic":
            from .heuristic import respond
            return json.dumps(respond(system, user)), 0, 0

        raise ValueError(f"Unknown provider: {p}")
