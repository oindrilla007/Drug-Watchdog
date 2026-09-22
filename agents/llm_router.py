"""
llm_router.py  ·  Drug Watchdog Phase 4
=========================================
Multi-provider LLM router with automatic fallback chain.

Chain (in order):
  1. Groq  — Llama 3.3 70B (primary, 500-800 tok/s, free tier 30 req/min)
  2. Groq  — Llama 4 Scout (vision-capable, used for image tasks)
  3. OpenRouter — DeepSeek-R1 (deep reasoning fallback, free tier)
  4. Ollama — Llama 3.2 3B (local, offline, zero cost, last resort)

Responsibilities
----------------
  • Route text completion requests to the best available provider
  • Route vision (image) requests to Groq Llama 4 Scout specifically
  • Handle 429 rate-limit retries with exponential backoff
  • Track which provider served each request (for logging)
  • Provide a streaming interface and a blocking interface
  • Raise RouterExhaustedError only when ALL providers fail

Usage
-----
  router = LLMRouter()

  # Text
  response = router.complete(
      messages=[{"role": "user", "content": "Explain warfarin CYP2C9 interaction"}],
      mode="clinical",        # "clinical" | "patient" | "reasoning"
      max_tokens=1500,
  )
  print(response.text)
  print(response.provider_used)  # e.g. "groq-llama-3.3-70b"

  # Vision (prescription scan, pill photo, lab report)
  response = router.vision_complete(
      image_b64="...",
      prompt="Extract all drug names and doses from this prescription",
  )
  print(response.text)   # structured JSON string
"""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import requests

try:
    from .env_loader import load_project_env
except ImportError:
    from env_loader import load_project_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── Config ──────────────────────────────────────────

load_project_env()

GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

GROQ_BASE_URL       = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_CHAT_URL     = f"{OLLAMA_BASE_URL}/api/chat"

# Model IDs
MODEL_GROQ_TEXT     = "llama-3.3-70b-versatile"      # primary reasoning
MODEL_GROQ_VISION   = "meta-llama/llama-4-scout-17b-16e-instruct"  # vision
MODEL_OPENROUTER    = "deepseek/deepseek-r1:free"    # fallback reasoning
MODEL_OLLAMA        = "llama3.2:3b"                  # local offline fallback

REQUEST_TIMEOUT     = 60    # seconds
MAX_RETRIES         = 2     # per provider before moving to next
RETRY_WAIT_S        = 2.0   # base wait on 429


# ─────────────────────────── Data classes ────────────────────────────────────

class CompletionMode(str, Enum):
    CLINICAL   = "clinical"    # technical, citations, CYP pathways
    PATIENT    = "patient"     # plain language, no jargon
    REASONING  = "reasoning"   # deep chain-of-thought (routes to DeepSeek first)
    VISION     = "vision"      # image understanding (routes to Llama 4 Scout)


@dataclass
class RouterResponse:
    text:          str
    provider_used: str
    model_used:    str
    latency_ms:    float
    tokens_in:     int  = 0
    tokens_out:    int  = 0
    raw:           Any  = field(default=None, repr=False)


class RouterExhaustedError(RuntimeError):
    """Raised when all providers in the fallback chain have failed."""
    pass


# ─────────────────────────── System prompts ──────────────────────────────────

SYSTEM_PROMPTS = {
    CompletionMode.CLINICAL: (
        "You are Drug Watchdog, a clinical decision support AI. "
        "Respond with precise medical language suitable for pharmacists and physicians. "
        "Always cite evidence using the provided citation keys like [FDA-1], [FAERS-2], [PUB-3]. "
        "Include CYP enzyme pathways, pharmacokinetic mechanisms, and severity justification. "
        "Never fabricate citations. If evidence is insufficient, say so explicitly."
    ),
    CompletionMode.PATIENT: (
        "You are Drug Watchdog, a medication safety assistant speaking directly to a patient. "
        "Use plain, friendly language. Avoid medical jargon. "
        "Explain what the interaction means in everyday terms and what the patient should do. "
        "Never cause panic — be reassuring but clear about risks. "
        "Never recommend stopping medication without telling the patient to consult their doctor first."
    ),
    CompletionMode.REASONING: (
        "You are Drug Watchdog, a clinical reasoning AI. "
        "Think step by step through the pharmacological mechanisms before reaching a conclusion. "
        "Show your reasoning chain. Consider patient-specific factors like organ function and comorbidities. "
        "Cite evidence inline. Flag any uncertainty explicitly."
    ),
    CompletionMode.VISION: (
        "You are Drug Watchdog's vision module. "
        "Extract structured medical information from the provided image. "
        "Return ONLY valid JSON — no preamble, no markdown fences. "
        "Be precise about drug names, doses, and units. "
        "If a value is illegible, set it to null rather than guessing."
    ),
}


# ─────────────────────────── Provider helpers ────────────────────────────────

def _openai_headers(api_key: str, extra: dict | None = None) -> dict:
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if extra:
        h.update(extra)
    return h


def _extract_text(data: dict) -> str:
    """Parse the 'content' field from a standard OpenAI-format response."""
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        return str(data)


def _token_counts(data: dict) -> tuple[int, int]:
    usage = data.get("usage", {})
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


# ─────────────────────────── LLM Router ──────────────────────────────────────

class LLMRouter:
    """
    Routes LLM completions across Groq → OpenRouter → Ollama with
    automatic fallback on rate limits, timeouts, and API errors.

    All public methods return RouterResponse.
    """

    def __init__(self):
        self._request_log: list[dict] = []
        log.info(
            "LLMRouter ready — Groq: %s  OpenRouter: %s  Ollama: %s",
            "✓" if GROQ_API_KEY else "✗ (no key)",
            "✓" if OPENROUTER_API_KEY else "✗ (no key)",
            OLLAMA_BASE_URL,
        )

    # ── Public: text completion ───────────────────────────────────────────────

    def complete(
        self,
        messages:   list[dict],
        mode:       CompletionMode = CompletionMode.CLINICAL,
        max_tokens: int            = 1500,
        temperature: float         = 0.2,
    ) -> RouterResponse:
        """
        Route a text completion request through the provider chain.

        Parameters
        ----------
        messages    : OpenAI-format message list (role/content dicts)
        mode        : Determines system prompt and preferred provider
        max_tokens  : Max tokens to generate
        temperature : 0.0–1.0 (lower = more deterministic, better for clinical)

        Returns
        -------
        RouterResponse with .text and .provider_used
        """
        # Prepend system prompt if not already present
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPTS[mode]}] + messages

        # Reasoning mode: try OpenRouter DeepSeek first for better CoT
        if mode == CompletionMode.REASONING and OPENROUTER_API_KEY:
            try:
                return self._call_openrouter(messages, max_tokens, temperature)
            except Exception as exc:
                log.warning("OpenRouter failed for reasoning mode: %s — falling back", exc)

        # Standard chain: Groq → OpenRouter → Ollama
        providers = [
            ("groq",        lambda: self._call_groq_text(messages, max_tokens, temperature)),
            ("openrouter",  lambda: self._call_openrouter(messages, max_tokens, temperature)),
            ("ollama",      lambda: self._call_ollama(messages, max_tokens, temperature)),
        ]

        return self._run_chain(providers)

    # ── Public: vision completion ─────────────────────────────────────────────

    def vision_complete(
        self,
        prompt:     str,
        image_b64:  str | None = None,
        image_url:  str | None = None,
        max_tokens: int        = 1000,
    ) -> RouterResponse:
        """
        Run a vision (image + text) completion via Groq Llama 4 Scout.
        Falls back to OpenRouter if Groq is rate-limited.

        Parameters
        ----------
        prompt      : Text instruction for the model
        image_b64   : Base64-encoded image string (provide this OR image_url)
        image_url   : Public image URL (provide this OR image_b64)
        max_tokens  : Max tokens to generate

        Returns
        -------
        RouterResponse — .text is typically a JSON string
        """
        if not image_b64 and not image_url:
            raise ValueError("Provide either image_b64 or image_url")

        # Build image content block
        if image_b64:
            # Auto-detect format from base64 header or default to jpeg
            fmt = "jpeg"
            if image_b64.startswith("/9j/"):
                fmt = "jpeg"
            elif image_b64.startswith("iVBORw0KGgo"):
                fmt = "png"
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/{fmt};base64,{image_b64}"},
            }
        else:
            image_content = {"type": "image_url", "image_url": {"url": image_url}}

        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[CompletionMode.VISION]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                image_content,
            ]},
        ]

        providers = [
            ("groq-vision",  lambda: self._call_groq_vision(messages, max_tokens)),
            ("openrouter",   lambda: self._call_openrouter(messages, max_tokens, 0.1)),
        ]
        return self._run_chain(providers)

    # ── Internal: provider calls ──────────────────────────────────────────────

    def _call_groq_text(
        self,
        messages:    list[dict],
        max_tokens:  int,
        temperature: float,
    ) -> RouterResponse:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set")

        t0 = time.perf_counter()
        for attempt in range(MAX_RETRIES):
            resp = requests.post(
                GROQ_BASE_URL,
                headers=_openai_headers(GROQ_API_KEY),
                json={
                    "model":       MODEL_GROQ_TEXT,
                    "messages":    messages,
                    "max_tokens":  max_tokens,
                    "temperature": temperature,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 429:
                wait = RETRY_WAIT_S * (2 ** attempt)
                log.warning("Groq 429 rate limit — waiting %.1fs (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            ti, to = _token_counts(data)
            return RouterResponse(
                text          = _extract_text(data),
                provider_used = "groq",
                model_used    = MODEL_GROQ_TEXT,
                latency_ms    = (time.perf_counter() - t0) * 1000,
                tokens_in     = ti,
                tokens_out    = to,
                raw           = data,
            )
        raise RuntimeError(f"Groq text: exceeded {MAX_RETRIES} retries")

    def _call_groq_vision(self, messages: list[dict], max_tokens: int) -> RouterResponse:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set")

        t0 = time.perf_counter()
        for attempt in range(MAX_RETRIES):
            resp = requests.post(
                GROQ_BASE_URL,
                headers=_openai_headers(GROQ_API_KEY),
                json={
                    "model":      MODEL_GROQ_VISION,
                    "messages":   messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 429:
                wait = RETRY_WAIT_S * (2 ** attempt)
                log.warning("Groq vision 429 — waiting %.1fs", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            ti, to = _token_counts(data)
            return RouterResponse(
                text          = _extract_text(data),
                provider_used = "groq-vision",
                model_used    = MODEL_GROQ_VISION,
                latency_ms    = (time.perf_counter() - t0) * 1000,
                tokens_in     = ti,
                tokens_out    = to,
                raw           = data,
            )
        raise RuntimeError(f"Groq vision: exceeded {MAX_RETRIES} retries")

    def _call_openrouter(
        self,
        messages:    list[dict],
        max_tokens:  int,
        temperature: float,
    ) -> RouterResponse:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        t0 = time.perf_counter()
        resp = requests.post(
            OPENROUTER_BASE_URL,
            headers=_openai_headers(
                OPENROUTER_API_KEY,
                {"HTTP-Referer": "https://drugwatchdog.app", "X-Title": "DrugWatchdog"},
            ),
            json={
                "model":       MODEL_OPENROUTER,
                "messages":    messages,
                "max_tokens":  max_tokens,
                "temperature": temperature,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        ti, to = _token_counts(data)
        return RouterResponse(
            text          = _extract_text(data),
            provider_used = "openrouter",
            model_used    = MODEL_OPENROUTER,
            latency_ms    = (time.perf_counter() - t0) * 1000,
            tokens_in     = ti,
            tokens_out    = to,
            raw           = data,
        )

    def _call_ollama(
        self,
        messages:    list[dict],
        max_tokens:  int,
        temperature: float,
    ) -> RouterResponse:
        t0 = time.perf_counter()
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model":    MODEL_OLLAMA,
                "messages": messages,
                "stream":   False,
                "options":  {"num_predict": max_tokens, "temperature": temperature},
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message", {}).get("content", "")
        return RouterResponse(
            text          = text,
            provider_used = "ollama",
            model_used    = MODEL_OLLAMA,
            latency_ms    = (time.perf_counter() - t0) * 1000,
            raw           = data,
        )

    # ── Chain runner ──────────────────────────────────────────────────────────

    def _run_chain(self, providers: list[tuple[str, Any]]) -> RouterResponse:
        errors: list[str] = []
        for name, call_fn in providers:
            try:
                result = call_fn()
                self._log_request(name, success=True, latency_ms=result.latency_ms)
                log.info("Routed to %s (%.0f ms)", result.provider_used, result.latency_ms)
                return result
            except Exception as exc:
                log.warning("Provider '%s' failed: %s", name, exc)
                errors.append(f"{name}: {exc}")
                self._log_request(name, success=False)

        raise RouterExhaustedError(
            "All LLM providers exhausted. Errors:\n" + "\n".join(errors)
        )

    def _log_request(self, provider: str, success: bool, latency_ms: float = 0.0):
        self._request_log.append({
            "provider":   provider,
            "success":    success,
            "latency_ms": latency_ms,
            "timestamp":  time.time(),
        })

    def provider_stats(self) -> dict:
        """Return success rates per provider from this session's request log."""
        stats: dict[str, dict] = {}
        for entry in self._request_log:
            p = entry["provider"]
            if p not in stats:
                stats[p] = {"total": 0, "success": 0, "avg_latency_ms": 0.0}
            stats[p]["total"] += 1
            if entry["success"]:
                stats[p]["success"] += 1
                stats[p]["avg_latency_ms"] = (
                    (stats[p]["avg_latency_ms"] * (stats[p]["success"] - 1) + entry["latency_ms"])
                    / stats[p]["success"]
                )
        return stats


# ─────────────────────────── Singleton ───────────────────────────────────────

_router: LLMRouter | None = None

def get_router() -> LLMRouter:
    """Return the shared LLMRouter singleton (lazy init)."""
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router


# ─────────────────────────── CLI smoke-test ──────────────────────────────────

if __name__ == "__main__":
    import sys

    router = LLMRouter()

    # Text test
    print("\n── Text completion test ──")
    try:
        resp = router.complete(
            messages=[{"role": "user", "content":
                "In one sentence, what is the CYP2C9 interaction between warfarin and aspirin?"}],
            mode=CompletionMode.CLINICAL,
            max_tokens=200,
        )
        print(f"Provider : {resp.provider_used}")
        print(f"Model    : {resp.model_used}")
        print(f"Latency  : {resp.latency_ms:.0f} ms")
        print(f"Response : {resp.text[:300]}")
    except RouterExhaustedError as e:
        print(f"All providers failed: {e}")

    # Vision test (only if image path provided)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\n── Vision test: {image_path} ──")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        try:
            resp = router.vision_complete(
                prompt="Extract all drug names, doses, and frequencies from this image. Return JSON.",
                image_b64=b64,
            )
            print(f"Provider : {resp.provider_used}")
            print(f"Response : {resp.text[:500]}")
        except RouterExhaustedError as e:
            print(f"Vision failed: {e}")

    print(f"\nProvider stats: {json.dumps(router.provider_stats(), indent=2)}")
