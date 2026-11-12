"""Provider adapters over plain HTTP (no SDKs): one `complete()` per provider, same return shape.

Each returns a dict: text, finish_reason, tokens_in, tokens_out, tokens_thinking, model_version.
Retryable failures raise RetryableError (rate limits, overload, timeouts); others raise ProviderError.
"""

from __future__ import annotations

import os

import httpx


class ProviderError(Exception):
    pass


class QuotaExhausted(ProviderError):
    """Daily free-tier quota used up: retrying today is pointless, stop calling this model."""


class RetryableError(ProviderError):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


def _raise_for(r: httpx.Response) -> None:
    if r.status_code == 200:
        return
    try:
        msg = r.json().get("error", {})
        msg = msg.get("message", str(msg)) if isinstance(msg, dict) else str(msg)
    except Exception:
        msg = r.text[:300]
    retry_after = None
    if "retry-after" in r.headers:
        try:
            retry_after = float(r.headers["retry-after"])
        except ValueError:
            pass
    if r.status_code == 429 and "PerDay" in r.text:
        raise QuotaExhausted(f"HTTP 429 daily quota exhausted: {msg[:150]}")
    if r.status_code in (408, 429, 500, 502, 503, 504, 529):
        raise RetryableError(f"HTTP {r.status_code}: {msg[:200]}", retry_after)
    raise ProviderError(f"HTTP {r.status_code}: {msg[:300]}")


async def gemini(client: httpx.AsyncClient, model_id: str, prompt: str, params: dict) -> dict:
    gen = {"temperature": params.get("temperature", 1.0), "maxOutputTokens": params.get("max_output_tokens", 400)}
    if "thinking_level" in params:
        gen["thinkingConfig"] = {"thinkingLevel": params["thinking_level"]}
    if "thinking_budget" in params:
        gen["thinkingConfig"] = {"thinkingBudget": params["thinking_budget"]}
    r = await client.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        json={"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gen},
    )
    _raise_for(r)
    j = r.json()
    cand = (j.get("candidates") or [{}])[0]
    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    u = j.get("usageMetadata", {})
    return {
        "text": text,
        "finish_reason": cand.get("finishReason") or j.get("promptFeedback", {}).get("blockReason"),
        "tokens_in": u.get("promptTokenCount"),
        "tokens_out": u.get("candidatesTokenCount"),
        "tokens_thinking": u.get("thoughtsTokenCount"),
        "model_version": j.get("modelVersion"),
    }


async def openai(client: httpx.AsyncClient, model_id: str, prompt: str, params: dict) -> dict:
    body = {"model": model_id, "input": prompt, "max_output_tokens": params.get("max_output_tokens", 400)}
    if "temperature" in params:
        body["temperature"] = params["temperature"]
    if "reasoning_effort" in params:
        body["reasoning"] = {"effort": params["reasoning_effort"]}
    r = await client.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json=body,
    )
    _raise_for(r)
    j = r.json()
    text = "".join(
        c.get("text", "") for o in j.get("output", []) if o.get("type") == "message" for c in o.get("content", [])
    )
    u = j.get("usage", {})
    return {
        "text": text,
        "finish_reason": j.get("status"),
        "tokens_in": u.get("input_tokens"),
        "tokens_out": u.get("output_tokens"),
        "tokens_thinking": (u.get("output_tokens_details") or {}).get("reasoning_tokens"),
        "model_version": j.get("model"),
    }


_anthropic_client = None


async def anthropic(client: httpx.AsyncClient, model_id: str, prompt: str, params: dict) -> dict:
    """Official SDK. Retries are left to our runner (max_retries=0) so attempts are logged uniformly.

    Claude Haiku 5.5: temperature defaults to 1.0 and non-default values are rejected, so we don't send it;
    thinking is on by default and is switched off with {"type": "disabled"} (allowed at the default effort).
    """
    import anthropic as sdk

    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = sdk.AsyncAnthropic(max_retries=0, timeout=90.0)
    kwargs = {
        "model": model_id,
        "max_tokens": params.get("max_output_tokens", 400),
        "messages": [{"role": "user", "content": prompt}],
    }
    if params.get("thinking"):
        kwargs["thinking"] = {"type": params["thinking"]}
    try:
        msg = await _anthropic_client.messages.create(**kwargs)
    except sdk.RateLimitError as e:
        ra = e.response.headers.get("retry-after")
        raise RetryableError(f"HTTP 429: {e.message}", float(ra) if ra else None) from e
    except sdk.APIStatusError as e:
        if e.status_code >= 500 or e.status_code == 408:
            raise RetryableError(f"HTTP {e.status_code}: {e.message}") from e
        raise ProviderError(f"HTTP {e.status_code}: {e.message}") from e
    except sdk.APIConnectionError as e:  # includes timeouts
        raise RetryableError(f"{type(e).__name__}: {e}") from e
    return {
        "text": "".join(b.text for b in msg.content if b.type == "text"),
        "finish_reason": msg.stop_reason,
        "tokens_in": msg.usage.input_tokens,
        "tokens_out": msg.usage.output_tokens,
        "tokens_thinking": None,  # thinking disabled; output_tokens would include it otherwise
        "model_version": msg.model,
    }


PROVIDERS = {"gemini": gemini, "openai": openai, "anthropic": anthropic}
KEY_ENV = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
