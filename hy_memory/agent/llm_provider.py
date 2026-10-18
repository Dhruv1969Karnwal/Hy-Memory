"""
Agent Memory - LLM Provider

Unified LLM call interface supporting multiple model backends.

Features:
- Unified call interface
- Multiple model support (OpenAI, Hunyuan and others)
- Request retry and error handling
- Token statistics
- Model routing

Example:
    provider = LLMProvider(config)
    
    # Call the LLM. Do not pass temperature; config.llm decides it uniformly (including the "pass nothing" case,
    # see _resolve_temperature). Hardcoding it at a call site bypasses user configuration.)
    response = await provider.complete(
        prompt="Summarize the following content: ...",
        max_tokens=500,
    )
    
    print(response.content)
    print(response.tokens_used)
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import asyncio
import hashlib
import logging
import os
import weakref

from ..config import MemoryConfig
from ..core.workload import llm_slot

logger = logging.getLogger(__name__)


def _log_llm_content_enabled() -> bool:
    """Return whether sensitive LLM prompt/response bodies may be logged."""
    return os.getenv("MEMORY_LOG_LLM_CONTENT", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _content_fingerprint(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()[:16]


class LLMRetryError(RuntimeError):
    """All LLM retries failed. attempts carries the error detail of each attempt so the layers above can persist it into
    the EXTRACT/RECONCILE pipeline trace (parsed.attempts)."""

    def __init__(self, message: str, attempts: List[Dict[str, Any]]):
        super().__init__(message)
        self.attempts = attempts


def _attempt_record(attempt_no: int, exc: BaseException) -> Dict[str, Any]:
    """Normalize one attempt's exception into a persistable dict (including the HTTP status code and similar fields)."""
    import time as _t
    rec: Dict[str, Any] = {
        "attempt": attempt_no,
        "error_type": type(exc).__name__,
        "error": str(exc)[:2000],
        "ts": _t.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # Best-effort extraction of the HTTP status code / response body (OpenAI / requests / httpx styles)
    for _attr in ("status_code", "code"):
        _v = getattr(exc, _attr, None)
        if _v is not None:
            rec["status_code"] = _v
            break
    _resp = getattr(exc, "response", None)
    if _resp is not None:
        try:
            rec.setdefault("status_code", getattr(_resp, "status_code", None))
            _body = getattr(_resp, "text", None)
            if _body:
                body = str(_body)
                rec["response_body"] = (
                    body[:1000] if _log_llm_content_enabled()
                    else f"<redacted len={len(body)} sha256={_content_fingerprint(body)}>"
                )
        except Exception:
            pass
    return rec


def _is_retryable_llm_error(exc: BaseException) -> bool:
    """Retry transient transport errors and 408/429/5xx; fail fast on other 4xx."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 408 or status == 429 or status >= 500
    return isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)) or any(
        token in str(exc).lower()
        for token in ("timeout", "timed out", "connection", "read operation")
    )


def _build_retry_error(
    max_retries: int, model: str, backend: str,
    attempts: List[Dict[str, Any]], last_exc: Optional[BaseException],
) -> "LLMRetryError":
    actual_attempts = len(attempts)
    msg = (
        f"LLM call failed after {actual_attempts} attempt(s) "
        f"(configured max={max_retries}) "
        f"(model={model} backend={backend}); "
        f"last error: {type(last_exc).__name__}: {last_exc}"
    )
    err = LLMRetryError(msg, attempts)
    err.__cause__ = last_exc
    return err


def _parse_tool_calls_list(raw_calls: list, _json) -> List[Dict[str, Any]]:
    """
    Parse a tool_calls list.

    The original complete structure is kept; only arguments is normalized (dict → JSON str).
    """
    result = []
    for tc in raw_calls:
        tc_id = tc.get("id") or ""
        if not tc_id:
            logger.error(
                f"[llm] tool_call missing 'id'! raw: "
                f"{_json.dumps(tc, ensure_ascii=False, default=str)[:500]}"
            )

        # Keep the original tc structure intact; only normalize arguments
        tc_copy = dict(tc)  # shallow copy keeps every field (signature and so on)
        fn = tc_copy.get("function", {})
        if isinstance(fn, dict):
            args = fn.get("arguments", "")
            if isinstance(args, dict):
                fn["arguments"] = _json.dumps(args, ensure_ascii=False)
            tc_copy["function"] = fn

        # Ensure the basic fields exist
        tc_copy.setdefault("id", tc_id)
        tc_copy.setdefault("type", "function")

        result.append(tc_copy)
    return result


def _serialize_tool_calls(tool_calls_raw) -> Optional[List[Dict[str, Any]]]:
    """OpenAI SDK tool_call objects → a plain dict list (easy to pass around / persist).

    A single serialization failure only skips that entry and warns; it never fails the whole LLM call.
    """
    if not tool_calls_raw:
        return None
    result: List[Dict[str, Any]] = []
    for tc in tool_calls_raw:
        try:
            fn = getattr(tc, "function", None)
            result.append({
                "id": getattr(tc, "id", "") or "",
                "type": getattr(tc, "type", "function") or "function",
                "function": {
                    "name": getattr(fn, "name", "") if fn else "",
                    "arguments": getattr(fn, "arguments", "") if fn else "",
                },
            })
        except Exception as e:
            logger.warning(f"[llm] failed to serialize tool_call: {e}")
    return result or None


class LLMBackend(Enum):
    """LLM protocol backend."""
    OPENAI = "openai"


@dataclass
class LLMResponse:
    """
    LLM response
    """
    content: str
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    finish_reason: str = ""
    response_id: str = ""
    choices_count: int = 0
    # Chain-of-thought content and token count for reasoning models (hy3 / kimi-k2.5 / deepseek-v4 and so on).
    # When content is empty but reasoning_tokens fill up max_tokens, reasoning consumed the whole output budget,
    # so thinking must be disabled (hy3: reasoning_effort=no_think) or agent_max_tokens raised.
    reasoning_content: str = ""
    reasoning_tokens: int = 0
    # function/tool calling: filled when the LLM decides to call a tool, in OpenAI shape:
    #   [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "<json str>"}}]
    # callers should parse it with tools.base.parse_tool_calls_from_json
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class LLMConfig:
    """
    LLM configuration
    """
    backend: LLMBackend = LLMBackend.OPENAI
    model: str = "gpt-3.5-turbo"
    api_key: str = ""
    base_url: str = ""
    # None = **no temperature in the request at all**.
    # Some gateways (e.g. Opus 5) accept only temperature=1 or nothing at all; sending 0/0.1/0.3 returns 400.
    temperature: Optional[float] = 0.7
    timeout: int = 30
    max_retries: int = 5
    retry_delay: float = 1.0  # initial delay, increasing on each retry
    extra_headers: Optional[Dict[str, str]] = None
    extra_body: Optional[Dict[str, Any]] = None


class LLMProvider:
    """
    LLM provider

    Unified LLM call interface.

    _extract_response: safely extracts choices[0].message, giving a clear error when the API returns empty choices
    instead of 'NoneType' object is not subscriptable.
    """

    @staticmethod
    def _extract_response(response) -> tuple:
        """
        Extract message, usage and finish_reason from an OpenAI-compatible response.
        Raises a clear ValueError when choices is empty.

        Returns:
            (msg, usage, finish_reason, choices_count)
        """
        choices = getattr(response, "choices", None)
        if not choices:
            # Try to collect more diagnostic information
            raw = ""
            try:
                raw = str(response)[:500]
            except Exception:
                pass
            raise ValueError(
                f"LLM returned empty choices (no completion). "
                f"Response: {raw}"
            )
        msg = choices[0].message
        usage = response.usage
        finish_reason = choices[0].finish_reason
        return msg, usage, finish_reason, len(choices)

    @staticmethod
    def _extract_reasoning(msg, usage) -> tuple:
        """
        Extract the reasoning model's chain-of-thought content and token count.

        Field placement differs across platforms:
        - hy3 / kimi-k2.5: message.reasoning_content, also accepting
          message.provider_specific_fields.reasoning_content
        - reasoning_tokens：usage.completion_tokens_details.reasoning_tokens，
          some gateways flatten it onto usage.reasoning_tokens

        Returns:
            (reasoning_content, reasoning_tokens)
        """
        rc = getattr(msg, "reasoning_content", None) or ""
        if not rc:
            psf = getattr(msg, "provider_specific_fields", None)
            if isinstance(psf, dict):
                rc = psf.get("reasoning_content") or ""

        rt = 0
        if usage is not None:
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                rt = getattr(details, "reasoning_tokens", 0) or 0
            if not rt:
                rt = getattr(usage, "reasoning_tokens", 0) or 0
        return rc, int(rt or 0)

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize the LLM Provider
        
        Args:
            config: configuration object
        """
        self.config = config or MemoryConfig.from_env()
        self._llm_config = self._build_llm_config()
        self._client = None
        self._clients = weakref.WeakKeyDictionary()
        
        # Statistics
        self._total_calls = 0
        self._total_tokens = 0
        self._errors = 0
        
        logger.debug(f"LLMProvider initialized, backend={self._llm_config.backend.value}")
    
    def _build_llm_config(self) -> LLMConfig:
        """Build the LLM configuration from settings"""
        llm = self.config.llm
        return LLMConfig(
            backend=LLMBackend.OPENAI,
            model=llm.model,
            api_key=llm.api_key,
            base_url=llm.base_url,
            # No longer coercing None to 0.7: None is a legitimate way of saying "do not send it", and coercing it would make
            # settings such as MEMORY_LLM_TEMPERATURE=none stop working
            temperature=(
                None if getattr(llm, "omit_temperature", False)
                else llm.temperature
            ),
            timeout=llm.timeout or 120,
            max_retries=llm.max_retries if llm.max_retries is not None else 5,
            retry_delay=llm.retry_delay if llm.retry_delay is not None else 1.0,
            extra_headers=llm.extra_headers,
            extra_body=llm.extra_body,
        )
    
    def _resolve_temperature(self, temperature: Optional[float]) -> Optional[float]:
        """Converge the call site's temperature onto the **configuration**.

        Rule: use the value the call site passed; when nothing is passed (None) use `config.llm.temperature`.
        When the configuration itself is None, return None and `_chat_completions` omits the parameter entirely.

        Why it must converge in one place: call sites used to hardcode 0.1/0.2/0.3/0.4, and `complete()`'s
        default was 0.7, so the user's `MEMORY_LLM_TEMPERATURE` only took effect on a few paths.
        Switching to a gateway that accepts only certain temperatures (Opus 5 takes 1 or nothing) made all those hardcoded paths return 400.
        """
        if temperature is not None:
            return temperature
        return getattr(self._llm_config, "temperature", None)

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: Optional[float] = None,
        stop: List[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Call the LLM to complete text generation

        Args:
            prompt: the prompt
            max_tokens: maximum number of tokens
            temperature: temperature parameter
            stop: list of stop words
            tools: optional list of OpenAI function-calling style tool schemas
                   （[{"type":"function","function":{"name":..., "parameters":...}}, ...]）。
                   When passed, the LLM may return tool_calls in its response and the caller decides how to dispatch them.
            tool_choice: optional OpenAI-style tool_choice ("auto" / "none" / {"type":"function","function":{"name":...}})
            **kwargs: other parameters

        Returns:
            LLM response (response.tool_calls may be non-empty on success)
        """
        _last_exc: Optional[BaseException] = None
        _attempts: List[Dict[str, Any]] = []
        for attempt in range(self._llm_config.max_retries):
            try:
                if attempt == 0:
                    logger.debug(
                        f"[llm] model={self._llm_config.model} "
                        f"backend={self._llm_config.backend.value} "
                        f"base_url={self._llm_config.base_url or 'default'} "
                        f"prompt_len={len(prompt)} tools={len(tools) if tools else 0}"
                    )
                if _log_llm_content_enabled():
                    logger.debug(f"[llm] prompt:\n{prompt}")
                else:
                    logger.debug(
                        "[llm] prompt content hidden; len=%d sha256=%s",
                        len(prompt), _content_fingerprint(prompt),
                    )

                response = await self._call_llm(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    tools=tools,
                    tool_choice=tool_choice,
                    **kwargs
                )

                self._total_calls += 1
                self._total_tokens += response.tokens_used

                logger.debug(
                    f"[llm] response: {len(response.content)} chars, "
                    f"tokens={response.tokens_used}, "
                    f"tool_calls={len(response.tool_calls) if response.tool_calls else 0}"
                )
                if _log_llm_content_enabled():
                    logger.debug(f"[llm] response content: {response.content!r}")
                else:
                    logger.debug(
                        "[llm] response content hidden; len=%d sha256=%s",
                        len(response.content), _content_fingerprint(response.content),
                    )

                return response

            except Exception as e:
                self._errors += 1
                _last_exc = e
                _attempts.append(_attempt_record(attempt + 1, e))
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self._llm_config.max_retries}): "
                    f"{type(e).__name__}: {e}"
                )

                if not _is_retryable_llm_error(e):
                    break

                if attempt < self._llm_config.max_retries - 1:
                    delay = self._llm_config.retry_delay * (attempt + 1)
                    await asyncio.sleep(delay)

        # Every retry failed -- the exception carries each attempt's error (for the layers above to persist into the EXTRACT trace)
        raise _build_retry_error(
            self._llm_config.max_retries, self._llm_config.model,
            self._llm_config.backend.value, _attempts, _last_exc,
        )

    def _resolve_endpoint(self) -> tuple:
        """Return the configured OpenAI-compatible endpoint unchanged."""
        return self._llm_config.api_key, (self._llm_config.base_url or None)

    async def _call_llm(
        self,
        prompt: str,
        max_tokens: int,
        temperature: Optional[float] = None,
        stop: List[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        **kwargs
    ) -> LLMResponse:
        """Single-turn prompt → always routed through Chat Completions.

        The configured endpoint is expected to implement Chat Completions.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
        return await self._chat_completions(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
        )

    async def _chat_completions(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> LLMResponse:
        """The single implementation of the OpenAI Chat Completions path.

        All supported endpoints use the OpenAI-compatible Chat Completions API.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        api_key, base_url = self._resolve_endpoint()
        # complete()/complete_messages() already own the retry loop. Disable the
        # OpenAI SDK's hidden retries so one configured attempt maps to one HTTP
        # request and the configured timeout is actually observable.
        loop = asyncio.get_running_loop()
        base_api_key = (api_key or "").split("?", 1)[0]
        client_key = (base_url or "", base_api_key, self._llm_config.timeout)
        clients = self._clients.setdefault(loop, {})
        client = clients.get(client_key)
        if client is None:
            client_options: Dict[str, Any] = {
                "api_key": base_api_key,
                "base_url": base_url,
                "timeout": self._llm_config.timeout,
                "max_retries": 0,
            }
            transport_factory = getattr(self.config, "http_transport_factory", None)
            if callable(transport_factory):
                import httpx

                client_options["http_client"] = httpx.AsyncClient(
                    transport=transport_factory()
                )
            client = AsyncOpenAI(
                **client_options,
            )
            clients[client_key] = client

        # Ensure a system message exists (required by some internal platforms)
        if not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": ""}] + list(messages)

        kwargs: Dict[str, Any] = {
            "model": self._llm_config.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # Resolving to None → **the parameter is not sent at all** (gateways like Opus 5 accept only 1 or nothing)
        _temp = self._resolve_temperature(temperature)
        if _temp is not None:
            kwargs["temperature"] = _temp
        if stop:
            kwargs["stop"] = stop
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        request_headers = dict(self._llm_config.extra_headers or {})
        if api_key:
            request_headers["Authorization"] = f"Bearer {api_key}"
        if request_headers:
            kwargs["extra_headers"] = request_headers
        if self._llm_config.extra_body:
            kwargs["extra_body"] = self._llm_config.extra_body

        try:
            async with llm_slot():
                response = await client.chat.completions.create(**kwargs)
        except Exception:
            # A transport failure may poison a pooled connection.  Evict only
            # this endpoint so the outer retry creates a clean client.
            clients.pop(client_key, None)
            try:
                closed = client.close()
                if asyncio.iscoroutine(closed):
                    await closed
            except Exception:
                pass
            raise

        msg, usage, _finish, _choices_count = self._extract_response(response)
        _reasoning_content, _reasoning_tokens = self._extract_reasoning(msg, usage)

        return LLMResponse(
            content=msg.content or "",
            tokens_used=usage.total_tokens if usage else 0,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=getattr(response, "model", None) or self._llm_config.model,
            finish_reason=_finish,
            response_id=getattr(response, "id", "") or "",
            choices_count=_choices_count,
            reasoning_content=_reasoning_content,
            reasoning_tokens=_reasoning_tokens,
            tool_calls=_serialize_tool_calls(getattr(msg, "tool_calls", None)),
        )

    async def close(self) -> None:
        clients = [
            client
            for per_loop in list(self._clients.values())
            for client in list(per_loop.values())
        ]
        self._clients.clear()
        for client in clients:
            try:
                closed = client.close()
                if asyncio.iscoroutine(closed):
                    await closed
            except Exception:
                pass

    async def complete_messages(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 500,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> LLMResponse:
        """
        Multi-turn messages call (supports sending tool results back).

        Args:
            messages: the complete messages list, which may contain:
                - {"role": "system", "content": "..."}
                - {"role": "user", "content": "..."}
                - {"role": "assistant", "content": "...", "tool_calls": [...]}
                - {"role": "tool", "tool_call_id": "...", "content": "..."}
            max_tokens, temperature, tools, tool_choice: same as complete()

        Returns:
            LLMResponse
        """
        _last_exc: Optional[BaseException] = None
        _attempts: List[Dict[str, Any]] = []
        for attempt in range(self._llm_config.max_retries):
            try:
                if attempt == 0:
                    logger.debug(
                        f"[llm] model={self._llm_config.model} "
                        f"backend={self._llm_config.backend.value} "
                        f"base_url={self._llm_config.base_url or 'default'} "
                        f"msgs={len(messages)} tools={len(tools) if tools else 0}"
                    )

                response = await self._chat_completions(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tools=tools,
                    tool_choice=tool_choice,
                )

                self._total_calls += 1
                self._total_tokens += response.tokens_used

                logger.debug(
                    f"[llm] response: {len(response.content)} chars, "
                    f"tokens={response.tokens_used}, "
                    f"tool_calls={len(response.tool_calls) if response.tool_calls else 0}"
                )

                return response

            except Exception as e:
                self._errors += 1
                _last_exc = e
                _attempts.append(_attempt_record(attempt + 1, e))
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self._llm_config.max_retries}): "
                    f"{type(e).__name__}: {e}"
                )
                if not _is_retryable_llm_error(e):
                    break
                if attempt < self._llm_config.max_retries - 1:
                    delay = self._llm_config.retry_delay * (attempt + 1)
                    await asyncio.sleep(delay)

        raise _build_retry_error(
            self._llm_config.max_retries, self._llm_config.model,
            self._llm_config.backend.value, _attempts, _last_exc,
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Multi-turn conversation
        
        Args:
            messages: message list [{"role": "user/assistant", "content": "..."}]
            max_tokens: maximum number of tokens
            temperature: temperature parameter
        
        Returns:
            LLM response
        """
        # Convert the multi-turn messages into a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        
        prompt = "\n".join(prompt_parts)
        
        return await self.complete(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics"""
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "errors": self._errors,
            "avg_tokens_per_call": (
                self._total_tokens / self._total_calls
                if self._total_calls > 0 else 0
            ),
            "error_rate": (
                self._errors / self._total_calls
                if self._total_calls > 0 else 0
            ),
        }
