"""Lease-bound leaf transport for the pinned HY Memory SDK, not context isolation.

The factory must hold an exclusive client lock on HY's own loop throughout this
async context, bind EVERY captured native provider/service before use, and keep
unsupported lazy-provider paths disabled. Store provenance, prompts, extraction,
reconcile and publication remain the caller's responsibility. No global SDK or
logging state is changed here. Saved pre-installation native methods are outside
this boundary; never expose an unbound client to permission-scoped work.

An embedding queue must be empty/idle at installation. Its cache and queue are
lease-local while bound. Exit cancels and awaits outstanding bound calls/timers,
clears pending futures, closes HTTP resources, then restores instance attributes.
Local encoder threads cannot be cancelled; exit waits for their completion.
The caller must stop creating child work before exit. Old bound callables remain
denied after exit; restored native methods require a NEW context before reuse.

Recheck happens at the physical transport boundary with no intervening await
before delegation. It cannot recall an in-flight send or make revocation atomic
with remote receipt. Responses and normal context exit are also rechecked.
After the last SDK/profile read, main may ``await transport.seal()`` before its
own governance projection. Seal requires idle children/queues, fully rechecks
the lease, and permanently closes request admission. Sealed exit does NOT
recheck context versions that main may just have updated; main must independently
recheck its frozen permission receipt before committing/publishing governance.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import re
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from http.cookiejar import CookieJar, DefaultCookiePolicy
from importlib import import_module, metadata
from time import monotonic
from types import MethodType, TracebackType
from typing import Any, Literal, Never, Protocol, Self, cast
from urllib.parse import urlsplit, urlunsplit

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, Omit

from one_memory.diagnostic_log import diagnostic_event, diagnostic_exception, elapsed_ms
from one_memory.hy_runtime_config import HY_MEMORY_DISTRIBUTION, HY_MEMORY_VERSION
from one_memory.hy_step_timing import HyStepTimer
from one_memory.model_profiles import (
    ModelProfileSnapshot,
    ProviderKind,
    attach_disable_thinking,
    validate_model_base_url,
)
from one_memory.model_request_budget import ModelRequestBudgetError

_MISSING = object()
_OWNER = "_one_memory_permission_transport"
_Role = Literal["generation", "embedding"]
_FailureKind = Literal["timeout", "connection", "transport", "invalid_response", "unknown"]
_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "connection",
        "content-length",
        "content-type",
        "user-agent",
        "host",
        "authorization",
        "x-stainless-async",
        "x-stainless-arch",
        "x-stainless-lang",
        "x-stainless-os",
        "x-stainless-package-version",
        "x-stainless-runtime",
        "x-stainless-runtime-version",
        "x-stainless-retry-count",
        "x-stainless-read-timeout",
        "x-stainless-raw-response",
    }
)


@dataclass(frozen=True)
class _Recipient:
    provider_kind: ProviderKind
    endpoint: str | None
    model: str
    dimensions: int | None = None


def _canonical_endpoint(provider: ProviderKind, value: str | None) -> str:
    if not value or any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError("missing_endpoint")
    validate_model_base_url(value)
    parsed = urlsplit(value)
    if parsed.query or parsed.fragment or "?" in value or "#" in value:
        raise ValueError("ambiguous_endpoint")
    hostname = (parsed.hostname or "").encode("idna").decode().lower()
    if not hostname or "%" in hostname or "\\" in value:
        raise ValueError("ambiguous_endpoint")
    host = f"[{hostname}]" if ":" in hostname else hostname
    port = parsed.port
    if port is not None and port != (443 if parsed.scheme == "https" else 80):
        host = f"{host}:{port}"
    path = parsed.path.rstrip("/")
    if provider is ProviderKind.OLLAMA and not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parsed.scheme.lower(), host, path, "", ""))


def _recipients(profile: ModelProfileSnapshot) -> tuple[_Recipient, _Recipient]:
    if profile.provider_kind not in {ProviderKind.OPENAI_COMPATIBLE, ProviderKind.OLLAMA}:
        raise HyPermissionTransportError("unsupported_backend")
    embedding_provider = profile.embedding_provider_kind or profile.provider_kind
    if not profile.embedding_model or embedding_provider not in {
        ProviderKind.OPENAI_COMPATIBLE,
        ProviderKind.OLLAMA,
        ProviderKind.LOCAL_ONNX,
    }:
        raise HyPermissionTransportError("unsupported_backend")
    try:
        generation = _Recipient(
            profile.provider_kind,
            _canonical_endpoint(profile.provider_kind, profile.base_url),
            profile.model,
        )
        embedding = _Recipient(
            embedding_provider,
            (
                None
                if embedding_provider is ProviderKind.LOCAL_ONNX
                else _canonical_endpoint(
                    embedding_provider,
                    profile.embedding_base_url or profile.base_url,
                )
            ),
            profile.embedding_model,
            profile.embedding_dimensions,
        )
    except (UnicodeError, ValueError):
        raise HyPermissionTransportError("endpoint_invalid") from None
    return generation, embedding


class PermissionLease(Protocol):
    @property
    def user_id(self) -> str: ...

    @property
    def profile(self) -> ModelProfileSnapshot: ...

    def recheck(self) -> None: ...


class HyPermissionTransportError(RuntimeError):
    """Stable code only: never retains a request, credential or upstream error."""

    def __init__(self, reason: str) -> None:
        self.code = f"model_content_transport_{reason}"
        super().__init__(self.code)


class _RequestFailure(HyPermissionTransportError):
    def __init__(
        self,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        failure_kind: _FailureKind = "unknown",
    ) -> None:
        super().__init__("request_failed")
        self.retryable = retryable
        self.status_code = status_code
        self.failure_kind = failure_kind


def _failure_kind(error: Exception) -> _FailureKind:
    if isinstance(error, (APITimeoutError, httpx.TimeoutException, TimeoutError)):
        return "timeout"
    if isinstance(error, (APIConnectionError, httpx.ConnectError, ConnectionError)):
        return "connection"
    if isinstance(error, httpx.TransportError):
        return "transport"
    return "unknown"


@dataclass(frozen=True)
class _Spec:
    role: _Role
    endpoint: str
    model: str
    dimensions: int | None
    api_key: str = field(repr=False)
    retries: int
    delay: float
    timeout: float
    extra_body: str


@dataclass
class _Remote:
    instance: Any = field(repr=False)
    spec: _Spec
    client: AsyncOpenAI = field(repr=False)


@dataclass(frozen=True)
class _Envelope:
    remote: _Remote = field(repr=False)
    body: str = field(repr=False)


class _PhysicalTransport(httpx.AsyncBaseTransport):
    def __init__(self, owner: HyPermissionTransport, inner: httpx.AsyncBaseTransport) -> None:
        self.owner = owner
        self.inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # After all httpx request hooks, immediately before the physical send.
        self.owner.assert_allowed()
        if not isinstance(request.stream, httpx.AsyncByteStream):
            self.owner._deny("unsupported_stream")
        content = b"".join([part async for part in request.stream])
        request.stream = httpx.ByteStream(content)
        self.owner._check_request(request, content=content)
        self.owner._physical_requests += 1
        try:
            try:
                response = await self.inner.handle_async_request(request)
            except ModelRequestBudgetError:
                self.owner._deny("budget_exceeded")
            if 300 <= response.status_code < 400:
                await response.aclose()
                self.owner._deny("redirect")
            return response
        finally:
            self.owner._physical_requests -= 1

    async def aclose(self) -> None:
        await self.inner.aclose()


class HyPermissionTransport:
    """Single-use context; bind methods return the same native instance.

    ``http_transport`` is an optional trusted test transport, owned and closed by
    this context. Production uses a direct httpx transport with no environment
    proxy/auth configuration, HTTP redirects or connection retries.
    """

    def __init__(
        self,
        lease: PermissionLease,
        *,
        http_transport: httpx.AsyncBaseTransport | None = None,
        step_timer: HyStepTimer | None = None,
    ) -> None:
        self.lease = lease
        self._step_timer = step_timer
        self._profile = lease.profile.model_copy(deep=True)
        self._transport = http_transport
        self._active = False
        self._ended = False
        self._requests_sealed = False
        self._physical_requests = 0
        self._denial: str | None = None
        self._patches: list[tuple[Any, str, Any, Any]] = []
        self._tasks: dict[asyncio.Task[Any], int] = {}
        self._local_work: set[asyncio.Task[Any]] = set()
        self._services: list[Any] = []
        self._clients: list[AsyncOpenAI] = []
        self._http_client: httpx.AsyncClient
        self._envelope: ContextVar[_Envelope | None] = ContextVar("hy_leaf_envelope", default=None)

    @property
    def denied_code(self) -> str | None:
        return f"model_content_transport_{self._denial}" if self._denial else None

    @property
    def requests_sealed(self) -> bool:
        return self._requests_sealed

    async def seal(self) -> None:
        """Fully check an idle lease, then permanently refuse SDK/request admission.

        Call on HY's loop after ALL SDK work, including profile readback. Active
        children/queues cause ``active_requests`` without sealing or latching a
        permission denial: await those children and try again. No work is dropped
        or implicitly approved. There is no await between the idle check, full
        lease check and state change. A successful seal is idempotent.

        Main owns the subsequent frozen-permission check and governance commit.
        This method deliberately does not call an optional recheck_permission.
        """
        if self._ended or not self._active:
            raise HyPermissionTransportError("lease_ended")
        if self._denial is not None:
            raise HyPermissionTransportError(self._denial)
        if self._requests_sealed:
            return
        if (
            self._tasks
            or self._physical_requests
            or self._local_work
            or any(_embedding_queue_active(service) for service in self._services)
        ):
            raise HyPermissionTransportError("active_requests")
        self.assert_allowed()
        self._requests_sealed = True

    async def __aenter__(self) -> Self:
        if self._active or self._ended:
            raise HyPermissionTransportError("already_used")
        self._active = True
        try:
            self._verify_sdk()
            self.assert_allowed()
            inner = self._transport or httpx.AsyncHTTPTransport(retries=0, trust_env=False)
            self._http_client = httpx.AsyncClient(
                transport=_PhysicalTransport(self, inner),
                trust_env=False,
                follow_redirects=False,
                # A gateway may set cookies that alter the next call in the
                # same task. Model requests never rely on cookies, so never
                # store any.
                # Pass the jar itself: httpx copies a Cookies object into a
                # fresh jar with the default policy, which would store again.
                cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])),
                event_hooks={"request": [self._request_hook]},
            )
        except BaseException:
            self._active = False
            self._ended = True
            if self._transport is not None:
                await self._transport.aclose()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        failure: HyPermissionTransportError | None = None
        if exc_type is None and not self._requests_sealed:
            try:
                self.assert_allowed()
            except HyPermissionTransportError as error:
                failure = error
        self._active = False
        self._ended = True
        cleanup = asyncio.create_task(self._cleanup(asyncio.current_task()))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise
        if exc_type is None:
            if failure is not None:
                raise failure
            if self._denial is not None:
                raise HyPermissionTransportError(self._denial)

    def assert_allowed(self) -> None:
        """Recheck publication/child boundaries; a denial is terminal for this context."""
        self._assert_active()
        try:
            self.lease.recheck()
        except Exception as error:
            reason = str(error).removeprefix("model_content_")
            self._deny(
                f"lease_denied_{reason}"
                if re.fullmatch(r"[a-z_]{1,60}", reason)
                else "lease_denied"
            )
        if self.lease.profile != self._profile:
            self._deny("lease_changed")

    def _assert_active(self) -> None:
        if self._ended or not self._active:
            raise HyPermissionTransportError("lease_ended")
        if self._denial is not None:
            raise HyPermissionTransportError(self._denial)
        if self._requests_sealed:
            raise HyPermissionTransportError("requests_sealed")

    def _deny(self, reason: str) -> Never:
        if self._denial is None:
            self._denial = reason
        raise HyPermissionTransportError(self._denial) from None

    @staticmethod
    def _verify_sdk() -> None:
        if metadata.version(HY_MEMORY_DISTRIBUTION) != HY_MEMORY_VERSION:
            raise HyPermissionTransportError("sdk_version")
        llm = import_module("hy_memory.agent.llm_provider").LLMProvider
        embed = import_module("hy_memory.core.embed_service").EmbedService
        prompt = (
            "self",
            "prompt",
            "max_tokens",
            "temperature",
            "stop",
            "tools",
            "tool_choice",
            "kwargs",
        )
        messages = (
            "self",
            "messages",
            "max_tokens",
            "temperature",
            "tools",
            "tool_choice",
        )
        chat = (
            "self",
            "messages",
            "max_tokens",
            "temperature",
            "stop",
            "tools",
            "tool_choice",
        )
        for cls, name, expected in (
            (llm, "_call_llm", prompt),
            (llm, "complete", prompt),
            (llm, "complete_messages", messages),
            (llm, "_chat_completions", chat),
            (embed, "_embed_openai", ("self", "texts")),
        ):
            method = getattr(cls, name, None)
            if (
                not inspect.iscoroutinefunction(method)
                or tuple(inspect.signature(method).parameters) != expected
            ):
                raise HyPermissionTransportError("sdk_shape")

    @asynccontextmanager
    async def _operation(self) -> AsyncIterator[None]:
        self._assert_active()
        task = asyncio.current_task()
        assert task is not None
        self._tasks[task] = self._tasks.get(task, 0) + 1
        try:
            yield
            self.assert_allowed()
        finally:
            count = self._tasks[task] - 1
            if count:
                self._tasks[task] = count
            else:
                del self._tasks[task]

    def _patch(self, instance: Any, name: str, value: Any) -> None:
        previous = vars(instance).get(name, _MISSING)
        self._patches.append((instance, name, previous, value))
        setattr(instance, name, value)

    def _bind_method(
        self,
        instance: Any,
        name: str,
        call: Callable[..., Awaitable[Any]],
        check: Callable[[], None],
        *,
        drain_local: bool = False,
        step: str | None = None,
    ) -> None:
        original = getattr(instance, name, call)
        timer = self._step_timer
        signature = inspect.signature(original)

        @wraps(getattr(original, "__func__", original))
        async def bound(_instance: Any, *args: Any, **kwargs: Any) -> Any:
            async with self._operation():
                check()
                try:
                    signature.bind(*args, **kwargs)
                    with (
                        timer.step(step, items=_step_items(args))
                        if timer is not None and step is not None
                        else nullcontext()
                    ):
                        if drain_local:
                            return await self._drain_local(call(*args, **kwargs))
                        return await call(*args, **kwargs)
                except HyPermissionTransportError:
                    raise
                except TypeError:
                    self._deny("unsupported_operation")
                except Exception as error:
                    # Only these two replacement methods create this native
                    # error, with sanitized attempts. Other native/local errors
                    # must not propagate provider details to HY's queue logger.
                    retry_type = import_module("hy_memory.agent.llm_provider").LLMRetryError
                    if name in {"complete", "complete_messages"} and type(error) is retry_type:
                        raise
                    raise _RequestFailure() from None

        self._patch(instance, name, MethodType(bound, instance))

    async def _drain_local(self, call: Awaitable[Any]) -> Any:
        async def run() -> Any:
            return await call

        task = asyncio.create_task(run())
        self._local_work.add(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            self._local_work.discard(task)

    def _claim(self, instance: Any, expected_type: type[Any], names: tuple[str, ...]) -> bool:
        self._assert_active()
        if type(instance) is not expected_type:
            self._deny("unsupported_instance")
        owner = vars(instance).get(_OWNER)
        if owner is self:
            return False
        if owner is not None:
            self._deny("instance_already_bound")
        for name in names:
            if getattr(getattr(instance, name, None), "__func__", None) is not getattr(
                expected_type, name, None
            ):
                self._deny("unsupported_instance_method")
        return True

    def _spec(self, instance: Any, role: _Role) -> _Spec:
        generation, embedding = _recipients(self._profile)
        recipient = generation if role == "generation" else embedding
        cfg = instance._llm_config if role == "generation" else instance._embed_config
        if recipient.provider_kind not in {ProviderKind.OPENAI_COMPATIBLE, ProviderKind.OLLAMA}:
            self._deny("unsupported_backend")
        if role == "generation":
            if (
                getattr(cfg.backend, "value", None) != "openai"
                or instance.config.llm.provider != "openai"
            ):
                self._deny("unsupported_backend")
        elif cfg.provider != "openai":
            self._deny("unsupported_backend")
        try:
            endpoint = _canonical_endpoint(ProviderKind.OPENAI_COMPATIBLE, cfg.base_url)
            path = httpx.URL(endpoint).path
            if (
                "%" in endpoint
                or any(part in {".", ".."} for part in cfg.base_url.split("/"))
                or "//" in path
            ):
                self._deny("endpoint_invalid")
        except (ValueError, RuntimeError):
            self._deny("endpoint_invalid")
        if endpoint != recipient.endpoint:
            self._deny("endpoint_mismatch")
        if cfg.model != recipient.model:
            self._deny("model_mismatch")
        dimensions = (cfg.embedding_dims or None) if role == "embedding" else None
        if dimensions != recipient.dimensions or (
            dimensions is not None and type(dimensions) is not int
        ):
            self._deny("dimensions_mismatch")
        extras = attach_disable_thinking(self._profile, {}) if role == "generation" else {}
        if cfg.extra_headers or (cfg.extra_body or {}) != extras:
            self._deny("unexpected_overrides")
        if (
            not isinstance(cfg.api_key, str)
            or not cfg.api_key
            or any(c.isspace() for c in cfg.api_key)
        ):
            self._deny("credentials_invalid")
        if type(cfg.max_retries) is not int or cfg.max_retries < (1 if role == "generation" else 0):
            self._deny("retry_config_invalid")
        if cfg.retry_delay is None or cfg.retry_delay < 0:
            self._deny("retry_config_invalid")
        return _Spec(
            role,
            endpoint,
            cfg.model,
            dimensions,
            cfg.api_key,
            cfg.max_retries,
            float(cfg.retry_delay),
            float(cfg.timeout or 3),
            _json(extras),
        )

    def _remote(self, instance: Any, role: _Role) -> _Remote:
        spec = self._spec(instance, role)
        client = AsyncOpenAI(
            api_key=spec.api_key,
            base_url=spec.endpoint,
            max_retries=0,
            timeout=spec.timeout,
            organization="",
            project="",
            http_client=self._http_client,
            default_headers=cast(Any, {"OpenAI-Organization": Omit(), "OpenAI-Project": Omit()}),
        )
        self._clients.append(client)
        if getattr(client, "_provider_runtime", None) is not None:
            self._deny("unsupported_backend")
        return _Remote(instance, spec, client)

    def _check_remote(self, remote: _Remote) -> None:
        self._assert_active()
        if self._spec(remote.instance, remote.spec.role) != remote.spec:
            self._deny("configuration_changed")

    async def _request_hook(self, request: httpx.Request) -> None:
        self._check_request(request)

    def _check_request(self, request: httpx.Request, *, content: bytes | None = None) -> None:
        self._assert_active()
        envelope = self._envelope.get()
        if envelope is None:
            self._deny("unsupported_request")
        remote = envelope.remote
        self._check_remote(remote)
        suffix = "/chat/completions" if remote.spec.role == "generation" else "/embeddings"
        url = httpx.URL(remote.spec.endpoint + suffix)
        if request.method != "POST":
            self._deny("method_mismatch")
        if request.url != url or request.url.raw_path != url.raw_path:
            self._deny("endpoint_mismatch")
        if (
            set(request.headers) - _HEADERS
            or request.headers.get_list("authorization") != [f"Bearer {remote.spec.api_key}"]
            or request.headers.get_list("host") != [url.netloc.decode("ascii")]
            or request.headers.get("content-type") != "application/json"
        ):
            self._deny("headers_mismatch")
        try:
            content = request.content if content is None else content
            if request.headers.get_list("content-length") != [str(len(content))]:
                self._deny("headers_mismatch")
            body = json.loads(content)
            matches = _json(body) == envelope.body
        except (ValueError, TypeError, httpx.RequestNotRead):
            self._deny("body_invalid")
        if not matches or body.get("model") != remote.spec.model:
            self._deny("body_mismatch")
        if remote.spec.role == "embedding" and body.get("dimensions") != remote.spec.dimensions:
            self._deny("dimensions_mismatch")

    async def _exchange(self, remote: _Remote, payload: dict[str, Any]) -> tuple[Any, int]:
        started = monotonic()
        phase = "embedding" if remote.spec.role == "embedding" else "generation"
        self._check_remote(remote)
        diagnostic_event(
            level="info",
            component="hy.transport",
            event="hy.transport_call.started",
            outcome="started",
            details={"phase": phase, "call_count": 1},
        )
        # Freeze the complete native payload, not just model/destination fields.
        body = _json(payload)
        token = self._envelope.set(_Envelope(remote, body))
        try:
            if remote.spec.role == "generation":
                extras = json.loads(remote.spec.extra_body)
                kwargs = {key: value for key, value in payload.items() if key not in extras}
                result = await remote.client.chat.completions.create(
                    **kwargs, extra_body=extras or None
                )
                status = 200
            else:
                raw = await remote.client.embeddings.with_raw_response.create(**payload)
                result, status = raw.parse(), raw.status_code
            self.assert_allowed()
            diagnostic_event(
                level="info",
                component="hy.transport",
                event="hy.transport_call.completed",
                outcome="succeeded",
                duration_ms=elapsed_ms(started),
                details={"phase": phase, "call_count": 1, "status": str(status)},
            )
            return result, status
        except Exception as error:
            if self._denial is not None:
                raise HyPermissionTransportError(self._denial) from None
            status_code = getattr(error, "status_code", None)
            failure_kind = _failure_kind(error)
            diagnostic_event(
                level="error",
                component="hy.transport",
                event="hy.transport_call.failed",
                outcome="failed",
                duration_ms=elapsed_ms(started),
                error_code=f"hy_transport_{failure_kind}",
                **diagnostic_exception(error),
                details={
                    "phase": phase,
                    "call_count": 1,
                    "status": str(status_code) if type(status_code) is int else "unknown",
                },
            )
            raise _RequestFailure(
                retryable=isinstance(
                    error, (APIConnectionError, httpx.TransportError, TimeoutError, ConnectionError)
                ),
                status_code=status_code if type(status_code) is int else None,
                failure_kind=failure_kind,
            ) from None
        finally:
            self._envelope.reset(token)

    def bind_provider(self, provider: Any) -> Any:
        """Bind a native LLMProvider, including _call_llm and both complete paths."""
        module = import_module("hy_memory.agent.llm_provider")
        names = ("complete", "complete_messages", "_call_llm", "_chat_completions")
        if not self._claim(provider, module.LLMProvider, names):
            return provider
        remote = self._remote(provider, "generation")
        self._patch(provider, _OWNER, self)

        def check() -> None:
            self._check_remote(remote)

        async def leaf_messages(
            messages: list[dict[str, Any]],
            max_tokens: int,
            temperature: float | None,
            tools: Any = None,
            tool_choice: Any = None,
            *,
            stop: Any = None,
        ) -> Any:
            if not any(message.get("role") == "system" for message in messages):
                messages = [{"role": "system", "content": ""}, *messages]
            payload: dict[str, Any] = {
                "model": remote.spec.model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                payload["temperature"] = temperature
            if stop:
                payload["stop"] = stop
            if tools:
                payload["tools"] = tools
                if tool_choice is not None:
                    payload["tool_choice"] = tool_choice
            payload.update(json.loads(remote.spec.extra_body))
            response, _ = await self._exchange(remote, payload)
            try:
                choice = response.choices[0]
                message, usage = choice.message, response.usage
                calls = [
                    {
                        "id": tc.id or "",
                        "type": tc.type or "function",
                        "function": {
                            "name": getattr(tc.function, "name", ""),
                            "arguments": getattr(tc.function, "arguments", ""),
                        },
                    }
                    for tc in (message.tool_calls or [])
                ]
                return module.LLMResponse(
                    content=message.content or "",
                    model=remote.spec.model,
                    tokens_used=_token_count(usage.total_tokens) if usage else 0,
                    prompt_tokens=_token_count(usage.prompt_tokens) if usage else 0,
                    completion_tokens=_token_count(usage.completion_tokens) if usage else 0,
                    finish_reason=choice.finish_reason,
                    tool_calls=calls or None,
                )
            except Exception:
                raise _RequestFailure(failure_kind="invalid_response") from None

        async def leaf(
            prompt: str,
            max_tokens: int,
            temperature: float | None,
            stop: Any = None,
            tools: Any = None,
            tool_choice: Any = None,
        ) -> Any:
            return await leaf_messages(
                [{"role": "user", "content": prompt}],
                max_tokens,
                temperature,
                tools,
                tool_choice,
                stop=stop,
            )

        async def dispatch(
            prompt: str,
            max_tokens: int,
            temperature: float | None,
            stop: Any = None,
            tools: Any = None,
            tool_choice: Any = None,
            **kwargs: Any,
        ) -> Any:
            if kwargs:
                self._deny("unexpected_overrides")
            return await leaf(prompt, max_tokens, temperature, stop, tools, tool_choice)

        async def retry(call: Callable[[], Awaitable[Any]]) -> Any:
            # HY's total-attempt budget/counters/backoff, without native prompt
            # logging or raw upstream exceptions in its attempt receipts.
            attempts = []
            for attempt in range(remote.spec.retries):
                check()
                try:
                    result = await call()
                    self.assert_allowed()
                    provider._total_calls += 1
                    provider._total_tokens += result.tokens_used
                    return result
                except HyPermissionTransportError as error:
                    provider._errors += 1
                    if self._denial is not None or not isinstance(error, _RequestFailure):
                        raise
                    record: dict[str, Any] = {
                        "attempt": attempt + 1,
                        "error_type": "TransportError",
                        "error": error.code,
                        "failure_kind": error.failure_kind,
                    }
                    if error.status_code is not None:
                        record["status_code"] = error.status_code
                    attempts.append(record)
                if attempt + 1 < remote.spec.retries:
                    await asyncio.sleep(remote.spec.delay * (attempt + 1))
            raise module.LLMRetryError("hy_memory_model_request_failed", attempts) from None

        async def complete(
            prompt: str,
            max_tokens: int = 500,
            temperature: float | None = None,
            stop: Any = None,
            tools: Any = None,
            tool_choice: Any = None,
            **kwargs: Any,
        ) -> Any:
            if kwargs:
                self._deny("unexpected_overrides")
            return await retry(
                lambda: provider._call_llm(
                    prompt, max_tokens, temperature, stop, tools, tool_choice
                )
            )

        async def complete_messages(
            messages: list[dict[str, Any]],
            max_tokens: int = 500,
            temperature: float | None = None,
            tools: Any = None,
            tool_choice: Any = None,
        ) -> Any:
            return await retry(
                lambda: leaf_messages(messages, max_tokens, temperature, tools, tool_choice)
            )

        for name, call in (
            ("complete", complete),
            ("complete_messages", complete_messages),
            ("_call_llm", dispatch),
            ("_chat_completions", leaf_messages),
        ):
            self._bind_method(provider, name, call, check, step="model")
        return provider

    async def _unsupported_backend(self, *args: Any, **kwargs: Any) -> Any:
        self._deny("unsupported_backend")

    def bind_embed_service(self, service: Any) -> Any:
        """Bind native remote EmbedService; local ONNX uses the separate method."""
        cls = import_module("hy_memory.core.embed_service").EmbedService
        names = (
            "embed",
            "embed_batch",
            "embed_queued",
            "embed_queued_with_meta",
            "embed_with_meta",
            "embed_with_cache_info",
            "_embed_texts",
            "_embed_texts_with_meta",
            "_embed_openai",
            "_delayed_flush",
            "_send_batch",
        )
        if not self._claim(service, cls, names):
            return service
        if _embedding_queue_active(service):
            self._deny("foreign_embedding_queue")
        remote = self._remote(service, "embedding")
        self._patch(service, _OWNER, self)
        isolated_state: tuple[tuple[str, Any], ...] = (
            ("_cache", OrderedDict()),
            ("_last_embed_meta", {}),
            ("_queue_locks", {}),
            ("_queue_pending", {}),
            ("_queue_flush_tasks", {}),
            ("_client_cache", {}),
        )
        for name, value in isolated_state:
            self._patch(service, name, value)
        self._services.append(service)

        def check() -> None:
            self._check_remote(remote)

        async def leaf(texts: list[str]) -> tuple[list[list[float]], dict[str, Any]]:
            payload: dict[str, Any] = {
                "model": remote.spec.model,
                "input": texts,
                "encoding_format": "float",
            }
            if remote.spec.dimensions is not None:
                payload["dimensions"] = remote.spec.dimensions
            for attempt in range(remote.spec.retries + 1):
                check()
                try:
                    result, status = await self._exchange(remote, payload)
                    vectors = [item.embedding for item in result.data]
                    if len(vectors) != len(texts) or any(
                        not isinstance(vector, list)
                        or (
                            remote.spec.dimensions is not None
                            and len(vector) != remote.spec.dimensions
                        )
                        or any(
                            type(value) not in {int, float} or not math.isfinite(value)
                            for value in vector
                        )
                        for vector in vectors
                    ):
                        raise _RequestFailure()
                    service._last_embed_meta = {
                        "http_status": status,
                        "model": remote.spec.model,
                        "attempts": attempt + 1,
                        "usage": {
                            "prompt_tokens": _token_count(result.usage.prompt_tokens),
                            "total_tokens": _token_count(result.usage.total_tokens),
                        }
                        if result.usage
                        else None,
                    }
                    return vectors, dict(service._last_embed_meta)
                except _RequestFailure as error:
                    service._last_embed_meta = {
                        "attempts": attempt + 1,
                        "retryable": error.retryable,
                    }
                    if not error.retryable or attempt == remote.spec.retries:
                        raise
                    await asyncio.sleep(remote.spec.delay)
            raise _RequestFailure()  # Unreachable with validated nonnegative retry count.

        for name in names:
            call = leaf if name == "_embed_openai" else getattr(service, name)
            self._bind_method(service, name, call, check, step="embedding")
        return service

    def bind_local_embed_service(self, service: Any) -> Any:
        """Keep the product's LocalHyEmbedService local; never substitute a cloud leaf."""
        from one_memory.embedding_assets import (
            BUILTIN_EMBEDDING_DIMENSIONS,
            BUILTIN_EMBEDDING_MODEL,
        )
        from one_memory.hy_memory_bridge import LocalHyEmbedService

        names = (
            "embed",
            "embed_batch",
            "embed_batch_with_meta",
            "embed_queued",
            "embed_queued_with_meta",
            "embed_with_meta",
        )
        if not self._claim(service, LocalHyEmbedService, names):
            return service
        _, recipient = _recipients(self._profile)
        if (
            recipient.provider_kind != ProviderKind.LOCAL_ONNX
            or recipient.endpoint is not None
            or recipient.model != BUILTIN_EMBEDDING_MODEL
            or recipient.dimensions != BUILTIN_EMBEDDING_DIMENSIONS
        ):
            self._deny("local_embedding_mismatch")
        encoder = service._embedder

        def check() -> None:
            self.assert_allowed()
            if (
                service._embedder is not encoder
                or service.get_embedding_dims() != recipient.dimensions
            ):
                self._deny("local_embedding_changed")

        self._patch(service, _OWNER, self)
        for name in names:
            self._bind_method(
                service, name, getattr(service, name), check, drain_local=True, step="embedding"
            )
        self._bind_method(service, "_embed_openai", self._unsupported_backend, check)
        return service

    async def _cleanup(self, exiting: asyncio.Task[Any] | None) -> None:
        tasks = {
            task
            for task in self._tasks
            if task is not exiting and task not in self._local_work and not task.done()
        }
        # Native HY schedules the timer before its bound coroutine can register
        # itself. Include done timers too, to retrieve an admission-time failure.
        for service in self._services:
            tasks.update(task for task in _embedding_flush_tasks(service) if task is not exiting)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._local_work:
            await asyncio.gather(*self._local_work, return_exceptions=True)
        for service in self._services:
            for future in _embedding_pending_futures(service):
                if not future.done():
                    future.cancel()
            service._queue_pending.clear()
            service._cache.clear()
        try:
            clients: list[AsyncOpenAI | httpx.AsyncClient] = [*self._clients, self._http_client]
            for client in clients:
                try:
                    if isinstance(client, httpx.AsyncClient):
                        await client.aclose()
                    else:
                        await client.close()
                except Exception:
                    self._denial = self._denial or "cleanup_failed"
        finally:
            for instance, name, previous, installed in reversed(self._patches):
                # Queue/meta attributes are intentionally mutated by native HY.
                dynamic = name in {
                    "_last_embed_meta",
                    "_queue_locks",
                    "_queue_pending",
                    "_queue_flush_tasks",
                    "_client_cache",
                }
                if not dynamic and vars(instance).get(name, _MISSING) is not installed:
                    self._denial = self._denial or "restoration_conflict"
                    continue
                if previous is _MISSING:
                    delattr(instance, name)
                else:
                    setattr(instance, name, previous)
            self._patches.clear()
            self._clients.clear()
            self._services.clear()


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _embedding_flush_tasks(service: Any) -> tuple[asyncio.Task[Any], ...]:
    tasks = getattr(service, "_queue_flush_tasks", {})
    if not isinstance(tasks, dict):
        return ()
    return tuple(task for task in tasks.values() if isinstance(task, asyncio.Task))


def _embedding_pending_futures(service: Any) -> tuple[asyncio.Future[Any], ...]:
    pending = getattr(service, "_queue_pending", {})
    batches = pending.values() if isinstance(pending, dict) else (pending,)
    futures: list[asyncio.Future[Any]] = []
    for batch in batches:
        if not isinstance(batch, list):
            continue
        for item in batch:
            if isinstance(item, tuple) and len(item) > 1 and isinstance(item[1], asyncio.Future):
                futures.append(item[1])
    return tuple(futures)


def _embedding_queue_active(service: Any) -> bool:
    pending = getattr(service, "_queue_pending", {})
    has_pending = (
        any(bool(batch) for batch in pending.values())
        if isinstance(pending, dict)
        else bool(pending)
    )
    return has_pending or any(not task.done() for task in _embedding_flush_tasks(service))


def _step_items(args: tuple[Any, ...]) -> int:
    first = args[0] if args else None
    return len(first) if isinstance(first, list) else 1


def _token_count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise _RequestFailure()
    return value
