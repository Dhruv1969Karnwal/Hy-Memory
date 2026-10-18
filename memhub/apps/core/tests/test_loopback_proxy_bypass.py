"""Loopback addresses must not go through HTTP(S)_PROXY.

Some environments set a system HTTP proxy; httpx defaults to trust_env=True and may send
Handing 127.0.0.1 / localhost / ::1 requests to a proxy shows up as "local Ollama will not
connect / health never comes up", and debugging gets dragged off course. Remote HTTPS still must.

The criterion is **transport layer ownership**, not "did the fake proxy receive a connection":
the latter is too long a chain, and any link that fails to start silently reads as "not hijacked",
so the positive control dies unnoticed. Here we ask the client which pool a URL would use.

This file does not go green by setting NO_PROXY -- that is an environment bypass, not a fix.
"""

from __future__ import annotations

import asyncio
import os

import httpx
import pytest

from one_memory.http_client import create_async_client, create_client

# A stand-in for a corporate proxy. It only decides ownership and never connects, so it need not
# live.
FAKE_PROXY = "http://127.0.0.1:9"

LOOPBACK_URLS = (
    "http://127.0.0.1:8791/api/v1/health",
    "http://localhost:11434/api/tags",
    "http://[::1]:11434/api/tags",
)
REMOTE_URLS = (
    "https://models.example.com/v1/chat/completions",
    "https://generation.example.net/v1/chat/completions",
)


def _origin(url: object) -> str:
    """Normalizes both httpcore.URL and httpx.URL to `scheme://host:port`."""
    scheme = getattr(url, "scheme", b"")
    host = getattr(url, "host", b"")
    port = getattr(url, "port", None)
    if isinstance(scheme, bytes):
        scheme = scheme.decode()
    if isinstance(host, bytes):
        host = host.decode()
    return f"{scheme}://{host}:{port}"


def proxy_for(client: httpx.Client | httpx.AsyncClient, url: str) -> str | None:
    """Returns the proxy httpx picked for that URL (`scheme://host:port`), None when direct.

    Depends on httpx 0.28.x internals (`_transport_for_url` and `pool._proxy_url`).
    If the structure changes, fail here instead of letting every assertion read as "direct".
    The pool's `_proxy_url` is an httpcore.URL whose fields are bytes, so it is not comparable.
    """
    transport = client._transport_for_url(httpx.URL(url))
    pool = getattr(transport, "_pool", None)
    assert pool is not None, (
        f"httpx transport {type(transport).__name__} has no _pool, the internals changed and this "
        f"file's criterion needs a "
        f"rewrite"
    )
    known_pool = type(pool).__name__ in {
        "ConnectionPool",
        "AsyncConnectionPool",
        "HTTPProxy",
        "AsyncHTTPProxy",
    }
    assert known_pool, (
        f"unknown connection pool type {type(pool).__name__}, cannot decide proxy ownership;"
        "the httpx version changed, please rewrite the criterion in this file"
    )
    proxy = getattr(pool, "_proxy_url", None)
    return None if proxy is None else _origin(proxy)


@pytest.fixture
def env_proxy(monkeypatch: pytest.MonkeyPatch) -> str:
    # Clear **every** *_proxy variable in the environment before setting the fake proxy.
    # Setting only the uppercase HTTP_PROXY is not enough: urllib's getproxies_environment()
    # collects names lowercased, so a lowercase http_proxy already on the machine (corporate
    # proxies often set it that way) overrides the fake proxy here, and the positive control
    # reads a real proxy address instead of FAKE_PROXY: red on such machines, green elsewhere.
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTP_PROXY", FAKE_PROXY)
    monkeypatch.setenv("HTTPS_PROXY", FAKE_PROXY)
    return FAKE_PROXY


def test_plain_httpx_sends_loopback_through_env_proxy(env_proxy: str) -> None:
    """Positive control: unpatched, loopback really is hijacked, which gives the negatives \
meaning."""
    expected = _origin(httpx.URL(env_proxy))
    with httpx.Client() as client:
        hijacked = {url: proxy_for(client, url) for url in LOOPBACK_URLS}
    assert all(proxy == expected for proxy in hijacked.values()), (
        f"the env proxy did not hijack loopback, the positive control is dead: {hijacked}"
    )


def test_create_client_keeps_loopback_direct_while_http_proxy_is_set(
    env_proxy: str,
) -> None:
    del env_proxy
    with create_client() as client:
        routed = {url: proxy_for(client, url) for url in LOOPBACK_URLS}
    assert all(proxy is None for proxy in routed.values()), (
        f"loopback still goes through a proxy: {routed}"
    )


def test_create_async_client_keeps_loopback_direct_while_http_proxy_is_set(
    env_proxy: str,
) -> None:
    del env_proxy

    async def routes() -> dict[str, str | None]:
        async with create_async_client() as client:
            return {url: proxy_for(client, url) for url in LOOPBACK_URLS}

    routed = asyncio.run(routes())
    assert all(proxy is None for proxy in routed.values()), (
        f"loopback still goes through a proxy: {routed}"
    )


def test_remote_requests_still_use_env_proxy(env_proxy: str) -> None:
    """Only loopback bypasses the proxy; remote HTTPS still uses it."""
    expected = _origin(httpx.URL(env_proxy))
    with create_client() as client:
        sync_routed = {url: proxy_for(client, url) for url in REMOTE_URLS}

    async def routes() -> dict[str, str | None]:
        async with create_async_client() as client:
            return {url: proxy_for(client, url) for url in REMOTE_URLS}

    async_routed = asyncio.run(routes())
    assert all(proxy == expected for proxy in sync_routed.values()), (
        f"remote HTTPS bypassed the proxy: {sync_routed}"
    )
    assert all(proxy == expected for proxy in async_routed.values()), (
        f"remote HTTPS bypassed the proxy: {async_routed}"
    )


def test_injected_transport_is_not_shadowed_by_the_mounts(env_proxy: str) -> None:
    """A caller's transport= must not be overwritten by the loopback mount, or MockTransport \
leaks."""
    del env_proxy
    mock = httpx.MockTransport(lambda _request: httpx.Response(200))
    with create_client(transport=mock) as client:
        for url in (*LOOPBACK_URLS, *REMOTE_URLS):
            assert client._transport_for_url(httpx.URL(url)) is mock


def test_removing_loopback_mounts_lets_the_proxy_take_loopback_back(
    env_proxy: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: removing the loopback direct mount must send loopback back to the proxy."""
    import one_memory.http_client as http_client

    monkeypatch.setattr(http_client, "LOOPBACK_DIRECT_MOUNTS", {})
    expected = _origin(httpx.URL(env_proxy))
    with create_client() as client:
        routed = {url: proxy_for(client, url) for url in LOOPBACK_URLS}
    assert all(proxy == expected for proxy in routed.values()), (
        f"loopback is still direct without the mount, so the guard is not guarding the mount: "
        f"{routed}"
    )
