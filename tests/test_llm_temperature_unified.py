"""LLM generation parameters must all go through user config; no call site may hardcode them.

Background: the Opus 5 gateway only accepts `temperature=1` or **nothing at all**; passing 0 / 0.1 / 0.3 / 0.6 all return 400.
The SDK previously had three kinds of writes that bypassed user config:

1. Hardcoded values at call sites (abstractor 0.3/0.4/0.2, reflector 0.1/0.3, merger 0.3)
2. Hardcoded dataclass defaults in sub-configs (emotion_analyzer 0.1, intention_detector 0.2/0.1,
   topic_summary 0.3, episode_summary 0.3)
3. provider entry default 0.7 -- if the caller passes nothing it falls to 0.7, fully unrelated to user config

As a result `MEMORY_LLM_TEMPERATURE` only took effect on a few paths, and switching gateways broke all the rest.
These tests lock in the invariant that "the single convergence point is the provider".
"""

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hy_memory.agent.llm_provider import LLMProvider
from hy_memory.config import LLMConfig, MemoryConfig

_SRC = Path(__file__).resolve().parent.parent / "hy_memory"


# ======================================================================
# Static guard: no hardcoded temperature may reappear
# ======================================================================

class TestNoHardcodedTemperature:
    def test_no_numeric_temperature_at_call_sites(self):
        """Literals like `temperature=0.3` are never allowed."""
        offenders = []
        for py in _SRC.rglob("*.py"):
            for i, line in enumerate(py.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # only inspect numeric literals in arguments/defaults
                if "temperature=" in stripped:
                    after = stripped.split("temperature=", 1)[1][:6]
                    if after and after[0].isdigit():
                        offenders.append(f"{py.relative_to(_SRC)}:{i}: {stripped}")
        assert not offenders, (
            "these call sites hardcode temperature and would bypass user config:\n" + "\n".join(offenders)
        )

    def test_no_float_annotated_temperature_defaults(self):
        """In sub-configs, `temperature: float = 0.3` is also a config-bypassing write."""
        offenders = []
        for py in _SRC.rglob("*.py"):
            for i, line in enumerate(py.read_text().splitlines(), 1):
                s = line.strip()
                if s.startswith("#"):
                    continue
                if ("temperature" in s and ": float" in s and "=" in s
                        and "Optional" not in s):
                    val = s.split("=", 1)[1].strip()
                    if val and val[0].isdigit():
                        offenders.append(f"{py.relative_to(_SRC)}:{i}: {s}")
        assert not offenders, (
            "these defaults hardcode temperature:\n" + "\n".join(offenders)
        )

    def test_provider_entrypoints_default_to_none(self):
        """The provider entry defaults must be None (= follow config), not 0.7."""
        for name in ("complete", "complete_messages", "chat",
                     "_call_llm", "_chat_completions"):
            param = inspect.signature(getattr(LLMProvider, name)).parameters["temperature"]
            assert param.default is None, (
                f"LLMProvider.{name}'s temperature default is {param.default!r}; "
                "callers that pass nothing would bypass user config"
            )


# ======================================================================
# Config resolution
# ======================================================================

class TestTemperatureResolution:
    def _provider(self, *, temperature=None, omit=False):
        cfg = MemoryConfig()
        cfg.llm.api_key = "sk-test"
        cfg.llm.model = "opus-5"
        if temperature is not None:
            cfg.llm.temperature = temperature
        cfg.llm.omit_temperature = omit
        return LLMProvider(cfg)

    def test_call_site_none_falls_back_to_config(self):
        p = self._provider(temperature=1.0)
        assert p._resolve_temperature(None) == 1.0

    def test_explicit_call_site_value_wins(self):
        """The few cases that genuinely need a specific temperature can still specify it explicitly."""
        p = self._provider(temperature=1.0)
        assert p._resolve_temperature(0.2) == 0.2

    def test_omit_config_resolves_to_none(self):
        p = self._provider(omit=True)
        assert p._resolve_temperature(None) is None

    def test_zero_is_not_treated_as_unset(self):
        """0.0 is a valid temperature and must not be swallowed by a falsy check."""
        p = self._provider(temperature=1.0)
        assert p._resolve_temperature(0.0) == 0.0


class TestConfigEnv:
    def test_env_numeric(self, monkeypatch):
        monkeypatch.setenv("MEMORY_LLM_TEMPERATURE", "1.0")
        cfg = LLMConfig()
        assert cfg.temperature == 1.0 and cfg.omit_temperature is False

    @pytest.mark.parametrize("val", ["none", "None", "null", "omit", "unset", ""])
    def test_env_omit_spellings(self, monkeypatch, val):
        monkeypatch.setenv("MEMORY_LLM_TEMPERATURE", val)
        assert LLMConfig().omit_temperature is True

    def test_env_explicit_omit_flag(self, monkeypatch):
        monkeypatch.setenv("MEMORY_LLM_OMIT_TEMPERATURE", "true")
        assert LLMConfig().omit_temperature is True

    def test_default_unchanged(self, monkeypatch):
        """Default behavior is unchanged (0.1, still sent), avoiding silently altering existing deployments."""
        monkeypatch.delenv("MEMORY_LLM_TEMPERATURE", raising=False)
        monkeypatch.delenv("MEMORY_LLM_OMIT_TEMPERATURE", raising=False)
        cfg = LLMConfig()
        assert cfg.temperature == 0.1 and cfg.omit_temperature is False


# ======================================================================
# Landing in the real request payload
# ======================================================================

class TestRequestPayload:
    """Final criterion: whether the request body carries it at all, and with what value."""

    def _capture(self):
        seen = []

        def _fake_client(*a, **kw):
            cli = MagicMock()

            async def _create(**kwargs):
                seen.append(kwargs)
                m = MagicMock()
                m.choices = [MagicMock(
                    message=MagicMock(content="ok", tool_calls=None),
                    finish_reason="stop",
                )]
                m.usage = MagicMock(
                    prompt_tokens=1, completion_tokens=1, total_tokens=2,
                )
                return m

            cli.chat.completions.create = _create
            return cli

        return seen, _fake_client

    async def _run(self, *, temperature=None, omit=False, call_kwargs=None):
        cfg = MemoryConfig()
        cfg.llm.api_key = "sk-test"
        cfg.llm.model = "opus-5"
        if temperature is not None:
            cfg.llm.temperature = temperature
        cfg.llm.omit_temperature = omit
        provider = LLMProvider(cfg)

        seen, fake = self._capture()
        with patch("openai.AsyncOpenAI", fake):
            await provider.complete(
                prompt="hi", max_tokens=10, **(call_kwargs or {}),
            )
        return seen[0]

    @pytest.mark.asyncio
    async def test_configured_value_is_sent(self):
        payload = await self._run(temperature=1.0)
        assert payload["temperature"] == 1.0

    @pytest.mark.asyncio
    async def test_omit_removes_key_entirely(self):
        """Opus 5's requirement: the parameter is **entirely absent**, not sent with a default value."""
        payload = await self._run(omit=True)
        assert "temperature" not in payload

    @pytest.mark.asyncio
    async def test_call_site_override_still_works(self):
        payload = await self._run(temperature=1.0, call_kwargs={"temperature": 0.2})
        assert payload["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_max_tokens_still_sent(self):
        """Omitting temperature must not accidentally drop other parameters."""
        payload = await self._run(omit=True)
        assert payload["max_tokens"] == 10
        assert payload["model"] == "opus-5"


# ======================================================================
# Independent resolvers on the coding path
# ======================================================================

class TestCodingPathsRespectConfig:
    """The three modules under coding previously each fell back via `getattr(..., 0.1)`, bypassing omit."""

    class _Provider:
        def __init__(self, temperature, omit):
            self.config = MagicMock()
            self.config.llm.temperature = temperature
            self.config.llm.omit_temperature = omit

    @pytest.mark.parametrize("module_path", [
        "hy_memory.coding.extractor",
        "hy_memory.coding.reconciler",
        "hy_memory.coding.fact_reconciler",
    ])
    def test_resolver_honors_omit_and_value(self, module_path):
        import importlib

        mod = importlib.import_module(module_path)
        fn = mod._resolve_configured_temperature
        assert fn(self._Provider(0.1, True)) is None, "must return None when omit is set"
        assert fn(self._Provider(1.0, False)) == 1.0

    def test_judge_resolver_honors_omit(self):
        from hy_memory.coding.judge import _resolve_llm_temperature

        assert _resolve_llm_temperature(self._Provider(0.1, True)) is None
        assert _resolve_llm_temperature(self._Provider(1.0, False)) == 1.0
