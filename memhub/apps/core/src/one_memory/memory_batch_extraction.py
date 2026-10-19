import json
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from one_memory.http_client import create_async_client
from one_memory.imports.models import EvidenceRecord
from one_memory.model_profiles import (
    ModelProfileSnapshot,
    ProviderKind,
    attach_disable_thinking,
)


class MemoryBatchExtractionError(RuntimeError):
    pass


class ExtractedMemoryDraft(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(min_length=1, max_length=512)
    support_quote: str = Field(min_length=2, max_length=120)
    content: str = Field(min_length=1, max_length=160)
    layer: str = Field(pattern="^(L0_BASIC_INFO|L2_FACT|L4_IDENTITY|L7_INTENTION)$")


class _ModelBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memories: tuple[ExtractedMemoryDraft, ...] = Field(max_length=24)


class MemoryBatchExtractor:
    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.transport = transport

    async def extract(
        self,
        records: Sequence[EvidenceRecord],
        *,
        profile: ModelProfileSnapshot,
        api_key: str | None,
    ) -> tuple[ExtractedMemoryDraft, ...]:
        if not records:
            return ()
        if profile.provider_kind not in {
            ProviderKind.OLLAMA,
            ProviderKind.OPENAI_COMPATIBLE,
        }:
            raise MemoryBatchExtractionError("memory_batch_provider_unsupported")
        if profile.provider_kind is ProviderKind.OPENAI_COMPATIBLE and not api_key:
            raise MemoryBatchExtractionError("model_api_key_missing")

        records_by_id = {record.evidence_id: record for record in records}
        payload = {
            "model": profile.model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a high-precision personal memory extractor. The input is several "
                        "independent real person quotes. "
                        "Emit at most one long-term useful memory per quote, or none at all. "
                        "Never write a task request, "
                        "temporary context, a question, assistant behaviour or a guess as a "
                        "memory. content must be self-contained, "
                        "in the language of the input, starting with 'The user' in English and "
                        "the Chinese equivalent of user otherwise, "
                        "and at most 160 characters. support_quote must be copied verbatim from "
                        "the matching verbatim for "
                        "2 to 120 characters with no rewriting; emit nothing when there is no "
                        "directly citable evidence. "
                        "evidence_id must copy the input id exactly. "
                        "layer may only be L0_BASIC_INFO, L2_FACT, L4_IDENTITY or "
                        "L7_INTENTION. Output strict JSON: "
                        '{"memories":[{"evidence_id":"...","support_quote":"source text",'
                        '"content":"...",'
                        '"layer":"L2_FACT"}]}. '
                        "Negative examples: 'research X for me', 'Can you review this?' and 'I "
                        "need you to change the page' "
                        "are tasks, not personal memories. Positive example: 'I have always "
                        "preferred concise answers' can be extracted, "
                        "and its support_quote must be 'I have always preferred concise answers'."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "records": [
                                {
                                    "evidence_id": record.evidence_id,
                                    "verbatim": record.verbatim,
                                }
                                for record in records
                            ]
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "one_memory_batch",
                    "strict": True,
                    "schema": _ModelBatch.model_json_schema(),
                },
            },
        }
        attach_disable_thinking(profile, payload)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        base_url = (profile.base_url or "").rstrip("/")
        if profile.provider_kind is ProviderKind.OLLAMA and not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        try:
            async with create_async_client(
                timeout=profile.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                parsed = _parse_model_batch(response.json())
        except (httpx.HTTPError, ValueError, ValidationError, KeyError, TypeError) as error:
            raise MemoryBatchExtractionError("memory_batch_extraction_failed") from error

        seen: set[str] = set()
        accepted: list[ExtractedMemoryDraft] = []
        for memory in parsed.memories:
            record = records_by_id.get(memory.evidence_id)
            if (
                record is None
                or memory.evidence_id in seen
                or memory.support_quote not in record.verbatim
            ):
                continue
            seen.add(memory.evidence_id)
            accepted.append(memory)
        return tuple(accepted)


def _parse_model_batch(body: dict[str, Any]) -> _ModelBatch:
    content = body["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("model content is not text")
    raw = json.loads(content)
    if not isinstance(raw, dict) or not isinstance(raw.get("memories"), list):
        raise TypeError("model batch is not a memories object")
    memories = []
    for item in raw["memories"]:
        try:
            memories.append(ExtractedMemoryDraft.model_validate(item))
        except ValidationError:
            continue
    return _ModelBatch(memories=tuple(memories))
