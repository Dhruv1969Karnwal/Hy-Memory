"""Where an HY write spends the time that is not a model call."""

import asyncio
import json
import time

import httpx

from one_memory.diagnostic_log import configure_diagnostics, reset_diagnostics
from one_memory.hy_permission_context import permission_bridge_factory
from one_memory.hy_step_timing import HyStepTimer
from tests.test_hy_permission_context import whole_client


def _handler(record):
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        if request.url.path.endswith("embeddings"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "test-embedding",
                    "data": [
                        {"index": i, "object": "embedding", "embedding": [1.0] + [0.0] * 383}
                        for i, _ in enumerate(body["input"])
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "fixture-response",
                "object": "chat.completion",
                "created": 1,
                "model": "test-generation",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                [
                                    {
                                        "op": "ADD",
                                        "content": record.verbatim,
                                        "layer": "L2_FACT",
                                        "owner": "agent",
                                    }
                                ]
                            ),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    return handler


def test_leased_ingest_logs_each_hy_step_with_duration_and_count(tmp_path, monkeypatch):
    # Per-step tracing is its own tier on top of the master switch, so a test
    # that reads the step events has to ask for both.
    monkeypatch.setenv("ONE_MEMORY_DIAGNOSTICS", "1")
    monkeypatch.setenv("ONE_MEMORY_HY_STEP_TRACE", "1")
    receipt = configure_diagnostics(tmp_path / "diagnostics")

    async def scenario():
        with whole_client(tmp_path) as (supervisor, record, _, _):
            supervisor.permission_bridge_factory = permission_bridge_factory(
                supervisor._bridge, http_transport=httpx.MockTransport(_handler(record))
            )
            tasks = supervisor.prepare_conversation(record)
            try:
                result = await supervisor._create_runner().run(tasks[0].task_id)
                assert result.status == "completed", result.last_error_code
            finally:
                await supervisor.shutdown()
            return record, tasks[0].task_id

    try:
        record, task_id = asyncio.run(scenario())
    finally:
        reset_diagnostics()
    written = receipt.path.read_text()
    events = [json.loads(line) for line in written.splitlines()]
    steps = {
        event["details"]["phase"]: event
        for event in events
        if event["event"] == "hy.step.completed"
    }
    # Memory content with no prior candidate takes HY's no-model ADD path.
    assert "model" not in steps
    expected = {
        "embedding": "hy.embedding",
        "vector_store_read": "hy.vector_store",
        "vector_store_write": "hy.vector_store",
        "permission_check": "hy.permission_store",
        "permission_store_read": "hy.permission_store",
        "permission_store_write": "hy.permission_store",
        "write_verification": "hy.write_verification",
        "reconcile": "hy.reconcile",
    }
    assert expected.items() <= {name: event["component"] for name, event in steps.items()}.items()
    for event in steps.values():
        assert event["task_id"] == task_id
        assert event["request_id"]
        assert isinstance(event["duration_ms"], int)
        assert event["details"]["call_count"] >= 1
        assert event["details"]["summed_ms"] + 1 >= event["duration_ms"]
    summaries = [event for event in events if event["event"] == "hy.steps.summary"]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["component"] == "hy.bridge"
    assert summary["task_id"] == task_id
    details = summary["details"]
    assert details["model_ms"] == 0
    assert details["accounted_ms"] + details["unaccounted_ms"] == summary["duration_ms"]
    assert details["accounted_ms"] + 1 >= max(event["duration_ms"] for event in steps.values())
    assert record.verbatim not in written


def test_nested_same_step_counts_once_and_overlap_counts_wall_clock_once():
    timer = HyStepTimer()
    with timer.step("permission_check"), timer.step("permission_check"):
        time.sleep(0.02)

    async def overlapping():
        async def one():
            with timer.step("embedding", items=2):
                await asyncio.sleep(0.05)

        await asyncio.gather(one(), one())

    asyncio.run(overlapping())
    measured = timer.measurements()
    assert measured["permission_check"].call_count == 1
    embedding = measured["embedding"]
    assert embedding.call_count == 2
    assert embedding.item_count == 4
    assert embedding.summed_ms >= 95
    assert 45 <= embedding.busy_ms < 90
