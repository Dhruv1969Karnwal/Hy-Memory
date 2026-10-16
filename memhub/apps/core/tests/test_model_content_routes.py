import asyncio

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from one_memory.api import create_app
from one_memory.config import Settings
from one_memory.memory_server.config_state import ConfigStore
from one_memory.model_content_routes import model_content_router
from one_memory.secrets import InMemorySecretStore


def test_permission_route_reads_and_writes_only_config_final_state(tmp_path) -> None:
    async def scenario() -> None:
        backfills = 0

        def start_backfill() -> None:
            nonlocal backfills
            backfills += 1

        app = FastAPI()
        app.state.config_store = ConfigStore(tmp_path)
        app.state.start_memory_backfill = start_backfill
        app.include_router(model_content_router([]))

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            assert (await client.get("/api/v1/model-content-permission")).json() == {
                "allowed": False
            }
            enabled = await client.post(
                "/api/v1/model-content-permission",
                json={"allowed": True},
            )
            assert enabled.status_code == 200
            assert enabled.json() == {"allowed": True}
            assert backfills == 1
            assert ConfigStore(tmp_path).model_content_allowed()

            disabled = await client.post(
                "/api/v1/model-content-permission",
                json={"allowed": False},
            )
            assert disabled.status_code == 200
            assert disabled.json() == {"allowed": False}
            assert backfills == 1

            legacy = await client.post(
                "/api/v1/model-content-permission/preview",
                json={},
            )
            assert legacy.status_code == 404

    asyncio.run(scenario())


def test_core_composition_starts_with_config_permission_and_durable_drain(tmp_path) -> None:
    async def scenario() -> None:
        app = create_app(
            Settings(
                _env_file=None,
                data_dir=tmp_path,
                source_home=tmp_path,
                api_token=SecretStr("test-token"),
                automatic_memory_processing=False,
                live_sync_enabled=False,
            ),
            source_catalog={},
            secret_store=InMemorySecretStore(),
        )
        async with (
            app.router.lifespan_context(app),
            AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
                headers={"Authorization": "Bearer test-token"},
            ) as client,
        ):
            response = await client.get("/api/v1/model-content-permission")
            assert response.status_code == 200
            assert response.json() == {"allowed": False}
            assert app.state.memory_write_drain_task.done() is False
            assert app.state.processing_coordinator is None

    asyncio.run(scenario())
