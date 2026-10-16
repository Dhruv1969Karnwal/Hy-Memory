"""One global final-state permission for remote model content egress."""

import asyncio
from collections.abc import Sequence
from typing import Annotated, cast

from fastapi import APIRouter, Request
from fastapi.params import Depends
from pydantic import BaseModel, ConfigDict, StrictBool

from one_memory.memory_server.config_state import ConfigStore


class ModelContentPermissionState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool


class SetModelContentPermission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: StrictBool


def model_content_router(dependencies: Sequence[Depends]) -> APIRouter:
    router = APIRouter(prefix="/api/v1/model-content-permission", dependencies=dependencies)

    def config(request: Request) -> ConfigStore:
        return cast(ConfigStore, request.app.state.config_store)

    Config = Annotated[ConfigStore, Depends(config)]

    @router.get("", response_model=ModelContentPermissionState)
    async def current(store: Config) -> ModelContentPermissionState:
        return ModelContentPermissionState(allowed=store.model_content_allowed())

    @router.post("", response_model=ModelContentPermissionState)
    async def set_permission(
        request: Request,
        value: SetModelContentPermission,
        store: Config,
    ) -> ModelContentPermissionState:
        await asyncio.to_thread(store.set_model_content_allowed, value.allowed)
        if value.allowed:
            request.app.state.start_memory_backfill()
        return ModelContentPermissionState(allowed=value.allowed)

    return router
