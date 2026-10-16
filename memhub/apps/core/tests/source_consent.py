"""Explicit fixture-user confirmation through metadata-only public endpoints."""

from uuid import uuid4


async def authorized_request(client, source_id, *, confirm_environment_override=False):
    location = await client.post(f"/api/v1/sources/{source_id}/location")
    location.raise_for_status()
    assert location.json()["state"] == "found"
    return {
        "expected_root_id": location.json()["root_id"],
        "user_confirmed": True,
        "confirm_environment_override": confirm_environment_override,
    }


async def thaw_source(client, source_id, receipt):
    """Turn on the single Harness processing switch after authorization."""
    del receipt
    response = await client.post(
        f"/api/v1/sources/{source_id}/sync-permission",
        json={"enabled": True},
    )
    response.raise_for_status()
    assert response.json()["sync_enabled"] is True
    return response.json()


async def confirm_model_content(
    client,
    *,
    source_ids,
    evidence_ids=(),
    all_imported=None,
    future_additions=True,
    process=True,
):
    """Grant model processing the same way the product does: preview, then confirm.

    Live suites need this because importing Evidence is not consent to send it
    to a model; without it the processing guard refuses, which is the correct
    product behaviour and not something a test may bypass by writing the store.
    """
    profiles = await client.get("/api/v1/model-profiles")
    profiles.raise_for_status()
    active = next(item for item in profiles.json() if item["is_active"])
    preview = await client.post(
        "/api/v1/model-content-permission/preview",
        json={
            "profile_id": active["profile_id"],
            "profile_revision": active["revision"],
            "source_ids": list(source_ids),
            "evidence_ids": list(evidence_ids),
            # The manual source only accepts named originals, so "everything
            # from these sources" is not a default the caller can ignore.
            "all_imported_from_selected_sources": (
                not evidence_ids if all_imported is None else all_imported
            ),
            "include_future_additions": future_additions,
        },
    )
    # Report the refusal code, not just the status: every content refusal is a
    # named product decision and a live run that hides it cannot be diagnosed.
    assert preview.status_code == 200, preview.text
    current = await client.get("/api/v1/model-content-permission")
    current.raise_for_status()
    existing = current.json()
    confirmed = await client.post(
        "/api/v1/model-content-permission",
        json={
            "snapshot": preview.json(),
            "expected_revision": existing["revision"] if existing else 0,
            "user_confirmed": True,
        },
    )
    confirmed.raise_for_status()
    receipt = confirmed.json()
    if process:
        # What the product does after the user confirms: take the already
        # imported Evidence through the pipeline instead of waiting for the
        # next import to trigger a debounce.
        accepted = await client.post(
            "/api/v1/model-content-permission/process",
            json={"expected_revision": receipt["revision"], "user_confirmed": True},
        )
        accepted.raise_for_status()
    return receipt


async def connection_request(client, source_ids, **options):
    roots = {}
    for source_id in source_ids:
        location = await client.post(f"/api/v1/sources/{source_id}/location")
        location.raise_for_status()
        assert location.json()["state"] == "found"
        roots[source_id] = location.json()["root_id"]
    return {
        "source_ids": source_ids,
        "request_id": str(uuid4()),
        "source_roots": roots,
        **options,
    }
