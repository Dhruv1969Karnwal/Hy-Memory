"""client_meta pass-through for the L0 basic profile layer.

Observed in production: among the memories one write produces, l2_fact returns its meta
while l0_basic_info does not -- the profile uses its own upsert path and dropped it during pass-through.
This suite locks in the behaviour after the fix.
"""
import asyncio

from hy_memory.agent.tools.basic_profile import upsert_basic_profile
from hy_memory.models.memory import MemoryLayer, MemoryStatus


class FakeVectorStore:
    def __init__(self, nodes=None):
        self.nodes = list(nodes or [])
        self.upserted = []
        self.payload_updates = []

    async def list_by_user(self, **kwargs):
        return list(self.nodes)

    async def get_by_id(self, node_id):
        return next((n for n in self.nodes if n.node_id == node_id), None)

    async def upsert(self, node):
        self.upserted.append(node)
        self.nodes.append(node)

    async def update_payload(self, node_id, updates):
        self.payload_updates.append((node_id, updates))


class FakeEmbedService:
    async def embed_queued(self, text):
        return [0.1] * 8


FIELDS = ["name", "age", "location"]


def upsert(store, kv, meta=None, user="u1"):
    return asyncio.run(
        upsert_basic_profile(
            user_id=user,
            kv=kv,
            vector_store=store,
            embed_service=FakeEmbedService(),
            allowed_fields=FIELDS,
            client_meta=meta,
        )
    )


class TestBasicProfileClientMeta:
    def test_meta_attached_to_new_node(self):
        store = FakeVectorStore()
        meta = {"source": "codebuddy", "trace_id": "t-1"}

        r = upsert(store, {"name": "Zhang Wei"}, meta=meta)

        assert r.success and r.changed
        assert len(store.upserted) == 1
        node = store.upserted[0]
        assert node.layer == MemoryLayer.L0_BASIC_INFO
        assert node.custom["meta"] == [meta]

    def test_basic_info_kv_still_stored_alongside_meta(self):
        """meta must not displace the existing basic_info_kv, which is what the profile is assembled from."""
        store = FakeVectorStore()

        upsert(store, {"name": "Zhang Wei", "age": 35}, meta={"k": "v"})

        custom = store.upserted[0].custom
        assert custom["basic_info_kv"] == {"name": "Zhang Wei", "age": 35}
        assert custom["meta"] == [{"k": "v"}]

    def test_no_meta_key_when_not_provided(self):
        """When meta is not passed no such key should be invented, otherwise the shape of old data changes."""
        store = FakeVectorStore()

        upsert(store, {"name": "Zhang Wei"})

        assert "meta" not in store.upserted[0].custom

    def test_empty_meta_treated_as_absent(self):
        store = FakeVectorStore()

        upsert(store, {"name": "Zhang Wei"}, meta={})

        assert "meta" not in store.upserted[0].custom

    def test_meta_is_copied_not_referenced(self):
        """Mutating the caller's own dict after the write must not affect what was already persisted."""
        store = FakeVectorStore()
        meta = {"source": "a"}

        upsert(store, {"name": "Zhang Wei"}, meta=meta)
        meta["source"] = "mutated"

        assert store.upserted[0].custom["meta"] == [{"source": "a"}]

    def test_new_version_carries_only_current_meta(self):
        """Every L0 change creates a node and turns the old one into SUPERSEDED, mirroring SUPERSEDE:
        a new node carries only this call's meta and does not inherit the old chain, avoiding unbounded accumulation on long chains."""
        store = FakeVectorStore()

        upsert(store, {"name": "Zhang Wei"}, meta={"round": 1})
        upsert(store, {"name": "Zhang Wei", "age": 35}, meta={"round": 2})

        assert len(store.upserted) == 2
        assert store.upserted[1].custom["meta"] == [{"round": 2}]

        # The old node is marked superseded and its meta stays on the chain unchanged
        assert store.payload_updates
        old_id, updates = store.payload_updates[-1]
        assert old_id == store.upserted[0].node_id
        assert updates["status"] == MemoryStatus.SUPERSEDED.value
        assert store.upserted[0].custom["meta"] == [{"round": 1}]

    def test_no_diff_writes_nothing_even_with_meta(self):
        """With no content change no new version is persisted, so meta alone must not create a node out of thin air."""
        store = FakeVectorStore()

        upsert(store, {"name": "Zhang Wei"}, meta={"round": 1})
        r = upsert(store, {"name": "Zhang Wei"}, meta={"round": 2})

        assert r.success and not r.changed
        assert len(store.upserted) == 1
