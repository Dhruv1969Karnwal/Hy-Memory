# -*- coding: utf-8 -*-
"""Shared pytest fixtures."""

import os
import shutil
import tempfile
import uuid

import pytest

from ._helpers import MockEmbedService


@pytest.fixture
def tmp_data_dir():
    d = tempfile.mkdtemp(prefix="codingmem_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def coding_config(tmp_data_dir):
    """Temporary chroma + sqlite; each fixture gets its own collection_name so in-process singletons do not cross-contaminate."""
    from hy_memory.config import MemoryConfig

    cfg = MemoryConfig.from_env()
    cfg.history.db_path = os.path.join(tmp_data_dir, "history.db")
    cfg.coding.db_path = os.path.join(tmp_data_dir, "coding_memory.db")
    cfg.vector_store.provider = "chroma"
    cfg.vector_store.persist_directory = os.path.join(tmp_data_dir, "chroma")
    # The uuid suffix gives every test its own collection (the chroma client may cache a singleton)
    cfg.vector_store.collection_name = f"test_coding_{uuid.uuid4().hex[:8]}"
    cfg.vector_store.embedding_dims = 1536
    return cfg


@pytest.fixture
def mock_embed():
    return MockEmbedService()


@pytest.fixture
async def coding_store(coding_config, mock_embed):
    """A CodingMemoryStore that has already been initialized"""
    from hy_memory.coding.store import CodingMemoryStore
    store = CodingMemoryStore(coding_config, embed_service=mock_embed)
    await store.initialize()
    yield store
    await store.close()
