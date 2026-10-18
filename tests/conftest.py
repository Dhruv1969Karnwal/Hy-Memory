# -*- coding: utf-8 -*-
"""
HY Memory SDK - pytest configuration

Shared fixtures and configuration.
"""

import os
from pathlib import Path

import pytest


# Load the .env file (if present), searching parent directories
def _load_env():
    cur = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = cur / ".env"
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip()
                    if k and k not in os.environ:
                        os.environ[k] = v
            return
        cur = cur.parent


_load_env()


def pytest_configure(config):
    """pytest configuration hook"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


@pytest.fixture(scope="session")
def test_uid():
    """uid used by tests"""
    return "test_user"


@pytest.fixture(scope="session")
def test_agent_id():
    """agent_id used by tests"""
    return "test_scene"


@pytest.fixture
def sample_text_content():
    """Sample text content"""
    return "The user likes hiking on weekends, especially the Fragrant Hills in Beijing"


@pytest.fixture
def sample_qa_pair():
    """Sample QA pair"""
    from hy_memory.models import QAPair
    return QAPair(
        question="What is your favorite food?",
        answer="I love hot pot, especially Sichuan spicy hot pot"
    )


@pytest.fixture
def sample_dialogue():
    """Sample dialogue"""
    return [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
        {"role": "user", "content": "I would like to know about the weather"},
        {"role": "assistant", "content": "Beijing is sunny today, 15-25 degrees"}
    ]
