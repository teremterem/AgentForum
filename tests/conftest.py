"""Pytest configuration for the AgentCache framework. It is loaded by pytest automatically."""
import pytest

from agentforum.forum import Forum
from agentforum.storage import InMemoryStorage


@pytest.fixture
def forum() -> Forum:
    """Create a forum with in-memory storage."""
    return Forum(immutable_storage=InMemoryStorage())
