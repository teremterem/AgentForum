"""Pytest configuration for the AgentForum framework. It is loaded by pytest automatically."""
import pytest

from agentforum.forum import Forum


@pytest.fixture
def forum() -> Forum:
    """Create a forum with in-memory storage."""
    return Forum()
