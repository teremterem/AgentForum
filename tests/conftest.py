"""
Pytest configuration for the AgentForum framework. It is loaded by pytest automatically.
"""
import pytest

from agentforum.forum import Forum, InteractionContext


@pytest.fixture
def forum() -> Forum:
    """Create a forum with in-memory storage."""
    return Forum()


@pytest.fixture
def fake_interaction_context(forum: Forum) -> None:  # pylint: disable=redefined-outer-name
    """Create a fake interaction context."""
    # TODO TODO TODO Oleksandr: we shouldn't need this fixture as soon as aflatten_message_sequence() becomes
    #  independent of Forum
    # noinspection PyTypeChecker
    with InteractionContext(forum=forum, agent=None, request_messages=None, response_producer=None) as ctx:
        yield ctx
