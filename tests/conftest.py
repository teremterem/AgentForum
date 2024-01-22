"""
Pytest configuration for the AgentForum framework. It is loaded by pytest automatically.
"""
import pytest

from agentforum.forum import Forum, InteractionContext


@pytest.fixture
def forum() -> Forum:
    """
    Create a Forum instance with in-memory ForumTrees.
    """
    return Forum()


@pytest.fixture
def fake_interaction_context(forum: Forum) -> None:  # pylint: disable=redefined-outer-name
    """
    Activate a fake InteractionContext. Needed when utility functions are tested that depend on the presence of an
    active InteractionContext (because they need access to a Forum object and its ForumTrees to work). In the client
    code such functions are supposed to be called from within agents.
    """
    # noinspection PyTypeChecker
    with InteractionContext(forum=forum, agent=None, request_messages=None, response_producer=None) as ctx:
        yield ctx
