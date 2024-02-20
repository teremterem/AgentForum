# pylint: disable=redefined-outer-name
"""
Pytest configuration for the AgentForum framework. It is loaded by pytest automatically.
"""
from unittest.mock import MagicMock

import pytest

from agentforum.forum import Forum, InteractionContext, Agent


@pytest.fixture
def forum() -> Forum:
    """
    Create a Forum instance with in-memory ForumTrees.
    """
    return Forum()


@pytest.fixture
def fake_agent(forum: Forum) -> Agent:
    """
    Create a fake Agent instance. Needed in the `fake_interaction_context` fixture (see its docstring below for
    details).
    """
    # noinspection PyTypeChecker
    return Agent(forum=forum, func=lambda *args, **kwargs: None, alias="FAKE_AGENT")


@pytest.fixture
def fake_interaction_context(forum: Forum, fake_agent: Agent) -> InteractionContext:
    """
    Activate a fake InteractionContext. Needed when utility functions are tested that depend on the presence of an
    active InteractionContext (because they need access to a Forum object and its ForumTrees to work). In the client
    code such functions are supposed to be called from within agents.
    """
    # noinspection PyTypeChecker
    return InteractionContext(
        forum=forum,
        agent=fake_agent,
        history_tracker=MagicMock(),
        request_messages=MagicMock(),
        response_producer=MagicMock(),
    )
