"""Tests for the InteractionContext class."""

import asyncio
from unittest.mock import MagicMock

import pytest

from agentforum.forum import InteractionContext


@pytest.mark.asyncio
async def test_nested_interaction_contexts() -> None:
    """Assert that nesting of interaction contexts works as expected."""
    assert InteractionContext.get_current_context() is None

    async with _create_interaction_context("agent1") as ctx1:
        assert InteractionContext.get_current_context() is ctx1
        async with _create_interaction_context("agent2") as ctx2:
            assert InteractionContext.get_current_context() is ctx2
        assert InteractionContext.get_current_context() is ctx1

        async with _create_interaction_context("agent3") as ctx3:
            assert InteractionContext.get_current_context() is ctx3
            async with _create_interaction_context("agent4") as ctx4:
                assert InteractionContext.get_current_context() is ctx4
            assert InteractionContext.get_current_context() is ctx3

        assert InteractionContext.get_current_context() is ctx1

    assert InteractionContext.get_current_context() is None


@pytest.mark.asyncio
async def test_interaction_contexts_with_create_task() -> None:
    """Assert that nesting of interaction contexts works as expected even when asyncio.create_task() is involved."""
    ctx0 = _create_interaction_context("agent0")

    async def task1() -> None:
        assert InteractionContext.get_current_context() is ctx0
        async with _create_interaction_context("agent1") as ctx1:
            await asyncio.sleep(0.01)
            assert InteractionContext.get_current_context() is ctx1
            await asyncio.sleep(0.01)
        assert InteractionContext.get_current_context() is ctx0

    async def task2() -> None:
        assert InteractionContext.get_current_context() is ctx0
        async with _create_interaction_context("agent2") as ctx2:
            await asyncio.sleep(0.01)
            assert InteractionContext.get_current_context() is ctx2
            await asyncio.sleep(0.01)
        assert InteractionContext.get_current_context() is ctx0

    assert InteractionContext.get_current_context() is None
    async with ctx0:
        assert InteractionContext.get_current_context() is ctx0
        await asyncio.gather(asyncio.create_task(task1()), asyncio.create_task(task2()))
        assert InteractionContext.get_current_context() is ctx0
    assert InteractionContext.get_current_context() is None


def _create_interaction_context(agent_alias: str) -> InteractionContext:
    """Create an interaction context with the given agent alias."""
    return InteractionContext(
        forum=MagicMock(),
        agent=MagicMock(alias=agent_alias),
        request_messages=MagicMock(),
        response_producer=MagicMock(),
    )
