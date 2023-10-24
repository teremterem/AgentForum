"""Tests for the AgentContext class."""
import asyncio

import pytest

from agentcache.forum import AgentContext


def test_nested_agent_contexts() -> None:
    """Assert that nesting of agent contexts works as expected."""
    assert AgentContext.get_current_context() is None

    with AgentContext("agent1") as ctx1:
        assert AgentContext.get_current_context() is ctx1
        with AgentContext("agent2") as ctx2:
            assert AgentContext.get_current_context() is ctx2
        assert AgentContext.get_current_context() is ctx1

        with AgentContext("agent3") as ctx3:
            assert AgentContext.get_current_context() is ctx3
            with AgentContext("agent4") as ctx4:
                assert AgentContext.get_current_context() is ctx4
            assert AgentContext.get_current_context() is ctx3

        assert AgentContext.get_current_context() is ctx1

    assert AgentContext.get_current_context() is None


@pytest.mark.asyncio
async def test_agent_contexts_with_create_task() -> None:
    """Assert that nesting of agent contexts works as expected even when asyncio.create_task() is involved."""
    ctx0 = AgentContext("agent0")

    async def task1() -> None:
        assert AgentContext.get_current_context() is ctx0
        with AgentContext("agent1") as ctx1:
            await asyncio.sleep(0.01)
            assert AgentContext.get_current_context() is ctx1
            await asyncio.sleep(0.01)
        assert AgentContext.get_current_context() is ctx0

    async def task2() -> None:
        assert AgentContext.get_current_context() is ctx0
        with AgentContext("agent2") as ctx2:
            await asyncio.sleep(0.01)
            assert AgentContext.get_current_context() is ctx2
            await asyncio.sleep(0.01)
        assert AgentContext.get_current_context() is ctx0

    assert AgentContext.get_current_context() is None
    with ctx0:
        assert AgentContext.get_current_context() is ctx0
        await asyncio.gather(asyncio.create_task(task1()), asyncio.create_task(task2()))
        assert AgentContext.get_current_context() is ctx0
    assert AgentContext.get_current_context() is None
