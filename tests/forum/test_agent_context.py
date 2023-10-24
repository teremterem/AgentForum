"""Tests for the AgentContext class."""
import asyncio

import pytest

from agentcache.forum import AgentContext, Forum, StreamedMessage, MessageSequence
from agentcache.storage import InMemoryStorage


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


@pytest.mark.asyncio
async def test_agents_call_each_other() -> None:
    """Verify that if agents call each other, their conversation history is recorded correctly."""
    # TODO Oleksandr: move this test out to test_agent_collab_scenarios.py
    forum = Forum(immutable_storage=InMemoryStorage())

    @forum.agent
    async def _agent1(request: StreamedMessage, responses: MessageSequence) -> None:
        responses2 = _agent2.call(request=request)
        async for msg in responses2:
            responses.send(msg)
        responses.send(
            await forum.anew_message("agent1 also says hello", reply_to=await responses2.aget_concluding_message())
        )

    @forum.agent
    async def _agent2(request: StreamedMessage, responses: MessageSequence) -> None:
        msg1 = await forum.anew_message("agent2 says hello", reply_to=request)
        responses.send(msg1)
        responses.send(await forum.anew_message("agent2 says hello again", reply_to=msg1))

    responses1 = _agent1.call(await forum.anew_message("user says hello"))

    actual_conversation = [
        (msg.ac_model_, msg.sender_alias, msg.content)
        for msg in [
            await streamed_msg.aget_full_message()
            for streamed_msg in await (await responses1.aget_concluding_message()).aget_full_chat()
        ]
    ]
    expected_conversation = [
        ("message", "USER", "user says hello"),
        ("message", "_agent2", "agent2 says hello"),
        ("message", "_agent2", "agent2 says hello again"),
        ("message", "_agent1", "agent1 also says hello"),
    ]
    assert actual_conversation == expected_conversation
