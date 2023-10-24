"""Test different agent collaboration scenarios."""
from typing import List, Tuple

import pytest

from agentcache.forum import Forum, StreamedMessage, MessageSequence


@pytest.mark.asyncio
async def test_two_nested_agents(forum: Forum) -> None:
    """
    Verify that when one agent, in order to serve the user, calls another agent "behind the scenes", the conversation
    history is recorded correctly.
    """

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

    await aassert_conversation(
        responses1,
        [
            ("message", "USER", "user says hello"),
            ("message", "_agent2", "agent2 says hello"),
            ("message", "_agent2", "agent2 says hello again"),
            ("message", "_agent1", "agent1 also says hello"),
        ],
    )


async def aassert_conversation(responses: MessageSequence, expected_conversation: List[Tuple[str, str, str]]) -> None:
    """
    Assert that the conversation recorded in the given responses matches the expected conversation. This function
    tests the full chat history, including messages that preceded the current `responses` MessageSequence.
    """
    actual_conversation = [
        (msg.ac_model_, msg.sender_alias, msg.content)
        for msg in [
            await streamed_msg.aget_full_message()
            for streamed_msg in await (await responses.aget_concluding_message()).aget_full_chat()
        ]
    ]
    assert actual_conversation == expected_conversation
