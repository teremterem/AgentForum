"""Test different agent collaboration scenarios."""
from typing import List, Tuple, Union

import pytest

from agentcache.forum import Forum, MessagePromise, MessageSequence


@pytest.mark.asyncio
async def test_api_call_error_recovery(forum: Forum) -> None:
    """
    ("message", "USER", "set a reminder for me for tomorrow at 10am"),
        # _assistant forwards the message to _reminder_api (separate conversation thread)
    ("message", "_reminder_api", "api error: invalid date format"),
        # _critic is a proxy agent that intercepts error messages and tells _reminder_api what to correct
    ("message", "_critic", "try swapping the month and day"),
    ("message", "_reminder_api", "success: reminder set"),
        # _assistant forwards this last message to the user (in the original conversation thread) as its own
    """

    @forum.agent
    async def _assistant(request: MessagePromise, responses: MessageSequence) -> None:
        api_response = await _reminder_api.call(request=request).aget_concluding_message()
        if (await api_response.aget_content()).startswith("api error:"):
            # TODO Oleksandr: implement actual ErrorMessage class
            correction = await _critic.call(request=api_response).aget_concluding_message()
            api_response = await _reminder_api.call(request=correction).aget_concluding_message()

        await aassert_conversation(
            api_response,
            [
                ("message", "USER", "set a reminder for me for tomorrow at 10am"),
                ("message", "_reminder_api", "api error: invalid date format"),
                ("message", "_critic", "try swapping the month and day"),
                ("message", "_reminder_api", "success: reminder set"),
            ],
        )

        response = await forum.anew_message(await api_response.aget_content(), reply_to=request)
        responses.send(response)

    @forum.agent
    async def _reminder_api(request: MessagePromise, responses: MessageSequence) -> None:
        if request.sender_alias == "_critic":
            responses.send(await forum.anew_message("success: reminder set", reply_to=request))
        else:
            responses.send(await forum.anew_message("api error: invalid date format", reply_to=request))

    @forum.agent
    async def _critic(request: MessagePromise, responses: MessageSequence) -> None:
        # TODO Oleksandr: turn this agent into a proxy agent
        responses.send(await forum.anew_message("try swapping the month and day", reply_to=request))

    assistant_responses = _assistant.call(await forum.anew_message("set a reminder for me for tomorrow at 10am"))

    await aassert_conversation(
        assistant_responses,
        [
            ("message", "USER", "set a reminder for me for tomorrow at 10am"),
            ("message", "_assistant", "success: reminder set"),
        ],
    )


@pytest.mark.asyncio
async def test_two_nested_agents(forum: Forum) -> None:
    """
    Verify that when one agent, in order to serve the user, calls another agent "behind the scenes", the conversation
    history is recorded correctly.
    """

    @forum.agent
    async def _agent1(request: MessagePromise, responses: MessageSequence) -> None:
        responses2 = _agent2.call(request=request)
        async for msg in responses2:
            responses.send(msg)
        responses.send(
            await forum.anew_message("agent1 also says hello", reply_to=await responses2.aget_concluding_message())
        )

    @forum.agent
    async def _agent2(request: MessagePromise, responses: MessageSequence) -> None:
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


async def aassert_conversation(
    response: Union[MessagePromise, MessageSequence], expected_conversation: List[Tuple[str, str, str]]
) -> None:
    """
    Assert that the conversation recorded in the given responses matches the expected conversation. This function
    tests the full chat history, including messages that preceded the current `responses` MessageSequence.
    """
    concluding_msg = response if isinstance(response, MessagePromise) else await response.aget_concluding_message()
    actual_conversation = [
        (msg.ac_model_, msg.sender_alias, msg.content)
        for msg in [await msg_promise.amaterialize() for msg_promise in await concluding_msg.aget_full_chat()]
    ]
    assert actual_conversation == expected_conversation
