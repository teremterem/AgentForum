"""Test different agent collaboration scenarios."""
from typing import List, Tuple, Union

import pytest

from agentcache.forum import Forum, MessagePromise, MessageSequence, InteractionContext


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
    async def _assistant(request: MessagePromise, ctx: InteractionContext) -> None:
        api_response = await _reminder_api.call(request).aget_concluding_message()
        if (await api_response.amaterialize()).content.startswith("api error:"):
            # TODO Oleksandr: implement actual ErrorMessage class
            correction = await _critic.call(api_response).aget_concluding_message()
            api_response = await _reminder_api.call(correction).aget_concluding_message()

        await aassert_conversation(
            api_response,
            [
                ("message", "USER", "set a reminder for me for tomorrow at 10am"),
                ("call", "_assistant", "_reminder_api"),
                ("message", "_reminder_api", "api error: invalid date format"),
                ("call", "_assistant", "_critic"),
                ("message", "_critic", "try swapping the month and day"),
                ("call", "_assistant", "_reminder_api"),
                ("message", "_reminder_api", "success: reminder set"),
            ],
        )

        ctx.respond(api_response)

    @forum.agent
    async def _reminder_api(request: MessagePromise, ctx: InteractionContext) -> None:
        if (await request.amaterialize()).sender_alias == "_critic":
            ctx.respond("success: reminder set")
        else:
            ctx.respond("api error: invalid date format")

    @forum.agent
    async def _critic(_: MessagePromise, ctx: InteractionContext) -> None:
        # TODO Oleksandr: turn this agent into a proxy agent
        ctx.respond("try swapping the month and day")

    assistant_responses = _assistant.call(forum.new_message_promise("set a reminder for me for tomorrow at 10am"))

    await aassert_conversation(
        assistant_responses,
        [
            ("message", "USER", "set a reminder for me for tomorrow at 10am"),
            ("call", "USER", "_assistant"),
            ("forward", "_assistant", "success: reminder set"),
        ],
    )


@pytest.mark.asyncio
async def test_two_nested_agents(forum: Forum) -> None:
    """
    Verify that when one agent, in order to serve the user, calls another agent "behind the scenes", the conversation
    history is recorded correctly.
    """

    @forum.agent
    async def _agent1(request: MessagePromise, ctx: InteractionContext) -> None:
        async for msg in _agent2.call(request):
            ctx.respond(msg)
        # TODO Oleksandr: replace the above with something like this, when ForwardedMessages are supported:
        #  responses.send(_agent2.call(request))
        ctx.respond("agent1 also says hello")

    @forum.agent
    async def _agent2(_: MessagePromise, ctx: InteractionContext) -> None:
        ctx.respond("agent2 says hello")
        ctx.respond("agent2 says hello again")

    responses1 = _agent1.call(forum.new_message_promise("user says hello"))

    await aassert_conversation(
        responses1,
        [
            ("message", "USER", "user says hello"),
            ("call", "USER", "_agent1"),
            # TODO Oleksandr: assert that the "original sender" is "_agent2" in the following two messages
            ("forward", "_agent1", "agent2 says hello"),
            ("forward", "_agent1", "agent2 says hello again"),
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
        for msg in await concluding_msg.amaterialize_history(skip_agent_calls=False)
    ]
    assert actual_conversation == expected_conversation
