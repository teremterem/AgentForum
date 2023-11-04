"""Test different agent collaboration scenarios."""
from typing import List, Union, Dict, Any

import pytest

from agentcache.forum import Forum, InteractionContext
from agentcache.models import Message, AgentCall
from agentcache.promises import MessagePromise, MessageSequence


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

        assert await represent_conversation_with_dicts(api_response) == [
            {
                "ac_model_": "message",
                "sender_alias": "USER",
                "content": "set a reminder for me for tomorrow at 10am",
            },
            {
                "ac_model_": "call",
                "sender_alias": "_assistant",
                "content": "_reminder_api",
                "messages_in_request": 1,
            },
            {
                "ac_model_": "message",
                "sender_alias": "_reminder_api",
                "content": "api error: invalid date format",
            },
            {
                "ac_model_": "call",
                "sender_alias": "_assistant",
                "content": "_critic",
                "messages_in_request": 1,
            },
            {
                "ac_model_": "message",
                "sender_alias": "_critic",
                "content": "try swapping the month and day",
            },
            {
                "ac_model_": "call",
                "sender_alias": "_assistant",
                "content": "_reminder_api",
                "messages_in_request": 1,
            },
            {
                "ac_model_": "message",
                "sender_alias": "_reminder_api",
                "content": "success: reminder set",
            },
        ]

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

    assert await represent_conversation_with_dicts(assistant_responses) == [
        {
            "ac_model_": "message",
            "sender_alias": "USER",
            "content": "set a reminder for me for tomorrow at 10am",
        },
        {
            "ac_model_": "call",
            "sender_alias": "USER",
            "content": "_assistant",
            "messages_in_request": 1,
        },
        {
            "ac_model_": "forward",
            "sender_alias": "_assistant",
            "original_msg": {
                "ac_model_": "message",
                "sender_alias": "_reminder_api",
                "content": "success: reminder set",
            },
        },
    ]


@pytest.mark.asyncio
async def test_two_nested_agents(forum: Forum) -> None:
    """
    Verify that when one agent, in order to serve the user, calls another agent "behind the scenes", the conversation
    history is recorded correctly.
    """

    @forum.agent
    async def _agent1(request: MessagePromise, ctx: InteractionContext) -> None:
        await ctx.arespond(_agent2.call(request))
        ctx.respond("agent1 also says hello")

    @forum.agent
    async def _agent2(_: MessagePromise, ctx: InteractionContext) -> None:
        ctx.respond("agent2 says hello")
        ctx.respond("agent2 says hello again")

    responses1 = _agent1.call(forum.new_message_promise("user says hello"))

    assert await represent_conversation_with_dicts(responses1) == [
        {
            "ac_model_": "message",
            "sender_alias": "USER",
            "content": "user says hello",
        },
        {
            "ac_model_": "call",
            "sender_alias": "USER",
            "content": "_agent1",
            "messages_in_request": 1,
        },
        {
            "ac_model_": "forward",
            "sender_alias": "_agent1",
            "original_msg": {
                "ac_model_": "message",
                "sender_alias": "_agent2",
                "content": "agent2 says hello",
            },
        },
        {
            "ac_model_": "forward",
            "sender_alias": "_agent1",
            "original_msg": {
                "ac_model_": "message",
                "sender_alias": "_agent2",
                "content": "agent2 says hello again",
            },
        },
        {
            "ac_model_": "message",
            "sender_alias": "_agent1",
            "content": "agent1 also says hello",
        },
    ]


async def represent_conversation_with_dicts(response: Union[MessagePromise, MessageSequence]) -> List[Dict[str, Any]]:
    """Represent the conversation as a list of dicts, omitting the hash keys and some other redundant fields."""

    def _get_msg_dict(msg: Message) -> Dict[str, Any]:
        msg_dict = msg.model_dump(exclude={"prev_msg_hash_key", "original_msg_hash_key", "msg_seq_start_hash_key"})
        if msg_dict["metadata"] == {"ac_model_": "freeform"}:
            # metadata is empty - remove it to reduce verbosity
            del msg_dict["metadata"]
        return msg_dict

    concluding_msg = response if isinstance(response, MessagePromise) else await response.aget_concluding_message()
    conversation = await concluding_msg.amaterialize_history(skip_agent_calls=False)

    conversation_dicts = []
    for idx, msg in enumerate(conversation):
        msg_dict = _get_msg_dict(msg)

        original_msg = msg.get_original_msg(return_self_if_none=False)
        if original_msg:
            msg_dict["original_msg"] = _get_msg_dict(original_msg)
            assert msg_dict["content"] == msg_dict["original_msg"]["content"]
            del msg_dict["content"]

        if isinstance(msg, AgentCall):
            messages_in_request = 0
            if msg.msg_seq_start_hash_key:
                for prev_idx in range(idx - 1, -1, -1):
                    if conversation[prev_idx].hash_key == msg.msg_seq_start_hash_key:
                        messages_in_request = idx - prev_idx
                        break
            msg_dict["messages_in_request"] = messages_in_request

        conversation_dicts.append(msg_dict)

    return conversation_dicts
