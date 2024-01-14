"""Test different agent collaboration scenarios."""
from typing import List, Union, Dict, Any

import pytest

from agentforum.forum import Forum, InteractionContext
from agentforum.models import Message, AgentCallMsg
from agentforum.promises import MessagePromise, AsyncMessageSequence


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
    async def _assistant(ctx: InteractionContext) -> None:
        assert await arepresent_conversation_with_dicts(ctx.request_messages) == [
            {
                "im_model_": "message",
                "sender_alias": "USER",
                "content": "set a reminder for me for tomorrow at 10am",
            },
        ]

        api_responses = _reminder_api.quick_call(ctx.request_messages)
        assert await arepresent_conversation_with_dicts(api_responses) == [
            {
                "im_model_": "message",
                "sender_alias": "USER",
                "content": "set a reminder for me for tomorrow at 10am",
            },
            {
                "im_model_": "call",
                "sender_alias": "",
                "content": "_REMINDER_API",
                "messages_in_request": 1,
            },
            {
                "im_model_": "message",
                "sender_alias": "_REMINDER_API",
                "content": "api error: invalid date format",
            },
        ]

        if (await api_responses.amaterialize_concluding_message()).content.startswith("api error:"):
            # TODO Oleksandr: implement actual ErrorMessage class
            corrections = _critic.quick_call(api_responses)

            assert await arepresent_conversation_with_dicts(corrections) == [
                {
                    "im_model_": "message",
                    "sender_alias": "USER",
                    "content": "set a reminder for me for tomorrow at 10am",
                },
                {
                    "im_model_": "call",
                    "sender_alias": "",
                    "content": "_REMINDER_API",
                    "messages_in_request": 1,
                },
                {
                    "im_model_": "message",
                    "sender_alias": "_REMINDER_API",
                    "content": "api error: invalid date format",
                },
                {
                    "im_model_": "call",
                    "sender_alias": "",
                    "content": "_CRITIC",
                    "messages_in_request": 1,
                },
                {
                    "im_model_": "message",
                    "sender_alias": "_CRITIC",
                    "content": "try swapping the month and day",
                },
            ]

            api_responses = _reminder_api.quick_call(corrections)

        assert await arepresent_conversation_with_dicts(api_responses) == [
            {
                "im_model_": "message",
                "sender_alias": "USER",
                "content": "set a reminder for me for tomorrow at 10am",
            },
            {
                "im_model_": "call",
                "sender_alias": "",
                "content": "_REMINDER_API",
                "messages_in_request": 1,
            },
            {
                "im_model_": "message",
                "sender_alias": "_REMINDER_API",
                "content": "api error: invalid date format",
            },
            {
                "im_model_": "call",
                "sender_alias": "",
                "content": "_CRITIC",
                "messages_in_request": 1,
            },
            {
                "im_model_": "message",
                "sender_alias": "_CRITIC",
                "content": "try swapping the month and day",
            },
            {
                "im_model_": "call",
                "sender_alias": "",
                "content": "_REMINDER_API",
                "messages_in_request": 1,
            },
            {
                "im_model_": "message",
                "sender_alias": "_REMINDER_API",
                "content": "success: reminder set",
            },
        ]

        ctx.respond(api_responses)

    @forum.agent
    async def _reminder_api(ctx: InteractionContext) -> None:
        if (await ctx.request_messages.amaterialize_concluding_message()).get_original_msg().sender_alias == "_CRITIC":
            ctx.respond("success: reminder set")
        else:
            ctx.respond("api error: invalid date format")

    @forum.agent
    async def _critic(ctx: InteractionContext) -> None:
        # TODO Oleksandr: turn this agent into a proxy agent
        ctx.respond("try swapping the month and day")

    assistant_responses = _assistant.quick_call("set a reminder for me for tomorrow at 10am")

    assert await arepresent_conversation_with_dicts(assistant_responses) == [
        {
            "im_model_": "message",
            "sender_alias": "USER",
            "content": "set a reminder for me for tomorrow at 10am",
        },
        {
            "im_model_": "call",
            "sender_alias": "",
            "content": "_ASSISTANT",
            "messages_in_request": 1,
        },
        {
            "im_model_": "forward",
            "sender_alias": "_ASSISTANT",
            "original_msg": {
                "im_model_": "message",
                "sender_alias": "_REMINDER_API",
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
    async def _agent1(ctx: InteractionContext) -> None:
        assert await arepresent_conversation_with_dicts(ctx.request_messages) == [
            {
                "im_model_": "message",
                "sender_alias": "USER",
                "content": "user says hello",
            },
        ]

        ctx.respond(_agent2.quick_call(ctx.request_messages))
        ctx.respond("agent1 also says hello")

    @forum.agent
    async def _agent2(ctx: InteractionContext) -> None:
        assert await arepresent_conversation_with_dicts(ctx.request_messages) == [
            {
                "im_model_": "message",
                "sender_alias": "USER",
                "content": "user says hello",
            },
        ]

        ctx.respond("agent2 says hello")
        ctx.respond("agent2 says hello again")

    responses1 = _agent1.quick_call("user says hello")

    assert await arepresent_conversation_with_dicts(responses1) == [
        {
            "im_model_": "message",
            "sender_alias": "USER",
            "content": "user says hello",
        },
        {
            "im_model_": "call",
            "sender_alias": "",
            "content": "_AGENT1",
            "messages_in_request": 1,
        },
        {
            "im_model_": "forward",
            "sender_alias": "_AGENT1",
            "original_msg": {
                "im_model_": "message",
                "sender_alias": "_AGENT2",
                "content": "agent2 says hello",
            },
        },
        {
            "im_model_": "forward",
            "sender_alias": "_AGENT1",
            "original_msg": {
                "im_model_": "message",
                "sender_alias": "_AGENT2",
                "content": "agent2 says hello again",
            },
        },
        {
            "im_model_": "message",
            "sender_alias": "_AGENT1",
            "content": "agent1 also says hello",
        },
    ]


@pytest.mark.parametrize("force_new_conversation", [True, False])
@pytest.mark.parametrize("materialize_beforehand", [True, False])
@pytest.mark.parametrize("dont_send_promises", [True, False])
@pytest.mark.asyncio
async def test_agent_force_new_conversation(
    forum: Forum, force_new_conversation: bool, materialize_beforehand: bool, dont_send_promises: bool
) -> None:
    """
    Verify that when one agent, in order to serve the user, calls another agent "behind the scenes", the conversation
    history is recorded according to the `force_new_conversation` flag (if `force_new_conversation` is True then the
    conversation history for the _agent1 starts from the response of _agent2 and not from the user's greeting).

    If agent interaction mechanism is implemented correctly then materialize_beforehand and dont_send_promises should
    not affect the conversation history.
    """

    @forum.agent
    async def _agent1(ctx: InteractionContext) -> None:
        if materialize_beforehand:
            await ctx.request_messages.amaterialize_as_list()

        request_messages = ctx.request_messages
        if dont_send_promises:
            request_messages = await request_messages.amaterialize_as_list()

        # echoing the request messages back
        ctx.respond(request_messages)

    @forum.agent
    async def _agent2(ctx: InteractionContext) -> None:
        if materialize_beforehand:
            await ctx.request_messages.amaterialize_as_list()

        ctx.respond("agent2 says hello")
        ctx.respond("agent2 says hello again")

    responses2 = _agent2.quick_call("user says hello")
    if dont_send_promises:
        responses2 = await responses2.amaterialize_as_list()

    responses1 = _agent1.quick_call(responses2, force_new_conversation=force_new_conversation)

    if force_new_conversation:
        assert await arepresent_conversation_with_dicts(responses1) == [
            {
                "im_model_": "forward",
                "sender_alias": "USER",
                "original_msg": {
                    "im_model_": "message",
                    "sender_alias": "_AGENT2",
                    "content": "agent2 says hello",
                },
            },
            {
                "im_model_": "forward",
                "sender_alias": "USER",
                "original_msg": {
                    "im_model_": "message",
                    "sender_alias": "_AGENT2",
                    "content": "agent2 says hello again",
                },
            },
            {
                "im_model_": "call",
                "sender_alias": "",
                "content": "_AGENT1",
                "messages_in_request": 2,
            },
            {
                "im_model_": "forward",
                "sender_alias": "_AGENT1",
                "original_msg": {
                    "im_model_": "forward",
                    "sender_alias": "USER",
                    "original_msg": {
                        "im_model_": "message",
                        "sender_alias": "_AGENT2",
                        "content": "agent2 says hello",
                    },
                },
            },
            {
                "im_model_": "forward",
                "sender_alias": "_AGENT1",
                "original_msg": {
                    "im_model_": "forward",
                    "sender_alias": "USER",
                    "original_msg": {
                        "im_model_": "message",
                        "sender_alias": "_AGENT2",
                        "content": "agent2 says hello again",
                    },
                },
            },
        ]
    else:
        assert await arepresent_conversation_with_dicts(responses1) == [
            {
                "im_model_": "message",
                "sender_alias": "USER",
                "content": "user says hello",
            },
            {
                "im_model_": "call",
                "sender_alias": "",
                "content": "_AGENT2",
                "messages_in_request": 1,
            },
            {
                "im_model_": "message",
                "sender_alias": "_AGENT2",
                "content": "agent2 says hello",
            },
            {
                "im_model_": "message",
                "sender_alias": "_AGENT2",
                "content": "agent2 says hello again",
            },
            {
                "im_model_": "call",
                "sender_alias": "",
                "content": "_AGENT1",
                "messages_in_request": 2,
            },
            {
                "im_model_": "forward",
                "sender_alias": "_AGENT1",
                "original_msg": {
                    "im_model_": "message",
                    "sender_alias": "_AGENT2",
                    "content": "agent2 says hello",
                },
            },
            {
                "im_model_": "forward",
                "sender_alias": "_AGENT1",
                "original_msg": {
                    "im_model_": "message",
                    "sender_alias": "_AGENT2",
                    "content": "agent2 says hello again",
                },
            },
        ]


async def arepresent_conversation_with_dicts(
    response: Union[MessagePromise, AsyncMessageSequence]
) -> List[Dict[str, Any]]:
    """Represent the conversation as a list of dicts, omitting the hash keys and some other redundant fields."""

    def _get_msg_dict(msg_: Message) -> Dict[str, Any]:
        msg_dict_ = msg_.model_dump(exclude={"prev_msg_hash_key", "original_msg_hash_key", "msg_seq_start_hash_key"})
        if msg_dict_.get("function_kwargs") == {}:
            # function_kwargs exists, and it is empty - remove it to reduce verbosity
            del msg_dict_["function_kwargs"]

        original_msg_ = msg_.get_original_msg(return_self_if_none=False)
        if original_msg_:
            assert msg_dict_["content"] == original_msg_.content
            msg_dict_["original_msg"] = _get_msg_dict(original_msg_)
            del msg_dict_["content"]
        return msg_dict_

    concluding_msg = response if isinstance(response, MessagePromise) else await response.aget_concluding_msg_promise()
    conversation = await concluding_msg.amaterialize_full_history(skip_agent_calls=False)

    conversation_dicts = []
    for idx, msg in enumerate(conversation):
        msg_dict = _get_msg_dict(msg)

        if isinstance(msg, AgentCallMsg):
            messages_in_request = 0
            if msg.msg_seq_start_hash_key:
                for prev_idx in range(idx - 1, -1, -1):
                    if conversation[prev_idx].hash_key == msg.msg_seq_start_hash_key:
                        messages_in_request = idx - prev_idx
                        break
            msg_dict["messages_in_request"] = messages_in_request

        conversation_dicts.append(msg_dict)

    return conversation_dicts
