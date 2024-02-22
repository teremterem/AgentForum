"""Test different agent collaboration scenarios."""

from typing import Union, Any

import pytest

from agentforum.forum import Forum, InteractionContext
from agentforum.models import Message, AgentCallMsg
from agentforum.promises import MessagePromise, AsyncMessageSequence
from agentforum.utils import NO_VALUE


@pytest.mark.asyncio
async def test_assistant_googler_browser_scenario(forum: Forum) -> None:
    """
    Verify the message history in the following scenario:
    - The `user` asks the `assistant` a question.
    - The `assistant` asks the `googler` to find the answer.
    - The `googler` gives the `browser` search results that it found.
    - The `browser` asks the `browser` (asks itself) to navigate to a url from search results.
    - The `browser` (on behalf of the `assistant`) responds to the `user` with the answer that it found.
    """

    @forum.agent
    async def browser(ctx: InteractionContext, emulate_answer_found: bool = False) -> None:
        if emulate_answer_found:
            ctx.respond("Quite a long distance.")
        else:
            ctx.this_agent.tell("Follow this url.", emulate_answer_found=True)

    @forum.agent
    async def googler(_: InteractionContext) -> None:
        browser.tell("Here are some search results I found.")

    @forum.agent
    async def assistant(ctx: InteractionContext) -> None:
        # `branch_from=NO_VALUE` prevents forwarding of the same request messages into the same history branch
        # TODO TODO TODO Oleksandr: this trick with `branch_from=NO_VALUE` is not intuitive, what to do about it ?
        googler.tell(ctx.request_messages, branch_from=NO_VALUE)

    assistant_responses = assistant.ask(
        [
            "What's the distance between the Earth and the Moon?!",
            "Tell me now!",
        ]
    )

    # print()
    # print()
    # for msg in await assistant_responses.amaterialize_full_history():
    #     print(msg.hash_key)
    #     pprint(msg.as_dict())
    #     while msg := msg.get_original_msg(return_self_if_none=False):
    #         print(msg.hash_key)
    #         pprint(msg.as_dict())
    #     print()
    #     print()
    # assert False

    assert await arepresent_history_with_dicts(assistant_responses) == [
        {
            "im_model_": "message",
            "final_sender_alias": "USER",
            "content": "What's the distance between the Earth and the Moon?!",
        },
        {
            "reply_to": "What's the distance between the Earth and the Moon?!",
            "im_model_": "message",
            "final_sender_alias": "USER",
            "content": "Tell me now!",
        },
        {
            "im_model_": "message",
            "final_sender_alias": "GOOGLER",
            "content": "Here are some search results I found.",
        },
        {
            "im_model_": "message",
            "final_sender_alias": "BROWSER",
            "content": "Follow this url.",
        },
        {
            "reply_to": "Tell me now!",
            "im_model_": "message",
            "final_sender_alias": "ASSISTANT",
            "content": "Quite a long distance.",
        },
    ]


# @pytest.mark.skip
@pytest.mark.asyncio
async def test_two_nested_agents(forum: Forum) -> None:
    """
    Verify that when one agent, in order to serve the user, calls another agent "behind the scenes", the conversation
    history is recorded correctly.
    """

    @forum.agent
    async def agent1(ctx: InteractionContext) -> None:
        assert await arepresent_history_with_dicts(ctx.request_messages) == [
            {
                "im_model_": "message",
                "final_sender_alias": "USER",
                "content": "user says hello",
            },
        ]

        ctx.respond(agent2.ask(ctx.request_messages))
        ctx.respond("agent1 also says hello")

    @forum.agent
    async def agent2(ctx: InteractionContext) -> None:
        assert await arepresent_history_with_dicts(ctx.request_messages) == [
            {
                "im_model_": "message",
                "final_sender_alias": "USER",
                "content": "user says hello",
            },
            {
                "before_forward": {
                    "im_model_": "message",
                    "final_sender_alias": "USER",
                    "content": "user says hello",
                },
                "im_model_": "forward",
                "final_sender_alias": "AGENT1",
            },
        ]

        ctx.respond("agent2 says hello")
        ctx.respond("agent2 says hello again")

    responses1 = agent1.ask("user says hello")

    assert await arepresent_history_with_dicts(responses1) == [
        {
            "im_model_": "message",
            "final_sender_alias": "USER",
            "content": "user says hello",
        },
        {
            "reply_to": "user says hello",
            "im_model_": "forward",
            "final_sender_alias": "AGENT1",
            "before_forward": {
                "reply_to": "user says hello",
                "im_model_": "message",
                "final_sender_alias": "AGENT2",
                "content": "agent2 says hello",
            },
        },
        {
            "reply_to": "agent2 says hello",
            "im_model_": "forward",
            "final_sender_alias": "AGENT1",
            "before_forward": {
                "reply_to": "agent2 says hello",
                "im_model_": "message",
                "final_sender_alias": "AGENT2",
                "content": "agent2 says hello again",
            },
        },
        {
            "reply_to": "agent2 says hello again",
            "im_model_": "message",
            "final_sender_alias": "AGENT1",
            "content": "agent1 also says hello",
        },
    ]


@pytest.mark.skip  # TODO TODO TODO TODO TODO
@pytest.mark.parametrize("blank_history", [True, False])
@pytest.mark.parametrize("materialize_beforehand", [True, False])
@pytest.mark.parametrize("dont_send_promises", [True, False])
@pytest.mark.asyncio
async def test_agent_new_conversation(
    forum: Forum, blank_history: bool, materialize_beforehand: bool, dont_send_promises: bool
) -> None:
    """
    Verify that when one agent, in order to serve the user, calls another agent "behind the scenes", the conversation
    history is recorded according to the `new_conversation` flag (if `new_conversation` is True then the
    conversation history for the _agent1 starts from the response of _agent2 and not from the user's greeting).

    If agent interaction mechanism is implemented correctly then materialize_beforehand and dont_send_promises should
    not affect the conversation history.
    """

    @forum.agent
    async def agent1(ctx: InteractionContext) -> None:
        if materialize_beforehand:
            await ctx.request_messages.amaterialize_as_list()

        request_messages = ctx.request_messages
        if dont_send_promises:
            request_messages = await request_messages.amaterialize_as_list()

        # echoing the request messages back
        ctx.respond(request_messages)

    @forum.agent
    async def agent2(ctx: InteractionContext) -> None:
        if materialize_beforehand:
            await ctx.request_messages.amaterialize_as_list()

        ctx.respond("agent2 says hello")
        ctx.respond("agent2 says hello again")

    responses2 = agent2.ask("user says hello")
    if dont_send_promises:
        responses2 = await responses2.amaterialize_as_list()

    responses1 = agent1.ask(responses2, blank_history=blank_history)

    if blank_history:
        assert await arepresent_history_with_dicts(responses1) == [
            {
                "im_model_": "forward",
                "final_sender_alias": "USER",
                "before_forward": {
                    "im_model_": "message",
                    "final_sender_alias": "AGENT2",
                    "content": "agent2 says hello",
                },
            },
            {
                "im_model_": "forward",
                "final_sender_alias": "USER",
                "before_forward": {
                    "im_model_": "message",
                    "final_sender_alias": "AGENT2",
                    "content": "agent2 says hello again",
                },
            },
            {
                "im_model_": "call",
                "final_sender_alias": "SYSTEM",
                "receiver_alias": "AGENT1",
                "messages_in_request": 2,
            },
            {
                "im_model_": "forward",
                "final_sender_alias": "AGENT1",
                "before_forward": {
                    "im_model_": "forward",
                    "final_sender_alias": "USER",
                    "before_forward": {
                        "im_model_": "message",
                        "final_sender_alias": "AGENT2",
                        "content": "agent2 says hello",
                    },
                },
            },
            {
                "im_model_": "forward",
                "final_sender_alias": "AGENT1",
                "before_forward": {
                    "im_model_": "forward",
                    "final_sender_alias": "USER",
                    "before_forward": {
                        "im_model_": "message",
                        "final_sender_alias": "AGENT2",
                        "content": "agent2 says hello again",
                    },
                },
            },
        ]
    else:
        assert await arepresent_history_with_dicts(responses1) == [
            {
                "im_model_": "message",
                "final_sender_alias": "USER",
                "content": "user says hello",
            },
            {
                "im_model_": "call",
                "final_sender_alias": "SYSTEM",
                "receiver_alias": "AGENT2",
                "messages_in_request": 1,
            },
            {
                "im_model_": "message",
                "final_sender_alias": "AGENT2",
                "content": "agent2 says hello",
            },
            {
                "im_model_": "message",
                "final_sender_alias": "AGENT2",
                "content": "agent2 says hello again",
            },
            {
                "im_model_": "call",
                "final_sender_alias": "SYSTEM",
                "receiver_alias": "AGENT1",
                "messages_in_request": 2,
            },
            {
                "im_model_": "forward",
                "final_sender_alias": "AGENT1",
                "before_forward": {
                    "im_model_": "message",
                    "final_sender_alias": "AGENT2",
                    "content": "agent2 says hello",
                },
            },
            {
                "im_model_": "forward",
                "final_sender_alias": "AGENT1",
                "before_forward": {
                    "im_model_": "message",
                    "final_sender_alias": "AGENT2",
                    "content": "agent2 says hello again",
                },
            },
        ]


async def arepresent_history_with_dicts(
    response: Union[MessagePromise, AsyncMessageSequence], follow_replies: bool = False
) -> list[dict[str, Any]]:
    """Represent the conversation as a list of dicts, omitting the hash keys and some other redundant fields."""

    async def _get_msg_dict(msg_: Message) -> dict[str, Any]:
        msg_dict_ = msg_.model_dump(
            exclude={
                "forum_trees",
                "msg_before_forward_hash_key",
                "msg_seq_start_hash_key",
                "prev_msg_hash_key",
                "reply_to_msg_hash_key",
            }
        )
        if msg_dict_.get("function_kwargs") == {}:
            del msg_dict_["function_kwargs"]
        if msg_dict_.get("is_error") is False:
            del msg_dict_["is_error"]
        if msg_dict_.get("is_detached") is False:
            del msg_dict_["is_detached"]
        if "content" in msg_dict_ and msg_dict_.get("content") is None:
            del msg_dict_["content"]
        if "content_template" in msg_dict_ and msg_dict_.get("content_template") is None:
            del msg_dict_["content_template"]

        before_forward_ = msg_.get_before_forward(return_self_if_none=False)
        if before_forward_:
            assert msg_dict_["content"] == before_forward_.content
            msg_dict_["before_forward"] = await _get_msg_dict(before_forward_)
            del msg_dict_["content"]
        reply_to_ = await msg_.aget_reply_to_msg()
        if reply_to_:
            msg_dict_["reply_to"] = reply_to_.content
        return msg_dict_

    concluding_msg = response if isinstance(response, MessagePromise) else await response.aget_concluding_msg_promise()
    conversation = await concluding_msg.amaterialize_full_history(follow_replies=follow_replies)

    conversation_dicts = []
    for idx, msg in enumerate(conversation):
        msg_dict = await _get_msg_dict(msg)

        if isinstance(msg, AgentCallMsg):
            messages_in_request = 0
            assert msg.msg_seq_start_hash_key  # this should always be set for AgentCallMsg
            for prev_idx in range(idx - 1, -1, -1):
                if conversation[prev_idx].hash_key == msg.msg_seq_start_hash_key:
                    messages_in_request = idx - prev_idx
                    break
            msg_dict["messages_in_request"] = messages_in_request

        conversation_dicts.append(msg_dict)

    return conversation_dicts
