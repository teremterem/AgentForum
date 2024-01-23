# pylint: disable=protected-access,unused-argument,redefined-outer-name
"""
Tests for the agentforum.utils module.
"""
from typing import Awaitable

import pytest

from agentforum.forum import Forum, ConversationTracker, InteractionContext
from agentforum.promises import AsyncMessageSequence
from agentforum.utils import arender_conversation


@pytest.fixture
async def athree_message_sequence(forum: Forum, fake_interaction_context: InteractionContext) -> AsyncMessageSequence:
    """
    A sequence of three messages.
    """
    sequence = AsyncMessageSequence(ConversationTracker(forum=forum), default_sender_alias="TEST_ALIAS")
    # noinspection PyProtectedMember
    producer = AsyncMessageSequence._MessageProducer(sequence)

    with producer:
        producer.send_zero_or_more_messages("message 1")
        producer.send_zero_or_more_messages("message 2", sender_alias="OVERRIDDEN_ALIAS")
        producer.send_zero_or_more_messages("message 3")

    return sequence


@pytest.mark.asyncio
async def test_arender_conversation_default_alias_resolver(
    athree_message_sequence: Awaitable[AsyncMessageSequence],
) -> None:
    """
    Test arender_conversation() with the default alias_resolver.
    """
    rendered_conversation = await arender_conversation(await athree_message_sequence)
    assert rendered_conversation == "TEST_ALIAS: message 1\n\nOVERRIDDEN_ALIAS: message 2\n\nTEST_ALIAS: message 3"


@pytest.mark.asyncio
async def test_arender_conversation_hardcoded_alias(athree_message_sequence: Awaitable[AsyncMessageSequence]) -> None:
    """
    Test arender_conversation() with a hardcoded alias.
    """
    rendered_conversation = await arender_conversation(await athree_message_sequence, alias_resolver="HARDCODED")
    assert rendered_conversation == "HARDCODED: message 1\n\nHARDCODED: message 2\n\nHARDCODED: message 3"


@pytest.mark.asyncio
async def test_arender_conversation_custom_alias_resolver(
    athree_message_sequence: Awaitable[AsyncMessageSequence],
) -> None:
    """
    Test arender_conversation() with a custom alias_resolver function.
    """
    rendered_conversation = await arender_conversation(
        await athree_message_sequence,
        alias_resolver=lambda msg: (
            None
            if msg.content == "message 1"  # don't render the first message
            else f"{''.join([word.capitalize() for word in msg.sender_alias.split('_')])}Bot"
        ),
    )
    assert rendered_conversation == "OverriddenAliasBot: message 2\n\nTestAliasBot: message 3"


@pytest.mark.asyncio
async def test_arender_conversation_with_dicts(
    athree_message_sequence: Awaitable[AsyncMessageSequence],
) -> None:
    """
    Test arender_conversation() when a list of regular dicts are passed together with AsyncMessageSequence.
    """
    sequence = [
        await athree_message_sequence,
        {
            "content": "message 4",
        },
        {
            "content": "message 5",
            "sender_alias": "ONE_MORE_SENDER_ALIAS",
        },
    ]
    rendered_conversation = await arender_conversation(sequence)
    assert rendered_conversation == (
        "TEST_ALIAS: message 1\n"
        "\n"
        "OVERRIDDEN_ALIAS: message 2\n"
        "\n"
        "TEST_ALIAS: message 3\n"
        "\n"
        "FAKE_AGENT: message 4\n"
        "\n"
        "ONE_MORE_SENDER_ALIAS: message 5"
    )
