# pylint: disable=unused-argument
"""
Tests for agentforum.utils module.
"""
import pytest

from agentforum.forum import Forum, ConversationTracker, InteractionContext
from agentforum.promises import AsyncMessageSequence
from agentforum.utils import arender_conversation


@pytest.mark.asyncio
async def test_arender_conversation_default_alias_resolver(
    forum: Forum, fake_interaction_context: InteractionContext
) -> None:
    """
    Test arender_conversation() with the default alias_resolver.
    """
    sequence = AsyncMessageSequence(ConversationTracker(forum=forum), default_sender_alias="TEST_ALIAS")
    producer = AsyncMessageSequence._MessageProducer(sequence)  # pylint: disable=protected-access

    with producer:
        producer.send_zero_or_more_messages("message 1")
        producer.send_zero_or_more_messages("message 2", "OVERRIDDEN_ALIAS")

    rendered_conversation = await arender_conversation(sequence)
    assert rendered_conversation == "TEST_ALIAS: message 1\n\nOVERRIDDEN_ALIAS: message 2"
