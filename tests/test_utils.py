"""
Tests for agentforum.utils module.
"""
# pylint: disable=protected-access
import pytest

from agentforum.forum import Forum, ConversationTracker
from agentforum.models import Message
from agentforum.promises import AsyncMessageSequence


@pytest.mark.asyncio
async def test_arender_conversation_default_sender_resolver(forum: Forum) -> None:
    """TODO TODO TODO"""
    sequence = AsyncMessageSequence(ConversationTracker(forum=forum), default_sender_alias="test")
    producer = AsyncMessageSequence._MessageProducer(sequence)

    with producer:
        producer.send_zero_or_more_messages({"content": "message 1"})
        producer.send_zero_or_more_messages(
            {"content": "message 2", "role": "some_role", "sender_alias": "some_alias"}
        )

    actual_messages = await sequence.amaterialize_as_list()
    assert len(actual_messages) == 2

    assert type(actual_messages[0]) is Message  # pylint: disable=unidiomatic-typecheck
    assert actual_messages[0].content == "message 1"
    assert not hasattr(actual_messages[0], "role")
    assert actual_messages[0].sender_alias == "test"
    assert actual_messages[0].prev_msg_hash_key is None

    assert type(actual_messages[1]) is Message  # pylint: disable=unidiomatic-typecheck
    assert actual_messages[1].content == "message 2"
    assert actual_messages[1].role == "some_role"
    assert actual_messages[1].sender_alias == "some_alias"
    assert actual_messages[1].prev_msg_hash_key == actual_messages[0].hash_key
