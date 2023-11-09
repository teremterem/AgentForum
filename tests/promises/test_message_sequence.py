# pylint: disable=protected-access
"""Tests for agentcache.promises.MessageSequence"""
import pytest

from agentcache.forum import Forum
from agentcache.promises import MessageSequence


@pytest.mark.asyncio
async def test_nested_message_sequences(forum: Forum) -> None:
    """Verify that message ordering in nested message sequences is preserved."""
    level1_sequence = MessageSequence(forum)
    level1_producer = MessageSequence._MessageProducer(level1_sequence)
    level2_sequence = MessageSequence(forum)
    level2_producer = MessageSequence._MessageProducer(level2_sequence)
    level3_sequence = MessageSequence(forum)
    level3_producer = MessageSequence._MessageProducer(level3_sequence)

    with level3_producer:
        level3_producer.send_msg("message 3")
        level3_producer.send_msg("message 4")

    with level1_producer:
        level1_producer.send_msg("message 1")
        level1_producer.send_msg(level2_sequence)
        level1_producer.send_msg("message 6")

    with level2_producer:
        level2_producer.send_msg("message 2")
        level2_producer.send_msg(level3_sequence)
        level2_producer.send_msg("message 5")

    actual_messages = await level1_sequence.amaterialize_all()
    actual_texts = [msg.content for msg in actual_messages]
    assert actual_texts == [
        "message 1",
        "message 2",
        "message 3",
        "message 4",
        "message 5",
        "message 6",
    ]
    # assert that each message in the sequence is linked to the previous one
    assert actual_messages[0].prev_msg_hash_key is None
    for msg1, msg2 in zip(actual_messages, actual_messages[1:]):
        assert msg1.hash_key == msg2.prev_msg_hash_key


# TODO Oleksandr: test sending exceptions into sequences
