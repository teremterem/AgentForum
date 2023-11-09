# pylint: disable=protected-access
"""Tests for agentcache.promises.MessageSequence"""
import pytest

from agentcache.forum import Forum
from agentcache.promises import MessageSequence


@pytest.mark.asyncio
async def test_nested_message_sequences(forum: Forum) -> None:
    """Verify that message ordering in nested message sequences is preserved."""
    outer_sequence = MessageSequence(forum)
    outer_producer = MessageSequence._MessageProducer(outer_sequence)
    inner_sequence = MessageSequence(forum)
    inner_producer = MessageSequence._MessageProducer(inner_sequence)

    with outer_producer:
        outer_producer.send_msg("outer message 1")
        outer_producer.send_msg(inner_sequence)
        outer_producer.send_msg("outer message 4")

    with inner_producer:
        inner_producer.send_msg("inner message 2")
        # TODO Oleksandr: try the third level of nesting
        inner_producer.send_msg("inner message 3")

    actual_sequence = [msg.content for msg in await outer_sequence.amaterialize_all()]
    assert actual_sequence == [
        "outer message 1",
        "inner message 2",
        "inner message 3",
        "outer message 4",
    ]


# TODO Oleksandr: test sending exceptions into sequences
