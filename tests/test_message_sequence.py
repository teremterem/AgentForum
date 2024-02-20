"""
Tests for agentforum.promises.AsyncMessageSequence
"""

# pylint: disable=protected-access
import pytest

from agentforum.conversations import ConversationTracker, HistoryTracker
from agentforum.forum import Forum
from agentforum.models import Message
from agentforum.promises import AsyncMessageSequence


@pytest.mark.asyncio
async def test_nested_message_sequences(forum: Forum) -> None:
    """
    Verify that message ordering in nested message sequences is preserved.
    """
    history_tracker = HistoryTracker(forum=forum)
    level1_sequence = AsyncMessageSequence(ConversationTracker(forum=forum), default_sender_alias="test")
    level1_producer = AsyncMessageSequence._MessageProducer(level1_sequence)
    level2_sequence = AsyncMessageSequence(ConversationTracker(forum=forum), default_sender_alias="test")
    level2_producer = AsyncMessageSequence._MessageProducer(level2_sequence)
    level3_sequence = AsyncMessageSequence(ConversationTracker(forum=forum), default_sender_alias="test")
    level3_producer = AsyncMessageSequence._MessageProducer(level3_sequence)

    with level3_producer:
        level3_producer.send_zero_or_more_messages("message 3", history_tracker)
        level3_producer.send_zero_or_more_messages("message 4", history_tracker)

    with level1_producer:
        level1_producer.send_zero_or_more_messages("message 1", history_tracker)
        level1_producer.send_zero_or_more_messages(level2_sequence, history_tracker)
        level1_producer.send_zero_or_more_messages("message 6", history_tracker)

    with level2_producer:
        level2_producer.send_zero_or_more_messages("message 2", history_tracker)
        level2_producer.send_zero_or_more_messages(level3_sequence, history_tracker)
        level2_producer.send_zero_or_more_messages("message 5", history_tracker)

    actual_messages = await level1_sequence.amaterialize_as_list()
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
    assert actual_messages[0].reply_to_msg_hash_key is None
    for msg1, msg2 in zip(actual_messages, actual_messages[1:]):
        assert msg1.hash_key == msg2.reply_to_msg_hash_key

    await level3_sequence.araise_if_error()  # no error should be raised
    await level2_sequence.araise_if_error()  # no error should be raised
    await level1_sequence.araise_if_error()  # no error should be raised


@pytest.mark.asyncio
async def test_error_in_message_sequence(forum: Forum) -> None:
    """
    Verify that an error in a message sequence comes out on the other end, but that the messages before the error
    are still processed.
    """
    history_tracker = HistoryTracker(forum=forum)
    level1_sequence = AsyncMessageSequence(ConversationTracker(forum=forum), default_sender_alias="test")
    level1_producer = AsyncMessageSequence._MessageProducer(level1_sequence)

    with level1_producer:
        level1_producer.send_zero_or_more_messages("message 1", history_tracker)
        level1_producer.send_zero_or_more_messages("message 2", history_tracker)

        try:
            raise ValueError("message 3")
        except ValueError as exc:
            level1_producer.send_zero_or_more_messages(exc, history_tracker)

        level1_producer.send_zero_or_more_messages("message 4", history_tracker)

    actual_messages = []
    async for msg in level1_sequence:
        actual_messages.append(await msg.amaterialize())
    with pytest.raises(ValueError) as exc_info:
        await level1_sequence.araise_if_error()
    assert str(exc_info.value) == "message 3"

    # assert that the messages were processed together with the error message
    assert [msg.content for msg in actual_messages] == ["message 1", "message 2", "ValueError: message 3", "message 4"]

    # assert that the error message was persisted
    persisted_error_msg = await forum.forum_trees.aretrieve_message(actual_messages[2].hash_key)
    assert persisted_error_msg.content == "ValueError: message 3"
    assert persisted_error_msg.is_error


@pytest.mark.asyncio
async def test_error_in_nested_message_sequence(forum: Forum) -> None:
    """
    Verify that an error in a NESTED message sequence comes out on the other end of the OUTER sequence, but that the
    messages before the error are still processed.
    """
    history_tracker = HistoryTracker(forum=forum)
    level1_sequence = AsyncMessageSequence(ConversationTracker(forum=forum), default_sender_alias="test")
    level1_producer = AsyncMessageSequence._MessageProducer(level1_sequence)
    level2_sequence = AsyncMessageSequence(ConversationTracker(forum=forum), default_sender_alias="test")
    level2_producer = AsyncMessageSequence._MessageProducer(level2_sequence)

    with level1_producer:
        level1_producer.send_zero_or_more_messages("message 1", history_tracker)
        level1_producer.send_zero_or_more_messages("message 2", history_tracker)
        level1_producer.send_zero_or_more_messages(level2_sequence, history_tracker)
        level1_producer.send_zero_or_more_messages("message 6", history_tracker)

    with level2_producer:
        level2_producer.send_zero_or_more_messages("message 3", history_tracker)

        try:
            raise ValueError("message 4")
        except ValueError as exc:
            level2_producer.send_zero_or_more_messages(exc, history_tracker)

        level2_producer.send_zero_or_more_messages("message 5", history_tracker)

    with pytest.raises(ValueError) as exc_info:
        await level1_sequence.araise_if_error()
    assert str(exc_info.value) == "message 4"
    actual_messages = []
    async for msg in level1_sequence:
        actual_messages.append(await msg.amaterialize())

    # assert that the messages before the error were successfully processed
    assert [msg.content for msg in actual_messages] == [
        "message 1",
        "message 2",
        "message 3",
        "ValueError: message 4",
        "message 5",
        "message 6",
    ]


@pytest.mark.asyncio
async def test_error_in_materialized_nested_sequence(forum: Forum) -> None:
    """
    Verify that an error in a MATERIALIZED NESTED message sequence comes out on the other end of the OUTER sequence.
    A separate unit test for this is necessary because in case of MATERIALIZED NESTED sequence, the error is
    propagated from Message to MessagePromise, and not from MessagePromise to MessagePromise, as in the previous test.
    """
    history_tracker = HistoryTracker(forum=forum)
    level1_sequence = AsyncMessageSequence(ConversationTracker(forum=forum), default_sender_alias="test")
    level1_producer = AsyncMessageSequence._MessageProducer(level1_sequence)
    level2_sequence = AsyncMessageSequence(ConversationTracker(forum=forum), default_sender_alias="test")
    level2_producer = AsyncMessageSequence._MessageProducer(level2_sequence)

    with level2_producer:
        try:
            raise ValueError("error message")
        except ValueError as exc:
            level2_producer.send_zero_or_more_messages(exc, history_tracker)

    with level1_producer:
        level1_producer.send_zero_or_more_messages(await level2_sequence.amaterialize_as_list(), history_tracker)

    with pytest.raises(ValueError) as exc_info:
        await level1_sequence.araise_if_error()
    assert str(exc_info.value) == "error message"
    actual_messages = []
    async for msg in level1_sequence:
        actual_messages.append(await msg.amaterialize())

    # assert that the messages before the error were successfully processed
    assert [msg.content for msg in actual_messages] == ["ValueError: error message"]


@pytest.mark.asyncio
async def test_dicts_in_message_sequences(forum: Forum) -> None:
    """
    Verify that dicts are eventually converted to Message objects when they are sent to a message sequence.
    """
    sequence = AsyncMessageSequence(ConversationTracker(forum=forum), default_sender_alias="test")
    producer = AsyncMessageSequence._MessageProducer(sequence)

    history_tracker = HistoryTracker(forum=forum)
    with producer:
        producer.send_zero_or_more_messages({"content": "message 1"}, history_tracker)
        producer.send_zero_or_more_messages(
            {"content": "message 2", "role": "some_role", "final_sender_alias": "some_alias"}, history_tracker
        )

    actual_messages = await sequence.amaterialize_as_list()
    assert len(actual_messages) == 2

    assert type(actual_messages[0]) is Message  # pylint: disable=unidiomatic-typecheck
    assert actual_messages[0].content == "message 1"
    assert not hasattr(actual_messages[0], "role")
    assert actual_messages[0].final_sender_alias == "test"
    assert actual_messages[0].prev_msg_hash_key is None

    assert type(actual_messages[1]) is Message  # pylint: disable=unidiomatic-typecheck
    assert actual_messages[1].content == "message 2"
    assert actual_messages[1].role == "some_role"
    assert actual_messages[1].final_sender_alias == "some_alias"
    assert actual_messages[1].prev_msg_hash_key == actual_messages[0].hash_key
