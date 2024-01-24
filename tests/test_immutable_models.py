# pylint: disable=redefined-outer-name
"""Tests for the Immutable models."""
import hashlib
from typing import Literal, Optional, Awaitable
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agentforum.forum import Forum
from agentforum.models import Immutable, Freeform, Message, ForwardedMessage, AgentCallMsg


class SampleImmutable(Immutable):
    """A sample Immutable subclass."""

    some_req_field: str
    some_opt_field: int = 2
    sub_immutable: Optional["SampleImmutable"] = None
    im_model_: Literal["sample_immutable"] = "sample_immutable"


def test_immutable_frozen() -> None:
    """Test that the `Immutable` class is frozen."""

    sample = SampleImmutable(some_req_field="test")

    with pytest.raises(ValidationError):
        sample.some_req_field = "test2"
    with pytest.raises(ValidationError):
        sample.some_opt_field = 3

    assert sample.some_req_field == "test"
    assert sample.some_opt_field == 2


def test_message_frozen(forum: Forum) -> None:
    """Test that the `Message` class is frozen."""
    message = Message(forum_trees=forum.forum_trees, content="test", sender_alias="user")

    with pytest.raises(ValidationError):
        message.content = "test2"

    assert message.content == "test"


def test_freeform_frozen() -> None:
    """Test that the `Freeform` class is frozen."""
    freeform = Freeform(some_field="some value")

    with pytest.raises(ValidationError):
        freeform.content = "some other value"

    assert freeform.some_field == "some value"


def test_immutable_hash_key() -> None:
    """Test the `Immutable.hash_key` property."""
    sample = SampleImmutable(
        some_req_field="test", sub_immutable=SampleImmutable(some_req_field="юнікод", some_opt_field=3)
    )

    # print(json.dumps(sample.model_dump(), ensure_ascii=False, sort_keys=True))
    expected_hash_key = hashlib.sha256(
        '{"im_model_": "sample_immutable", "some_opt_field": 2, "some_req_field": "test", "sub_immutable": '
        '{"im_model_": "sample_immutable", "some_opt_field": 3, "some_req_field": "юнікод", '
        '"sub_immutable": null}}'.encode("utf-8")
    ).hexdigest()
    assert sample.hash_key == expected_hash_key


def test_message_hash_key(forum: Forum) -> None:
    """Test the `Message.hash_key` property."""
    message = Message(
        forum_trees=forum.forum_trees, content="test", sender_alias="user", custom_field={"role": "user"}
    )
    # print(json.dumps(message.model_dump(), ensure_ascii=False, sort_keys=True))
    expected_hash_key = hashlib.sha256(
        '{"content": "test", "custom_field": {"role": "user"}, "im_model_": "message", '
        '"prev_msg_hash_key": null, "sender_alias": "user"}'.encode("utf-8")
    ).hexdigest()
    assert message.hash_key == expected_hash_key

    message = Message(forum_trees=forum.forum_trees, content="test", sender_alias="user")
    # print(json.dumps(message.model_dump(), ensure_ascii=False, sort_keys=True))
    expected_hash_key = hashlib.sha256(
        '{"content": "test", "im_model_": "message", '
        '"prev_msg_hash_key": null, "sender_alias": "user"}'.encode("utf-8")
    ).hexdigest()
    assert message.hash_key == expected_hash_key


def test_forwarded_message_hash_key(forum: Forum) -> None:
    """
    Assert that ForwardedMessage._original_msg is not serialized when calculating the hash_key of a ForwardedMessage
    (only ForwardedMessage.original_msg_hash_key is).
    """
    original_msg = Message(
        forum_trees=forum.forum_trees, content="message that is being forwarded", sender_alias="user"
    )

    message = ForwardedMessage(
        forum_trees=forum.forum_trees,
        content="test",
        sender_alias="user",
        msg_before_forward_hash_key=original_msg.hash_key,
    )
    message._original_msg = original_msg  # pylint: disable=protected-access

    # print(json.dumps(message.model_dump(), ensure_ascii=False, sort_keys=True))
    expected_hash_key = hashlib.sha256(
        '{"content": "test", "im_model_": "forward", '
        '"msg_before_forward_hash_key": "f2487bd3261d29745e4c47ae8f0256845a7eae939b437a5409258310486cd80a", '
        '"prev_msg_hash_key": null, "sender_alias": "user"}'.encode("utf-8")
    ).hexdigest()
    assert message.hash_key == expected_hash_key


def test_nested_object_not_copied() -> None:
    """Test that nested objects are not copied when the outer pydantic model is created."""
    sub_immutable = SampleImmutable(some_req_field="test")
    sample = SampleImmutable(some_req_field="test", sub_immutable=sub_immutable)

    assert sample.sub_immutable is sub_immutable


def test_nested_message_freeform_not_copied(forum: Forum) -> None:
    """Test that Freeform nested in Message is not copied."""
    # TODO Oleksandr: not sure if we still need this test - there are cases when nested objects have to be copied
    custom_field = Freeform(role="assistant", blah={"blah": "blah"})
    message = Message(forum_trees=forum.forum_trees, content="test", sender_alias="user", custom_field=custom_field)

    assert message.custom_field is custom_field


def test_immutable_hash_key_calculated_once() -> None:
    """
    Test that the `Immutable.hash_key` property is calculated only once and all subsequent calls return the same
    value without calculating it again.
    """
    original_sha256 = hashlib.sha256

    with patch("hashlib.sha256", side_effect=original_sha256) as mock_sha256:
        sample = SampleImmutable(some_req_field="test")
        mock_sha256.assert_not_called()  # not calculated yet

        assert sample.hash_key == "c9fb7f94ae479c289eeb08f071642f3a43332a33b6d152b703f22e8e9764b5fa"
        mock_sha256.assert_called_once()  # calculated once

        assert sample.hash_key == "c9fb7f94ae479c289eeb08f071642f3a43332a33b6d152b703f22e8e9764b5fa"
        mock_sha256.assert_called_once()  # check that it wasn't calculated again


def test_freeform_hash_key_vs_key_ordering() -> None:
    """
    Test that hash_key of Freeform (a Pydantic model that allows arbitrary metadata fields and is a parent class for
    Message) is not affected by the ordering of its fields.
    """
    freeform1 = Freeform(some_req_field="test", some_opt_field=2)
    freeform2 = Freeform(some_opt_field=2, some_req_field="test")

    assert freeform1.hash_key == freeform2.hash_key


def test_message_metadata_as_dict(forum: Forum) -> None:
    """
    Test that the `Message.metadata_as_dict` method returns only the custom fields as a dict.
    """
    message = Message(
        forum_trees=forum.forum_trees, content="test", sender_alias="user", custom_field={"role": "user"}
    )

    assert isinstance(message.custom_field, Freeform)  # make sure it wasn't stored as plain dict
    assert message.metadata_as_dict == {"custom_field": {"role": "user"}}


@pytest.fixture
async def amessage_on_branch(forum: Forum) -> Message:
    """A message on a branch."""
    message = Message(forum_trees=forum.forum_trees, content="message 1", sender_alias="user")
    await forum.forum_trees.astore_immutable(message)
    message = AgentCallMsg(
        forum_trees=forum.forum_trees,
        content="call 1",
        sender_alias="",
        prev_msg_hash_key=message.hash_key,
    )
    await forum.forum_trees.astore_immutable(message)
    message = AgentCallMsg(
        forum_trees=forum.forum_trees,
        content="call 2",
        sender_alias="",
        prev_msg_hash_key=message.hash_key,
    )
    await forum.forum_trees.astore_immutable(message)
    message = Message(
        forum_trees=forum.forum_trees,
        content="message 2",
        sender_alias="user",
        prev_msg_hash_key=message.hash_key,
    )
    await forum.forum_trees.astore_immutable(message)

    return message


@pytest.mark.asyncio
async def test_message_aget_previous_msg(amessage_on_branch: Awaitable[Message]) -> None:
    """
    Test that the `Message.aget_previous_msg` method returns the previous message if it exists.
    """
    message = await amessage_on_branch
    assert message.content == "message 2"  # a sanity check
    previous_message = await message.aget_previous_msg()

    # all the agent call messages were skipped by default
    assert previous_message.content == "message 1"
    assert type(previous_message) is Message  # pylint: disable=unidiomatic-typecheck

    # no more previous messages
    assert await previous_message.aget_previous_msg() is None


@pytest.mark.asyncio
async def test_message_aget_previous_msg_dont_skip_calls(amessage_on_branch: Awaitable[Message]) -> None:
    """
    Test that the `Message.aget_previous_msg` method returns the previous message if it exists, even if it's an agent
    call message.
    """
    message = await amessage_on_branch
    assert message.content == "message 2"  # a sanity check
    previous_message = await message.aget_previous_msg(skip_agent_calls=False)

    # agent calls were NOT skipped
    assert previous_message.content == "call 2"
    assert type(previous_message) is AgentCallMsg  # pylint: disable=unidiomatic-typecheck

    # more previous messages exist
    assert await previous_message.aget_previous_msg() is not None
