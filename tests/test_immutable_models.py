# pylint: disable=redefined-outer-name
"""Tests for the Immutable models."""
import hashlib
from typing import Literal, Optional
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agentforum.forum import InteractionContext
from agentforum.models import Immutable, Freeform, Message, ForwardedMessage


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


def test_message_frozen(fake_interaction_context: InteractionContext) -> None:
    """Test that the `Message` class is frozen."""
    message = Message(
        forum_trees=fake_interaction_context.forum_trees, content="test", final_sender_alias="user", is_detached=False
    )

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


def test_message_hash_key(fake_interaction_context: InteractionContext) -> None:
    """Test the `Message.hash_key` property."""
    message = Message(
        forum_trees=fake_interaction_context.forum_trees,
        content="test",
        final_sender_alias="user",
        custom_field={"role": "user"},
        is_detached=False,
    )
    # print(json.dumps(message.model_dump(exclude={"forum_trees"}), ensure_ascii=False, sort_keys=True))
    expected_hash_key = hashlib.sha256(
        '{"content": "test", "content_template": null, "custom_field": {"role": "user"}, '
        '"final_sender_alias": "user", "im_model_": "message", "is_error": false, "prev_msg_hash_key": null, '
        '"reply_to_msg_hash_key": null}'.encode("utf-8")
    ).hexdigest()
    assert message.hash_key == expected_hash_key

    message = Message(
        forum_trees=fake_interaction_context.forum_trees, content="test", final_sender_alias="user", is_detached=False
    )
    # print(json.dumps(message.model_dump(exclude={"forum_trees"}), ensure_ascii=False, sort_keys=True))
    expected_hash_key = hashlib.sha256(
        '{"content": "test", "content_template": null, "final_sender_alias": "user", "im_model_": "message", '
        '"is_error": false, "prev_msg_hash_key": null, "reply_to_msg_hash_key": null}'.encode("utf-8")
    ).hexdigest()
    assert message.hash_key == expected_hash_key


def test_forwarded_message_hash_key(fake_interaction_context: InteractionContext) -> None:
    """
    Assert that ForwardedMessage._original_msg is not serialized when calculating the hash_key of a ForwardedMessage
    (only ForwardedMessage.original_msg_hash_key is).
    """
    original_msg = Message(
        forum_trees=fake_interaction_context.forum_trees,
        content="message that is being forwarded",
        final_sender_alias="user",
        is_detached=False,
    )

    message = ForwardedMessage(
        forum_trees=fake_interaction_context.forum_trees,
        final_sender_alias="user",
        msg_before_forward_hash_key=original_msg.hash_key,
    )

    # print(json.dumps(original_msg.model_dump(exclude={"forum_trees"}), ensure_ascii=False, sort_keys=True))
    expected_original_hash_key = hashlib.sha256(
        '{"content": "message that is being forwarded", "content_template": null, "final_sender_alias": "user", '
        '"im_model_": "message", "is_error": false, "prev_msg_hash_key": null, "reply_to_msg_hash_key": null}'
        "".encode("utf-8")
    ).hexdigest()
    # print(json.dumps(message.model_dump(exclude={"forum_trees"}), ensure_ascii=False, sort_keys=True))
    expected_hash_key = hashlib.sha256(
        '{"final_sender_alias": "user", "im_model_": "forward", "is_error": false, '
        f'"msg_before_forward_hash_key": "{expected_original_hash_key}", "prev_msg_hash_key": null, '
        '"reply_to_msg_hash_key": null}'.encode("utf-8")
    ).hexdigest()
    assert message.hash_key == expected_hash_key


def test_nested_object_not_copied() -> None:
    """Test that nested objects are not copied when the outer pydantic model is created."""
    sub_immutable = SampleImmutable(some_req_field="test")
    sample = SampleImmutable(some_req_field="test", sub_immutable=sub_immutable)

    assert sample.sub_immutable is sub_immutable


def test_nested_message_freeform_not_copied(fake_interaction_context: InteractionContext) -> None:
    """Test that Freeform nested in Message is not copied."""
    # TODO Oleksandr: not sure if we still need this test - there are cases when nested objects have to be copied
    custom_field = Freeform(role="assistant", blah={"blah": "blah"})
    message = Message(
        forum_trees=fake_interaction_context.forum_trees,
        content="test",
        final_sender_alias="user",
        custom_field=custom_field,
        is_detached=False,
    )

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


def test_message_metadata_as_dict(fake_interaction_context: InteractionContext) -> None:
    """
    Test that the `Message.metadata_as_dict()` method returns only the custom fields as a dict.
    """
    message = Message(
        forum_trees=fake_interaction_context.forum_trees,
        content="test",
        final_sender_alias="user",
        custom_field={"role": "user"},
        is_detached=False,
    )

    assert isinstance(message.custom_field, Freeform)  # make sure it wasn't stored as plain dict
    assert message.metadata_as_dict() == {"custom_field": {"role": "user"}}


@pytest.mark.asyncio
async def test_message_aget_previous_msg(fake_interaction_context: InteractionContext) -> None:
    """
    Test that the `Message.aget_previous_msg` method returns the previous message if it exists.
    """
    message = Message(
        forum_trees=fake_interaction_context.forum_trees,
        content="message 1",
        final_sender_alias="user",
        is_detached=False,
    )
    await fake_interaction_context.forum_trees.astore_immutable(message)
    message = Message(
        forum_trees=fake_interaction_context.forum_trees,
        content="message 2",
        final_sender_alias="user",
        prev_msg_hash_key=message.hash_key,
        is_detached=False,
    )
    await fake_interaction_context.forum_trees.astore_immutable(message)

    assert message.content == "message 2"  # a sanity check
    previous_message = await message.aget_previous_msg()

    assert previous_message.content == "message 1"
    assert type(previous_message) is Message  # pylint: disable=unidiomatic-typecheck

    # no more previous messages
    assert await previous_message.aget_previous_msg() is None
