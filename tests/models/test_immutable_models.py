"""Tests for the Immutable models."""
import hashlib
from typing import Literal, Optional
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agentcache.models import Immutable, Freeform, Message, ForwardedMessage


class SampleImmutable(Immutable):
    """A sample Immutable subclass."""

    some_req_field: str
    some_opt_field: int = 2
    sub_immutable: Optional["SampleImmutable"] = None
    ac_model_: Literal["sample_immutable"] = "sample_immutable"


def test_immutable_frozen() -> None:
    """Test that the `Immutable` class is frozen."""

    sample = SampleImmutable(some_req_field="test")

    with pytest.raises(ValidationError):
        sample.some_req_field = "test2"
    with pytest.raises(ValidationError):
        sample.some_opt_field = 3

    assert sample.some_req_field == "test"
    assert sample.some_opt_field == 2


def test_message_frozen() -> None:
    """Test that the `Message` class is frozen."""
    message = Message(content="test", sender_alias="user")

    with pytest.raises(ValidationError):
        message.content = "test2"

    assert message.content == "test"


def test_freeform_frozen() -> None:
    """Test that the `Freeform` class is frozen."""
    metadata = Freeform(some_field="some value")

    with pytest.raises(ValidationError):
        metadata.content = "some other value"

    assert metadata.some_field == "some value"


def test_immutable_hash_key() -> None:
    """Test the `Immutable.hash_key` property."""
    sample = SampleImmutable(
        some_req_field="test", sub_immutable=SampleImmutable(some_req_field="юнікод", some_opt_field=3)
    )

    # print(sample.model_dump_json())
    expected_hash_key = hashlib.sha256(
        '{"ac_model_":"sample_immutable","some_req_field":"test","some_opt_field":2,"sub_immutable":{"ac_model_":'
        '"sample_immutable","some_req_field":"юнікод","some_opt_field":3,"sub_immutable":null}}'.encode("utf-8")
    ).hexdigest()
    assert sample.hash_key == expected_hash_key


def test_message_hash_key() -> None:
    """Test the `Message.hash_key` property."""
    message = Message(content="test", sender_alias="user", metadata=Freeform(role="user"))
    # print(message.model_dump_json())
    expected_hash_key = hashlib.sha256(
        '{"ac_model_":"message","content":"test","sender_alias":"user","metadata":'
        '{"ac_model_":"freeform","role":"user"},"prev_msg_hash_key":null}'.encode("utf-8")
    ).hexdigest()
    assert message.hash_key == expected_hash_key

    message = Message(content="test", sender_alias="user")
    # print(message.model_dump_json())
    expected_hash_key = hashlib.sha256(
        '{"ac_model_":"message","content":"test","sender_alias":"user","metadata":'
        '{"ac_model_":"freeform"},"prev_msg_hash_key":null}'.encode("utf-8")
    ).hexdigest()
    assert message.hash_key == expected_hash_key


def test_forwarded_message_hash_key() -> None:
    """
    Assert that ForwardedMessage._original_msg is not serialized when calculating the hash_key of a ForwardedMessage
    (only ForwardedMessage.original_msg_hash_key is).
    """
    original_msg = Message(content="message that is being forwarded", sender_alias="user")

    message = ForwardedMessage(content="test", sender_alias="user", original_msg_hash_key=original_msg.hash_key)
    message._original_msg = original_msg  # pylint: disable=protected-access

    print(message.model_dump_json())
    expected_hash_key = hashlib.sha256(
        '{"ac_model_":"forward","content":"test","sender_alias":"user","metadata":'
        '{"ac_model_":"freeform"},"prev_msg_hash_key":null,'
        '"original_msg_hash_key":"06a3098ed3b5742998ae0528a68b3f11c61ff7403e6c7378de7da5e72ffd13aa"}'.encode("utf-8")
    ).hexdigest()
    assert message.hash_key == expected_hash_key


def test_nested_object_not_copied() -> None:
    """Test that nested objects are not copied when the outer pydantic model is created."""
    sub_immutable = SampleImmutable(some_req_field="test")
    sample = SampleImmutable(some_req_field="test", sub_immutable=sub_immutable)

    assert sample.sub_immutable is sub_immutable


def test_nested_message_freeform_not_copied() -> None:
    """Test that Freeform nested in Message is not copied."""
    metadata = Freeform(role="assistant")
    message = Message(content="test", sender_alias="user", metadata=metadata)

    assert message.metadata is metadata


def test_immutable_hash_key_calculated_once() -> None:
    """
    Test that the `Immutable.hash_key` property is calculated only once and all subsequent calls return the same
    value without calculating it again.
    """
    original_sha256 = hashlib.sha256

    with patch("hashlib.sha256", side_effect=original_sha256) as mock_sha256:
        sample = SampleImmutable(some_req_field="test")
        mock_sha256.assert_not_called()  # not calculated yet

        assert sample.hash_key == "694b7a9dfab8ec9e3a74459f734f0af0d5f7486629f9618db603c60c46bf4027"
        mock_sha256.assert_called_once()  # calculated once

        assert sample.hash_key == "694b7a9dfab8ec9e3a74459f734f0af0d5f7486629f9618db603c60c46bf4027"
        mock_sha256.assert_called_once()  # check that it wasn't calculated again
