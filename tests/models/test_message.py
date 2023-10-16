"""Test the `Message` and `Metadata` models."""
import hashlib

import pytest
from pydantic import ValidationError

from agentcache.models import Message, Metadata


def test_message_frozen() -> None:
    """Test that the `Message` class is frozen."""
    message = Message(content="test")

    with pytest.raises(ValidationError):
        message.content = "test2"

    assert message.content == "test"


def test_metadata_frozen() -> None:
    """Test that the `Metadata` class is frozen."""
    metadata = Metadata(some_field="some value")

    with pytest.raises(ValidationError):
        metadata.content = "some other value"

    assert metadata.some_field == "some value"


def test_message_hash_key() -> None:
    """Test the `Message.hash_key` property."""
    message = Message(content="test", metadata=Metadata(role="user"))
    # print(message.model_dump_json())
    expected_hash_key = hashlib.sha256(
        '{"ac_model_":"message","content":"test","metadata":{"ac_model_":"metadata","role":"user"}}'.encode("utf-8")
    ).hexdigest()
    assert message.hash_key == expected_hash_key

    message = Message(content="test")
    # print(message.model_dump_json())
    expected_hash_key = hashlib.sha256(
        '{"ac_model_":"message","content":"test","metadata":{"ac_model_":"metadata"}}'.encode("utf-8")
    ).hexdigest()
    assert message.hash_key == expected_hash_key


def test_nested_object_not_copied() -> None:
    """Test that nested objects are not copied."""
    metadata = Metadata(role="assistant")
    message = Message(content="test", metadata=metadata)

    assert message.metadata is metadata
