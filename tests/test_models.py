"""Tests for the models module."""
import hashlib
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agentcache.models import Cacheable


class SampleCacheable(Cacheable):
    """A sample Cacheable subclass."""

    some_req_field: str
    some_opt_field: int = 2
    sub_cacheable: "SampleCacheable" = None


def test_cacheable_frozen() -> None:
    """Test that the `Cacheable` class is frozen."""

    sample = SampleCacheable(some_req_field="test")

    with pytest.raises(ValidationError):
        sample.some_req_field = "test2"
    with pytest.raises(ValidationError):
        sample.some_opt_field = 3

    assert sample.some_req_field == "test"
    assert sample.some_opt_field == 2


def test_cacheable_hash_key() -> None:
    """Test the `Cacheable.hash_key` property."""
    sample = SampleCacheable(
        some_req_field="test", sub_cacheable=SampleCacheable(some_req_field="юнікод", some_opt_field=3)
    )

    # print(sample.model_dump_json())
    expected_hash_key = hashlib.sha256(
        '{"some_req_field":"test","some_opt_field":2,"sub_cacheable":'
        '{"some_req_field":"юнікод","some_opt_field":3,"sub_cacheable":null}}'.encode("utf-8")
    ).hexdigest()
    assert sample.hash_key == expected_hash_key == "147b478a9adbdcad47474b23069b36ff169664068acb4c5ce2dc65069c8ec9e5"


def test_cacheable_hash_key_calculated_once() -> None:
    """
    Test that the `Cacheable.hash_key` property is calculated only once and all subsequent calls return the same
    value without calculating it again.
    """
    original_sha256 = hashlib.sha256

    with patch("hashlib.sha256", side_effect=original_sha256) as mock_sha256:
        sample = SampleCacheable(some_req_field="test")
        mock_sha256.assert_not_called()  # not calculated yet

        assert sample.hash_key == "4d7028d03126b82a63ade7a54fa69f4fc04da5eb3fa24e4869d140dcd4cf5126"
        mock_sha256.assert_called_once()  # calculated once

        assert sample.hash_key == "4d7028d03126b82a63ade7a54fa69f4fc04da5eb3fa24e4869d140dcd4cf5126"
        mock_sha256.assert_called_once()  # check that it wasn't calculated again


def test_nested_object_not_copied() -> None:
    """Test that nested objects are not copied when the outer pydantic model is created."""
    sub_cacheable = SampleCacheable(some_req_field="test")
    sample = SampleCacheable(some_req_field="test", sub_cacheable=sub_cacheable)

    assert sample.sub_cacheable is sub_cacheable
