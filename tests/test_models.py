"""Tests for the models module."""
import pytest
from pydantic import ValidationError

from agentcache.models import Cacheable


def test_cacheable_frozen() -> None:
    """Test that the `Cacheable` class is frozen."""

    class SampleCacheable(Cacheable):
        """A sample cacheable object."""

        some_req_field: str
        some_opt_field: int = 2

    sample = SampleCacheable(some_req_field="test")

    with pytest.raises(ValidationError):
        sample.some_req_field = "test2"
    with pytest.raises(ValidationError):
        sample.some_opt_field = 3

    assert sample.some_req_field == "test"
    assert sample.some_opt_field == 2
