"""Data models."""
import hashlib

from pydantic import BaseModel


class Cacheable(BaseModel):
    """
    A base class for objects that can be cached. It is frozen and has a git-style hash key that is calculated from the
    JSON representation of the object.
    """

    class Config:
        """Pydantic config."""

        frozen = True

    @property
    def hash_key(self) -> str:
        """Get the cache key for this object. The cache key is a hash of the JSON representation of the object."""
        if not hasattr(self, "_hash_key"):
            # pylint: disable=attribute-defined-outside-init
            # noinspection PyAttributeOutsideInit
            self._hash_key = hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()
        return self._hash_key
