"""TODO oleksandr"""
import hashlib

from pydantic import BaseModel


class Cacheable(BaseModel):
    """A base class for objects that can be cached."""

    class Config:
        """Pydantic config."""

        frozen = True

    @property
    def key(self) -> str:  # TODO oleksandr: is `key` a good name for this property ?
        """Get the cache key for this object. The cache key is a hash of the JSON representation of the object."""
        if not hasattr(self, "_key"):
            # pylint: disable=attribute-defined-outside-init
            # noinspection PyAttributeOutsideInit
            self._key = hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()
        return self._key
