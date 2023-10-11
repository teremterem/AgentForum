"""Data models."""
import hashlib

from pydantic import BaseModel


class Immutable(BaseModel):
    """
    A base class for immutable pydantic objects. It is frozen and has a git-style hash key that is calculated from the
    JSON representation of the object.
    """

    class Config:
        """Pydantic config."""

        frozen = True

    @property
    def hash_key(self) -> str:
        """Get the hash key for this object. It is a hash of the JSON representation of the object."""
        if not hasattr(self, "_hash_key"):
            # pylint: disable=attribute-defined-outside-init
            # noinspection PyAttributeOutsideInit
            self._hash_key = hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()
        return self._hash_key
