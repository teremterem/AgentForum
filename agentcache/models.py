"""Data models."""
import hashlib
from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, Dict, Any, Literal

from pydantic import BaseModel, model_validator


class Immutable(BaseModel):
    """
    A base class for immutable pydantic objects. It is frozen and has a git-style hash key that is calculated from the
    JSON representation of the object.
    """

    class Config:
        """Pydantic config."""

        frozen = True
        extra = "forbid"

    model_: str

    @property
    def hash_key(self) -> str:
        """Get the hash key for this object. It is a hash of the JSON representation of the object."""
        if not hasattr(self, "_hash_key"):
            # TODO Oleksandr: should the json dump also include the class name ?
            # pylint: disable=attribute-defined-outside-init
            # noinspection PyAttributeOutsideInit
            self._hash_key = hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()
        return self._hash_key

    # noinspection PyNestedDecorators
    @model_validator(mode="before")
    @classmethod
    def _validate_immutable_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively make sure that the field values of the object are immutable."""
        for key, value in values.items():
            cls._validate_value(key, value)
        return values

    @classmethod
    def _validate_value(cls, key: str, value: Any) -> None:
        """Recursively make sure that the field value is immutable."""
        if isinstance(value, tuple):
            for sub_value in value:
                cls._validate_value(key, sub_value)
        elif not isinstance(value, _ALLOWED_TYPES_IN_IMMUTABLE):
            raise ValueError(
                f"only {{{', '.join([t.__name__ for t in _ALLOWED_TYPES_IN_IMMUTABLE])}}} "
                f"are allowed as field values in Immutable, got {type(value).__name__} in `{key}`"
            )


_ALLOWED_TYPES_IN_IMMUTABLE = (str, int, float, bool, type(None), Immutable)


class Metadata(Immutable):
    """Metadata for a message. Supports arbitrary fields."""

    class Config:
        """Pydantic config."""

        extra = "allow"

    model_: Literal["metadata"] = "metadata"


class Message(Immutable):
    """A message."""

    content: str
    metadata: Metadata = Metadata()  # empty metadata by default
    model_: Literal["message"] = "message"


# TODO Oleksandr: introduce ErrorMessage for cases when something goes wrong (or maybe make it a part of Message ?)


class Token(Immutable):
    """
    A token. This class is used by StreamedMessage (when the message is streamed token by token instead of being
    returned all at once).
    """

    text: str
    model_: Literal["token"] = "token"


class StreamedMessage(ABC):
    """A message that is streamed token by token instead of being returned all at once."""

    @abstractmethod
    def get_full_message(self) -> Message:
        """
        Get the full message. This method will "await" until all the tokens are received and then return the complete
        message.
        """

    @abstractmethod
    async def aget_full_message(self) -> Message:
        """
        Get the full message. This method will "await" until all the tokens are received and then return the complete
        message (async version).
        """

    @abstractmethod
    def __next__(self) -> Token:
        """Get the next token of a message that is being streamed."""

    @abstractmethod
    async def __anext__(self) -> Token:
        """Get the next token of a message that is being streamed (async version)."""

    def __aiter__(self) -> AsyncIterator[Token]:
        return self

    def __iter__(self) -> Iterator[Token]:
        return self
