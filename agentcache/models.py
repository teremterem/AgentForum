"""Data models."""
import hashlib
from typing import Dict, Any, Literal, Type, Tuple, Optional

from pydantic import BaseModel, model_validator, ConfigDict

_PRIMITIVES_ALLOWED_IN_IMMUTABLE = (str, int, float, bool, type(None))


class Immutable(BaseModel):
    """
    A base class for immutable pydantic objects. It is frozen and has a git-style hash key that is calculated from the
    JSON representation of the object.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    ac_model_: str  # AgentCache model name

    @property
    def hash_key(self) -> str:
        """Get the hash key for this object. It is a hash of the JSON representation of the object."""
        if not hasattr(self, "_hash_key"):
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
        elif not isinstance(value, cls._allowed_value_types()):
            raise ValueError(
                f"only {{{', '.join([t.__name__ for t in cls._allowed_value_types()])}}} "
                f"are allowed as field values in {cls.__name__}, got {type(value).__name__} in `{key}`"
            )

    @classmethod
    def _allowed_value_types(cls) -> Tuple[Type[Any], ...]:
        return _TYPES_ALLOWED_IN_IMMUTABLE


_TYPES_ALLOWED_IN_IMMUTABLE = *_PRIMITIVES_ALLOWED_IN_IMMUTABLE, Immutable


class Freeform(Immutable):
    """
    An immutable generic model that has no predefined fields and only supports arbitrary ones. It also supports nested
    Freeform objects if necessary.
    """

    model_config = ConfigDict(extra="allow")
    ac_model_: Literal["freeform"] = "freeform"

    @property
    def as_kwargs(self) -> Dict[str, Any]:
        """Get the fields of the object as a dictionary of keyword arguments."""
        if not hasattr(self, "_as_kwargs"):
            # pylint: disable=attribute-defined-outside-init
            # noinspection PyAttributeOutsideInit
            self._as_kwargs = self.model_dump(exclude={"ac_model_"})
        return self._as_kwargs

    @classmethod
    def _allowed_value_types(cls) -> Tuple[Type[Any], ...]:
        return _TYPES_ALLOWED_IN_FREEFORM


_TYPES_ALLOWED_IN_FREEFORM = *_PRIMITIVES_ALLOWED_IN_IMMUTABLE, Freeform


class Message(Immutable):
    """A message."""

    ac_model_: Literal["message"] = "message"
    content: str
    # TODO Oleksandr: support `sender_alias: str`
    metadata: Freeform = Freeform()  # empty metadata by default
    prev_msg_hash_key: Optional[str] = None


class AgentCall(Message):
    """A subtype of Message that represents a call to an agent."""

    ac_model_: Literal["call"] = "call"

    @property
    def agent_alias(self) -> str:
        """Get the alias of the agent that is being called."""
        return self.content

    @property
    def request_hash_key(self) -> str:
        """
        Get the hash key of the request message. The message this "call" object is a response to is considered the
        request.
        """
        return self.prev_msg_hash_key

    @property
    def kwargs(self) -> Freeform:
        """Get the keyword arguments for the agent call."""
        return self.metadata


# TODO Oleksandr: introduce ErrorMessage for cases when something goes wrong (or maybe make it a part of Message ?)


class Token(Immutable):
    """
    A token. This class is used by StreamedMessage (when the message is streamed token by token instead of being
    returned all at once).
    """

    ac_model_: Literal["token"] = "token"
    text: str
