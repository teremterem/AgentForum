"""Data models."""
import hashlib
import json
from functools import cached_property
from typing import Dict, Any, Literal, Type, Tuple, Optional

from pydantic import BaseModel, model_validator, ConfigDict

_PRIMITIVES_ALLOWED_IN_IMMUTABLE = (type(None), str, int, float, bool, tuple, list, dict)


class Immutable(BaseModel):
    """
    A base class for immutable pydantic objects. It is frozen and has a git-style hash key that is calculated from the
    JSON representation of the object.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    im_model_: str  # immutable model type

    @cached_property
    def hash_key(self) -> str:
        """Get the hash key for this object. It is a hash of the JSON representation of the object."""
        return hashlib.sha256(
            json.dumps(self.model_dump(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    # noinspection PyNestedDecorators
    @model_validator(mode="before")
    @classmethod
    def _validate_immutable_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively make sure that the field values of the object are immutable."""
        for key, value in values.items():
            values[key] = cls._validate_value(key, value)
        return values

    @classmethod
    def _validate_value(cls, key: str, value: Any) -> Any:
        """Recursively make sure that the field value is immutable."""
        if isinstance(value, (tuple, list)):
            return tuple(cls._validate_value(key, sub_value) for sub_value in value)
        if isinstance(value, dict):
            return Freeform(**value)
        if not isinstance(value, cls._allowed_value_types()):
            raise ValueError(
                f"only {{{', '.join([t.__name__ for t in cls._allowed_value_types()])}}} "
                f"are allowed as field values in {cls.__name__}, got {type(value).__name__} in `{key}`"
            )
        return value

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
    im_model_: Literal["freeform"] = "freeform"

    @cached_property
    def as_kwargs(self) -> Dict[str, Any]:
        """Get the fields of the object as a dictionary of keyword arguments."""
        return self.model_dump(exclude={"im_model_"})

    @classmethod
    def _allowed_value_types(cls) -> Tuple[Type[Any], ...]:
        return _TYPES_ALLOWED_IN_FREEFORM


_TYPES_ALLOWED_IN_FREEFORM = *_PRIMITIVES_ALLOWED_IN_IMMUTABLE, Freeform


class Message(Freeform):
    """A message."""

    im_model_: Literal["message"] = "message"
    content: str
    sender_alias: str
    prev_msg_hash_key: Optional[str] = None

    def get_original_msg(self, return_self_if_none: bool = True) -> Optional["Message"]:
        """
        Get the original message that this message is a forward of. Because this is not a ForwardedMessage, it always
        returns either self or None, depending on return_self_if_none parameter (this implementation is overridden in
        ForwardedMessage).
        """
        return self if return_self_if_none else None


class ForwardedMessage(Message):
    """A subtype of Message that represents a message forwarded by an agent."""

    im_model_: Literal["message"] = "forward"
    original_msg_hash_key: str

    _original_msg: Optional["Message"] = None

    def get_original_msg(self, return_self_if_none: bool = True) -> Optional[Message]:
        """
        Get the original message that this message is a forward of. In the implementation found here in
        ForwardedMessage class the value of return_self_if_none parameter is irrelevant - it is illegal for
        ForwardedMessage not to have an original message.
        """
        if not self._original_msg:
            raise RuntimeError("original_msg property was not initialized")
        if self._original_msg.hash_key != self.original_msg_hash_key:
            raise RuntimeError(
                f"original_msg_hash_key does not match the hash_key of the original message: "
                f"{self.original_msg_hash_key} != {self._original_msg.hash_key}"
            )
        return self._original_msg


class AgentCallMsg(Message):
    """A subtype of Message that represents a call to an agent."""

    im_model_: Literal["call"] = "call"
    msg_seq_start_hash_key: Optional[str] = None

    @property
    def receiver_alias(self) -> str:
        """Get the alias of the agent that is being called."""
        return self.content

    @property
    def function_kwargs(self) -> Dict[str, Any]:  # TODO TODO TODO Oleksandr: you renamed it - did you break anything ?
        """Get the keyword arguments for the agent call."""
        # TODO TODO TODO Oleksandr: separate custom fields from the predefined ones
        return self.as_kwargs


# TODO Oleksandr: introduce ErrorMessage for cases when something goes wrong (or maybe make it a part of Message ?)


class ContentChunk(BaseModel):
    """A chunk of message content. For ex. a token if the message is streamed token by token."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str


class MessageParameters(BaseModel):
    """
    A set of parameters that can be converted into one or more messages or message promises.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    content: Any  # TODO Oleksandr: a newer version of Pydantic doesn't seem work with `MessageType` for some reason
    override_sender_alias: Optional[str] = None
    metadata: Freeform = Freeform()  # empty metadata by default  # TODO TODO TODO Oleksandr: keep as separate field ?
