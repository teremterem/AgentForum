"""Data models."""
import hashlib
from functools import cached_property
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

    @cached_property
    def hash_key(self) -> str:
        """Get the hash key for this object. It is a hash of the JSON representation of the object."""
        # TODO Oleksandr: use
        #   json.dumps(self.model_dump(), ensure_ascii=False, sort_keys=True)
        #  instead of
        #   self.model_dump_json()
        #  to ensure that the hash key is independent of the order of the fields in the JSON representation ?
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()

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

    @cached_property
    def as_kwargs(self) -> Dict[str, Any]:
        """Get the fields of the object as a dictionary of keyword arguments."""
        return self.model_dump(exclude={"ac_model_"})

    @classmethod
    def _allowed_value_types(cls) -> Tuple[Type[Any], ...]:
        return _TYPES_ALLOWED_IN_FREEFORM


_TYPES_ALLOWED_IN_FREEFORM = *_PRIMITIVES_ALLOWED_IN_IMMUTABLE, Freeform


class Message(Immutable):
    """A message."""

    ac_model_: Literal["message"] = "message"
    content: str
    sender_alias: str
    metadata: Freeform = Freeform()  # empty metadata by default
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

    ac_model_: Literal["message"] = "forward"
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


class AgentCall(Message):
    """A subtype of Message that represents a call to an agent."""

    ac_model_: Literal["call"] = "call"
    msg_seq_start_hash_key: Optional[str] = None

    @property
    def receiver_alias(self) -> str:
        """Get the alias of the agent that is being called."""
        return self.content

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Get the keyword arguments for the agent call."""
        return self.metadata.as_kwargs


# TODO Oleksandr: introduce ErrorMessage for cases when something goes wrong (or maybe make it a part of Message ?)


class Token(Immutable):
    """
    A token. This class is used by MessagePromise (when the message is streamed token by token instead of being
    returned all at once).
    """

    ac_model_: Literal["token"] = "token"
    text: str
