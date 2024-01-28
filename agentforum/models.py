"""
Data models.
"""
import hashlib
import json
from functools import cached_property
from typing import Any, Literal, Optional

from pydantic import BaseModel, model_validator, ConfigDict

from agentforum.storage.trees import ForumTrees

_PRIMITIVES_ALLOWED_IN_IMMUTABLE = (type(None), str, int, float, bool, tuple, list, dict)


class Immutable(BaseModel):
    """
    A base class for immutable pydantic objects. It is frozen and has a git-style hash key that is calculated from the
    JSON representation of the object.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @cached_property
    def hash_key(self) -> str:
        """
        Get the hash key for this object. It is a hash of the JSON representation of the object.
        """
        return hashlib.sha256(
            json.dumps(self.model_dump(exclude=self._exclude_from_hash()), ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()

    @cached_property
    def as_dict(self) -> dict[str, Any]:
        """
        Get the fields of the object as a dictionary. Omits im_model_ field (which may be defined in subclasses).
        """
        return self.model_dump(exclude=self._exclude_from_dict() | self._exclude_from_hash())

    @classmethod
    def _pre_process_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Pre-process the values before validation.
        """
        return values

    # noinspection PyNestedDecorators
    @model_validator(mode="before")
    @classmethod
    def _validate_immutable_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively make sure that the field values of the object are immutable.
        """
        values = cls._pre_process_values(values)
        for key, value in values.items():
            values[key] = cls._validate_value(key, value)
        return values

    @classmethod
    def _validate_value(cls, key: str, value: Any) -> Any:
        """
        Recursively make sure that the field value is immutable.
        """
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
    def _allowed_value_types(cls) -> tuple[type[Any], ...]:
        return _TYPES_ALLOWED_IN_IMMUTABLE

    # noinspection PyMethodMayBeStatic
    def _exclude_from_dict(self) -> set[str]:
        return {"im_model_"}

    def _exclude_from_hash(self) -> set[str]:
        return set()


_TYPES_ALLOWED_IN_IMMUTABLE = *_PRIMITIVES_ALLOWED_IN_IMMUTABLE, Immutable


class Freeform(Immutable):
    """
    An immutable generic model that has no predefined fields and only supports arbitrary ones. It also supports nested
    Freeform objects if necessary.
    """

    model_config = ConfigDict(extra="allow")

    @classmethod
    def _allowed_value_types(cls) -> tuple[type[Any], ...]:
        return _TYPES_ALLOWED_IN_FREEFORM


_TYPES_ALLOWED_IN_FREEFORM = *_PRIMITIVES_ALLOWED_IN_IMMUTABLE, Freeform


class Message(Freeform):
    """
    A message.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    im_model_: Literal["message"] = "message"
    forum_trees: ForumTrees
    content: str
    sender_alias: str
    prev_msg_hash_key: Optional[str] = None
    is_error: bool = False
    _error: BaseException = NotImplementedError("serialized error messages are not raisable yet")

    @property
    def original_sender_alias(self) -> str:
        """
        Get the alias of the original sender of the message.
        """
        return self.sender_alias

    @property
    def final_sender_alias(self) -> str:
        """
        Get the alias of the final sender of the message. It may be different from the original sender if the message
        is forwarded.
        """
        return self.sender_alias

    async def aget_previous_msg(self, skip_agent_calls: bool = True) -> Optional["Message"]:
        """
        Get the previous message in the forum.
        """
        if self.prev_msg_hash_key is None:
            return None
        previous_message = await self.forum_trees.aretrieve_message(self.prev_msg_hash_key)

        if skip_agent_calls:
            while previous_message and isinstance(previous_message, AgentCallMsg):
                previous_message = await previous_message.aget_previous_msg(
                    skip_agent_calls=False  # let's avoid further recursion - we have a loop instead
                )

        return previous_message

    @cached_property
    def metadata_as_dict(self) -> dict[str, Any]:
        """
        Get the metadata from a Message instance as a dictionary. All the custom fields (those which are not defined
        on the model) are considered metadata
        """
        return self.model_dump(exclude=set(self.model_fields))

    def get_original_msg(self, return_self_if_none: bool = True) -> Optional["Message"]:
        """
        Get the original message if this message is forwarded one or more times. If the message is forwarded multiple
        times (a forward of a forward of a forward and so on), this method returns the ultimate original message (all
        the way up the forwarding chain).

        For regular (not forwarded) messages, if `return_self_if_none` is True, returns self, otherwise returns None.
        """
        return self if return_self_if_none else None

    def get_before_forward(self, return_self_if_none: bool = True) -> Optional["Message"]:
        """
        Get the version of the message before the very last forward if this message is forwarded one or more times.
        This means that if, for ex., the message is a forward of a forward of a forward and so on, this method returns
        a message that itself is also a forwarded message too (this method does only one step in the forwarding chain).

        For regular (not forwarded) messages, if `return_self_if_none` is True, returns self, otherwise returns None.
        """
        return self if return_self_if_none else None

    def _exclude_from_hash(self):
        return super()._exclude_from_hash() | {"forum_trees"}

    @classmethod
    def _validate_value(cls, key: str, value: Any) -> Any:
        if key == "forum_trees":
            if not isinstance(value, ForumTrees):
                raise ValueError(f"`forum_trees` must be an instance of ForumTrees, got `{type(value).__name__}`")
            return value
        return super()._validate_value(key, value)


class ForwardedMessage(Message):
    """
    A subtype of Message that represents a message forwarded by an agent.
    """

    im_model_: Literal["message"] = "forward"
    msg_before_forward_hash_key: str

    _msg_before_forward: Optional["Message"] = None

    @cached_property
    def original_sender_alias(self) -> str:
        return self.get_original_msg().sender_alias

    def get_original_msg(self, return_self_if_none: bool = True) -> Optional["Message"]:
        original_msg = self
        while before_forward := original_msg.get_before_forward(return_self_if_none=False):
            original_msg = before_forward
        return original_msg

    def get_before_forward(self, return_self_if_none: bool = True) -> Optional["Message"]:
        if not self._msg_before_forward:
            raise RuntimeError("_msg_before_forward property was not initialized")
        return self._msg_before_forward

    def _set_msg_before_forward(self, msg_before_forward: Message) -> None:
        if msg_before_forward.hash_key != self.msg_before_forward_hash_key:
            raise RuntimeError(
                f"msg_before_forward_hash_key (left) does not match the hash key of the actual message before "
                f"forward (right): {self.msg_before_forward_hash_key} != {self._msg_before_forward.hash_key}"
            )
        self._msg_before_forward = msg_before_forward


class AgentCallMsg(Message):
    """
    A subtype of Message that represents a call to an agent.
    """

    im_model_: Literal["call"] = "call"
    function_kwargs: Freeform = Freeform()
    msg_seq_start_hash_key: Optional[str] = None

    @property
    def receiver_alias(self) -> str:
        """Get the alias of the agent that is being called."""
        return self.content  # TODO Oleksandr: stop using `content` for this purpose ?


class ContentChunk(BaseModel):
    """
    A chunk of message content. For ex. a token if the message is streamed token by token.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
