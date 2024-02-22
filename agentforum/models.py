"""
Data models.
"""

import hashlib
import json
from functools import cached_property
from typing import Any, Literal, Optional

from pydantic import BaseModel, model_validator, ConfigDict

from agentforum.errors import DetachedMessageError
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

    def as_dict(self) -> dict[str, Any]:
        """
        Get the fields of the object as a dictionary. Omits im_model_ field (which may be defined in subclasses).
        """
        return self.model_dump(exclude=self._exclude_from_dict() | self._exclude_from_hash())

    @classmethod
    def _preprocess_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Preprocess the values before validation.
        """
        return values

    # noinspection PyNestedDecorators
    @model_validator(mode="before")
    @classmethod
    def _validate_immutable_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively make sure that the field values of the object are immutable.
        """
        values = cls._preprocess_values(values)
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
    forum_trees: Optional[ForumTrees] = None
    final_sender_alias: Optional[str] = None
    content: Optional[str] = None
    content_template: Optional[str] = None
    prev_msg_hash_key: Optional[str] = None
    reply_to_msg_hash_key: Optional[str] = None
    is_error: bool = False
    is_detached: bool = True  # TODO Oleksandr: users should not be able to set this field
    _error: BaseException = NotImplementedError("serialized error messages are not raisable yet")

    @property
    def original_sender_alias(self) -> str:
        """
        Get the alias of the original sender of the message.
        """
        return self.final_sender_alias

    async def aget_previous_msg(self) -> Optional["Message"]:
        """
        Get the previous message in the forum.
        """
        if self.prev_msg_hash_key is None:
            return None
        return await self.forum_trees.aretrieve_message(self.prev_msg_hash_key)

    async def aget_reply_to_msg(self) -> Optional["Message"]:
        """
        Get the message that this message is a reply to.
        """
        if self.reply_to_msg_hash_key is None:
            return None
        return await self.forum_trees.aretrieve_message(self.reply_to_msg_hash_key)

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

    @classmethod
    def _require_content(cls) -> bool:
        return True

    def _exclude_from_hash(self):
        return super()._exclude_from_hash() | {"forum_trees", "is_detached"}

    def _exclude_from_dict(self):
        exclude = super()._exclude_from_dict()
        if self.is_detached:
            exclude |= {"prev_msg_hash_key"}  # detached messages do not have a previous message
        return exclude

    @cached_property
    def hash_key(self) -> str:
        if self.is_detached:
            raise DetachedMessageError("detached messages cannot be hashed")
        return super().hash_key

    @classmethod
    def _preprocess_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        values = super()._preprocess_values(values)

        if cls._require_content() and "content" not in values and "content_template" not in values:
            raise ValueError("either `content` or `content_template` must be present in a message")

        if values.get("is_detached", cls.model_fields["is_detached"].default):
            if "forum_trees" in values or "prev_msg_hash_key" in values or "reply_to_msg_hash_key" in values:
                raise ValueError(
                    "neither `forum_trees` nor `prev_msg_hash_key` nor `reply_to_msg_hash_key` can be present in a "
                    "detached message"
                )
            if values.get("content") is not None and values.get("content_template") is not None:
                raise ValueError("`content` and `content_template` cannot both be set in a detached message")
        else:
            if not values.get("forum_trees"):
                raise ValueError("`forum_trees` is required in a non-detached message")
            if not values.get("final_sender_alias"):
                raise ValueError("`final_sender_alias` is required in a non-detached message")

        content_template = values.get("content_template")
        if content_template is not None:
            values["content"] = content_template.format(**values)

        return values

    @classmethod
    def _validate_value(cls, key: str, value: Any) -> Any:
        if key == "forum_trees":
            # we are making an exception for the type of this field (doesn't have to be one of the types allowed by
            # Immutable)
            return value
        return super()._validate_value(key, value)


class ForwardedMessage(Message):
    """
    A subtype of Message that represents a message forwarded by an agent.
    """

    im_model_: Literal["forward"] = "forward"
    msg_before_forward_hash_key: str
    is_detached: Literal[False] = False  # forwarded messages cannot be detached

    _msg_before_forward: Optional["Message"] = None

    @cached_property
    def original_sender_alias(self) -> str:
        return self.get_original_msg().final_sender_alias

    def get_original_msg(self, return_self_if_none: bool = True) -> Optional["Message"]:
        original_msg = self
        while before_forward := original_msg.get_before_forward(return_self_if_none=False):
            original_msg = before_forward
        return original_msg

    def get_before_forward(self, return_self_if_none: bool = True) -> Optional["Message"]:
        if not self._msg_before_forward:
            raise RuntimeError("`_msg_before_forward` property was not initialized")
        return self._msg_before_forward

    def _exclude_from_hash(self):
        return super()._exclude_from_hash() | {"content", "content_template"}

    def _set_msg_before_forward(self, msg_before_forward: Message) -> None:
        if msg_before_forward.hash_key != self.msg_before_forward_hash_key:
            raise RuntimeError(
                f"`msg_before_forward_hash_key` (left) does not match the hash key of the actual message before "
                f"forward (right): {self.msg_before_forward_hash_key} != {self._msg_before_forward.hash_key}"
            )
        self._msg_before_forward = msg_before_forward
        # we need to make the original `content` and `content_template` fields available in the forwarded message
        # directly, so, as a workaround, below we are circumventing the frozen nature of the model
        # TODO Oleksandr: are there any bad consequences of this workaround ?
        object.__setattr__(self, "content", msg_before_forward.content)
        object.__setattr__(self, "content_template", msg_before_forward.content_template)

    @classmethod
    def _validate_value(cls, key: str, value: Any) -> Any:
        if key in ("content", "content_template"):
            raise ValueError("neither `content` nor `content_template` can be present in a forwarded message")
        return super()._validate_value(key, value)

    @classmethod
    def _require_content(cls) -> bool:
        return False


class AgentCallMsg(Message):
    """
    A subtype of Message that represents a call to an agent.
    """

    receiver_alias: str
    im_model_: Literal["call"] = "call"
    function_kwargs: Freeform = Freeform()
    msg_seq_start_hash_key: Optional[str] = None
    is_detached: Literal[False] = False  # agent call messages cannot be detached

    @classmethod
    def _require_content(cls) -> bool:
        return False


class ContentChunk(BaseModel):
    """
    A chunk of message content. For ex. a token if the message is streamed token by token.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
