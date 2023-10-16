"""Data models."""
import asyncio
import hashlib
import typing
from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, Dict, Any, Literal, Type, Tuple, List, Optional, Iterable

from pydantic import BaseModel, model_validator, PrivateAttr, ConfigDict

from agentcache.errors import MessageBundleClosedError, MessageBundleNotFinishedError
from agentcache.typing import MessageType
from agentcache.utils import END_OF_QUEUE

if typing.TYPE_CHECKING:
    from agentcache.message_tree import MessageTree

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


class Metadata(Immutable):
    """Metadata for a message. Supports arbitrary fields."""

    model_config = ConfigDict(extra="allow")
    ac_model_: Literal["metadata"] = "metadata"

    @classmethod
    def _allowed_value_types(cls) -> Tuple[Type[Any], ...]:
        return _TYPES_ALLOWED_IN_METADATA


_TYPES_ALLOWED_IN_METADATA = *_PRIMITIVES_ALLOWED_IN_IMMUTABLE, Metadata


class Message(Immutable):
    """A message."""

    ac_model_: Literal["message"] = "message"
    _message_tree: "MessageTree" = PrivateAttr()  # set by MessageTree.anew_message()
    content: str
    metadata: Metadata = Metadata()  # empty metadata by default
    prev_msg_hash_key: Optional[str] = None

    @property
    def message_tree(self) -> "MessageTree":
        """Get the message tree that this message belongs to."""
        return self._message_tree

    async def aget_previous_message(self) -> Optional["Message"]:
        """Get the previous message in the conversation."""
        if not self.prev_msg_hash_key:
            return None

        if not hasattr(self, "_prev_msg"):
            # pylint: disable=attribute-defined-outside-init
            # noinspection PyAttributeOutsideInit
            self._prev_msg = await self.message_tree.afind_message(self.prev_msg_hash_key)
        return self._prev_msg

    async def areply(self, content: str, metadata: Optional[Metadata] = None) -> "Message":
        """Reply to this message."""
        return await self.message_tree.anew_message(
            content=content, metadata=metadata, prev_msg_hash_key=self.hash_key
        )

    async def aget_full_chat(self) -> List["Message"]:
        """Get the full chat history for this message (including this message)."""
        # TODO Oleksandr: introduce a limit on the number of messages to fetch
        msg = self
        result = [msg]
        while msg := await msg.aget_previous_message():
            result.append(msg)
        result.reverse()
        return result


# TODO Oleksandr: introduce ErrorMessage for cases when something goes wrong (or maybe make it a part of Message ?)


class Token(Immutable):
    """
    A token. This class is used by StreamedMessage (when the message is streamed token by token instead of being
    returned all at once).
    """

    ac_model_: Literal["token"] = "token"
    text: str


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


class MessageBundle:
    """
    A bundle of messages that can be iterated over asynchronously. The speed at which messages can be sent to the
    bundle is independent of the speed at which consumers iterate over them.
    - If the bundle is `closed`, then it is not possible to send new messages to it anymore.
    - If the bundle is `complete`, then all the messages are already in the `messages_so_far` list and the async queue
      has been disposed.
    """

    def __init__(
        self,
        bundle_metadata: Optional[Metadata] = None,
        messages_so_far: Optional[Iterable[MessageType]] = None,
        complete: bool = False,
    ) -> None:
        self.bundle_metadata: Metadata = bundle_metadata or Metadata()
        self.messages_so_far: List[MessageType] = list(messages_so_far or [])
        self.closed: bool = complete

        self._message_queue = None if complete else asyncio.Queue()
        self._lock = asyncio.Lock()

    def __aiter__(self) -> AsyncIterator[MessageType]:
        # noinspection PyTypeChecker
        return self._Iterator(self)

    @property
    def complete(self) -> bool:
        """Check whether all the messages in the bundle have been fetched."""
        return not self._message_queue

    def get_all_messages(self) -> List[MessageType]:
        """Get all the messages in the bundle."""
        # TODO Oleksandr: drop support of this method ?
        if not self.complete:
            raise MessageBundleNotFinishedError(
                "MessageBundle hasn't finished fetching messages. Either finish iterating over it asynchronously "
                "first or use aget_all_messages() instead of get_all_messages() which would finish iterating over it "
                "automatically."
            )
        return self.messages_so_far

    async def aget_all_messages(self) -> List[MessageType]:
        """Get all the messages in the bundle, but make sure that all the messages are fetched first."""
        if not self.complete:
            async for _ in self:
                pass
        return self.messages_so_far

    def send_message(self, message: MessageType, close_bundle: bool) -> None:
        """Send a message to the bundle."""
        if self.closed:
            raise MessageBundleClosedError("Cannot send messages to a closed bundle.")
        self._message_queue.put_nowait(message)
        if close_bundle:
            self._message_queue.put_nowait(END_OF_QUEUE)
            self.closed = True

    def send_interim_message(self, message: MessageType) -> None:
        """Send an interim message to the bundle."""
        self.send_message(message, close_bundle=False)

    def send_final_message(self, message: MessageType) -> None:
        """Send the final message to the bundle. The bundle will be closed after this."""
        self.send_message(message, close_bundle=True)

    async def _wait_for_next_message(self) -> MessageType:
        if self.complete:
            raise StopAsyncIteration

        message = await self._message_queue.get()

        if message is END_OF_QUEUE:
            self._message_queue = None
            self.closed = True
            raise StopAsyncIteration

        self.messages_so_far.append(message)
        return message

    class _Iterator:
        def __init__(self, message_bundle: "MessageBundle") -> None:
            self._message_bundle = message_bundle
            self._index = 0

        async def __anext__(self) -> MessageType:
            if self._index < len(self._message_bundle.messages_so_far):
                message = self._message_bundle.messages_so_far[self._index]
            elif self._message_bundle.complete:
                raise StopAsyncIteration
            else:
                async with self._message_bundle._lock:
                    if self._index < len(self._message_bundle.messages_so_far):
                        message = self._message_bundle.messages_so_far[self._index]
                    else:
                        message = await self._message_bundle._wait_for_next_message()

            self._index += 1
            return message
