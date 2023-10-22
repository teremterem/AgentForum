"""
This module contains wrappers for the models defined in agentcache.models. These wrappers are used to add additional
functionality to the models without modifying the models themselves.
"""
from typing import Dict, Any, Optional, List

from agentcache.models import Message, Freeform, Token, _AgentCall
from agentcache.storage import ImmutableStorage
from agentcache.typing import IN
from agentcache.utils import Broadcastable


class StreamedMessage(Broadcastable[IN, Token]):
    """A message that is streamed token by token instead of being returned all at once."""

    def __init__(
        self, forum: ImmutableStorage, full_message: Message = None, reply_to: "StreamedMessage" = None
    ) -> None:
        if full_message and reply_to:
            raise ValueError("Only one of `full_message` and `reply_to` should be specified")

        super().__init__(
            items_so_far=[Token(text=full_message.content)] if full_message else None,
            completed=bool(full_message),
        )
        self.forum: ImmutableStorage = forum
        self._full_message: Message = full_message
        self._reply_to: StreamedMessage = reply_to
        self._metadata: Dict[str, Any] = {}

    async def aget_full_message(self) -> Message:
        """
        Get the full message. This method will "await" until all the tokens are received and then return the complete
        message (async version).
        """
        if not self._full_message:
            tokens = await self.aget_all()
            self._full_message = Message(
                content="".join([token.text for token in tokens]),
                metadata=Freeform(**self._metadata),  # TODO Oleksandr: create a separate function that does this ?
                prev_msg_hash_key=(await self._reply_to.aget_full_message()).hash_key if self._reply_to else None,
            )
            await self.forum.astore_immutable(self._full_message)
        return self._full_message

    async def aget_content(self) -> str:
        """Get the content of the full message (async version)."""
        return (await self.aget_full_message()).content

    async def aget_metadata(self) -> Freeform:
        """Get the metadata of the full message (async version)."""
        return (await self.aget_full_message()).metadata

    async def aget_previous_message(self) -> Optional["StreamedMessage"]:
        """Get the previous message in the conversation."""
        full_message = await self.aget_full_message()
        if not full_message.prev_msg_hash_key:
            return None

        if not hasattr(self, "_prev_msg"):
            prev_msg_hash_key = full_message.prev_msg_hash_key
            while isinstance(prev_msg := await self.forum.aretrieve_immutable(prev_msg_hash_key), _AgentCall):
                # skip agent calls
                prev_msg_hash_key = prev_msg.request_hash_key
            # pylint: disable=attribute-defined-outside-init
            # noinspection PyAttributeOutsideInit
            self._prev_msg = StreamedMessage(forum=self.forum, full_message=prev_msg)
        return self._prev_msg

    async def aget_full_chat(self) -> List["StreamedMessage"]:
        """Get the full chat history for this message (including this message)."""
        # TODO Oleksandr: introduce a limit on the number of messages to fetch
        msg = self
        result = [msg]
        while msg := await msg.aget_previous_message():
            result.append(msg)
        result.reverse()
        return result


class MessageSequence(Broadcastable[StreamedMessage, StreamedMessage]):
    """
    An asynchronous iterable over a sequence of messages that are being produced by an agent. Because the sequence is
    Broadcastable and relies on an internal async queue, the speed at which messages are produced and sent to the
    sequence is independent of the speed at which consumers iterate over them.
    """
