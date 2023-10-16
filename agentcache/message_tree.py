"""A tree of messages."""
from typing import Optional

from pydantic import BaseModel, ConfigDict

from agentcache.models import Metadata, Message
from agentcache.storage import ImmutableStorage


class MessageTree(BaseModel):
    """A tree of messages."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    immutable_storage: ImmutableStorage

    async def anew_message(
        self, content: str, metadata: Optional[Metadata] = None, prev_msg_hash_key: Optional[str] = None
    ) -> Message:
        """Create a new message."""
        message = Message(content=content, metadata=metadata or Metadata(), prev_msg_hash_key=prev_msg_hash_key)
        message._message_tree = self  # pylint: disable=protected-access
        await self.immutable_storage.astore_immutable(message)
        return message

    async def afind_message(self, hash_key: str) -> Message:
        """Find a message by its hash key."""
        message = await self.immutable_storage.aretrieve_immutable(hash_key)
        # TODO Oleksandr: how to guarantee that message._message_tree==self ?
        assert isinstance(message, Message)  # TODO Oleksandr: replace with a custom exception ?
        return message
