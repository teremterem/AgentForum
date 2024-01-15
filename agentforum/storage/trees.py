# pylint: disable=import-outside-toplevel
"""Storage classes of the AgentForum."""
import typing
from abc import ABC, abstractmethod

from agentforum.errors import WrongImmutableTypeError

if typing.TYPE_CHECKING:
    from agentforum.models import Immutable, Message


class ForumTrees(ABC):
    """
    "Write Once Read Many" storage for the message trees. Can only accept Immutable objects. Once an object is stored,
    it cannot be changed.
    """

    @abstractmethod
    async def astore_immutable(self, immutable: "Immutable") -> None:
        """
        Store an Immutable object.
        """

    @abstractmethod
    async def aretrieve_immutable(self, hash_key: str) -> "Immutable":
        """
        Retrieve an Immutable object.
        """

    async def aretrieve_message(self, hash_key: str) -> "Message":
        """
        Retrieve a Message object. Same as `aretrieve_immutable`, but checks the type of the retrieved object to be a
        Message (or a subclass of it).
        """
        from agentforum.models import Message

        message = await self.aretrieve_immutable(hash_key)
        if not isinstance(message, Message):
            raise WrongImmutableTypeError(f"Expected a Message, got a {type(message)} - hash_key={hash_key}")
        return message
