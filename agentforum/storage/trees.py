"""Storage classes of the AgentForum."""
import typing
from abc import ABC, abstractmethod

if typing.TYPE_CHECKING:
    from agentforum.models import Immutable


class ForumTrees(ABC):
    """
    "Write Once Read Many" storage for the message trees. Can only accept Immutable objects. Once an object is stored,
    it cannot be changed.
    """

    @abstractmethod
    async def astore_immutable(self, immutable: "Immutable") -> None:
        """Store an Immutable object."""

    @abstractmethod
    async def aretrieve_immutable(self, hash_key: str) -> "Immutable":
        """Retrieve an Immutable object."""
