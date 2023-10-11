"""Storage classes of the AgentCache."""
from abc import ABC, abstractmethod
from typing import Dict

from agentcache.models import Immutable


class WormStorage(ABC):
    """
    "Write Once Read Many" storage. Can only accept Immutable objects. Once an object is stored, it cannot be changed.
    """

    @abstractmethod
    def store(self, immutable: Immutable) -> None:
        """Store an object in the cache."""

    async def astore(self, immutable: Immutable) -> None:
        """Store an object in the cache (async version)."""
        self.store(immutable)

    @abstractmethod
    def retrieve(self, hash_key: str) -> Immutable:
        """Retrieve an object from the cache."""

    async def aretrieve(self, hash_key: str) -> Immutable:
        """Retrieve an object from the cache (async version)."""
        return self.retrieve(hash_key)


class InMemoryStorage(WormStorage):
    """An in-memory storage."""

    def __init__(self) -> None:
        self._storage: Dict[str, Immutable] = {}

    def store(self, immutable: Immutable) -> None:
        self._storage[immutable.hash_key] = immutable

    def retrieve(self, hash_key: str) -> Immutable:
        return self._storage[hash_key]
