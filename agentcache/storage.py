"""Storage classes of the AgentCache."""
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from agentcache.models import Immutable


class ImmutableStorage(ABC):
    """
    "Write Once Read Many" storage. Can only accept Immutable objects. Once an object is stored, it cannot be changed.
    """

    @abstractmethod
    def store_immutable(self, immutable: Immutable) -> None:
        """Store an Immutable object."""

    async def astore_immutable(self, immutable: Immutable) -> None:
        """Store an Immutable object (async version)."""
        self.store_immutable(immutable)

    @abstractmethod
    def retrieve_immutable(self, hash_key: str) -> Immutable:
        """Retrieve an Immutable object."""

    async def aretrieve_immutable(self, hash_key: str) -> Immutable:
        """Retrieve an Immutable object (async version)."""
        return self.retrieve_immutable(hash_key)


class StringStorage(ABC):
    """A mutable storage for strings. TODO Oleksandr: explain why we need this as a separate thing."""

    @abstractmethod
    def store_string(self, key: Tuple[str, ...], value: str) -> None:
        """Store a string."""

    async def astore_string(self, key: Tuple[str, ...], value: str) -> None:
        """Store a string (async version)."""
        self.store_string(key, value)

    @abstractmethod
    def retrieve_string(self, key: Tuple[str, ...]) -> str:
        """Retrieve a string."""

    async def aretrieve_string(self, key: Tuple[str, ...]) -> str:
        """Retrieve a string (async version)."""
        return self.retrieve_string(key)


class InMemoryStorage(ImmutableStorage, StringStorage):
    """An in-memory storage."""

    def __init__(self) -> None:
        self._immutable_data: Dict[str, Immutable] = {}
        self._string_data: Dict[Tuple[str, ...], str] = {}

    def store_immutable(self, immutable: Immutable) -> None:
        self._immutable_data[immutable.hash_key] = immutable

    def retrieve_immutable(self, hash_key: str) -> Immutable:
        return self._immutable_data[hash_key]

    def store_string(self, key: Tuple[str, ...], value: str) -> None:
        # TODO is any kind of locking needed here ?
        self._string_data[key] = value

    def retrieve_string(self, key: Tuple[str, ...]) -> str:
        return self._string_data[key]
