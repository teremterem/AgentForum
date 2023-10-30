"""Storage classes of the AgentCache."""
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from agentcache.models import Immutable


class ImmutableStorage(ABC):
    """
    "Write Once Read Many" storage. Can only accept Immutable objects. Once an object is stored, it cannot be changed.
    """

    @abstractmethod
    async def astore_immutable(self, immutable: Immutable) -> None:
        """Store an Immutable object."""

    @abstractmethod
    async def aretrieve_immutable(self, hash_key: str) -> Immutable:
        """Retrieve an Immutable object."""


class StringStorage(ABC):
    """A mutable storage for strings. TODO Oleksandr: explain why we need this as a separate thing."""

    @abstractmethod
    async def astore_string(self, key: Tuple[str, ...], value: str) -> None:
        """Store a string."""

    @abstractmethod
    async def aretrieve_string(self, key: Tuple[str, ...]) -> str:
        """Retrieve a string."""


class InMemoryStorage(ImmutableStorage, StringStorage):
    """An in-memory storage."""

    def __init__(self) -> None:
        self._immutable_data: Dict[str, Immutable] = {}
        self._string_data: Dict[Tuple[str, ...], str] = {}

    async def astore_immutable(self, immutable: Immutable) -> None:
        if immutable.hash_key in self._immutable_data:
            # TODO Oleksandr: introduce a custom exception for this case
            raise ValueError(f"an immutable object with hash key {immutable.hash_key} is already stored")
        self._immutable_data[immutable.hash_key] = immutable

    async def aretrieve_immutable(self, hash_key: str) -> Immutable:
        return self._immutable_data[hash_key]

    async def astore_string(self, key: Tuple[str, ...], value: str) -> None:
        self._string_data[key] = value

    async def aretrieve_string(self, key: Tuple[str, ...]) -> str:
        return self._string_data[key]
