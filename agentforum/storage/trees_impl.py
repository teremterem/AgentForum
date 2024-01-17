"""Storage classes of the AgentForum."""
from agentforum.errors import ImmutableDoesNotExist
from agentforum.models import Immutable
from agentforum.storage.trees import ForumTrees


class InMemoryTrees(ForumTrees):
    """An in-memory storage."""

    def __init__(self) -> None:
        self._immutable_data: dict[str, Immutable] = {}

    async def astore_immutable(self, immutable: Immutable) -> None:
        # TODO Oleksandr: uncomment the following lines when messages that evaded forwarding
        #  (do_not_forward_if_possible parameter) are not stored twice anymore
        # if immutable.hash_key in self._immutable_data:
        #     # TODO Oleksandr: introduce a custom exception for this case
        #     raise ValueError(f"an immutable object with hash key {immutable.hash_key} is already stored")
        self._immutable_data[immutable.hash_key] = immutable

    async def aretrieve_immutable(self, hash_key: str) -> Immutable:
        try:
            return self._immutable_data[hash_key]
        except KeyError as exc:
            raise ImmutableDoesNotExist(hash_key) from exc
