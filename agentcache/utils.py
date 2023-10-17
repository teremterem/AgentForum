"""Utility functions and classes for the AgentCache framework."""
import asyncio
from typing import Optional, Iterable, List, AsyncIterator, TypeVar, Generic

from agentcache.errors import SendClosedError


class Sentinel:
    """A sentinel object used pass special values through queues indicating things like "end of queue" etc."""


END_OF_QUEUE = Sentinel()

T = TypeVar("T")


class Broadcastable(Generic[T]):
    """
    A container of items that can be iterated over asynchronously. The support of multiple concurrent consumers is
    seamless. The speed at which items can be sent to the container is independent of the speed at which consumers
    iterate over them.
    - If `send_closed` is True, then it is not possible to send new items to the container anymore.
    - If `completed` is True, then all the items are already in the `items_so_far` list and the internal async queue
      has been disposed (all the items were already sent to the container AND at least one consumer already consumed
      them all).
    """

    def __init__(
        self,
        items_so_far: Optional[Iterable[T]] = None,
        completed: bool = False,
    ) -> None:
        self.send_closed: bool = completed
        self.items_so_far: List[T] = list(items_so_far or [])

        self._queue = None if completed else asyncio.Queue()
        self._lock = asyncio.Lock()

    def __aiter__(self) -> AsyncIterator[T]:
        return self._AsyncIterator(self)

    @property
    def completed(self) -> bool:
        """
        Return True if all the items have been sent to the container and at least one consumer already consumed them
        all.
        """
        return not self._queue

    async def aget_all(self) -> List[T]:
        """
        Get all the items in the container. This will block until all the items are available and sending is closed.
        """
        if not self.completed:
            async for _ in self:
                pass
        return self.items_so_far

    def send(self, item: T) -> None:
        """Send an item to the container."""
        if self.send_closed:
            raise SendClosedError("Cannot send items to a closed Broadcastable.")
        self._queue.put_nowait(item)

    def close(self) -> None:
        """Close the container for sending. Has no effect if the container is already closed."""
        # TODO Oleksandr: turn this into a context manager
        if not self.send_closed:
            self.send_closed = True
            self._queue.put_nowait(END_OF_QUEUE)

    async def _wait_for_next_item(self) -> T:
        if self.completed:
            raise StopAsyncIteration

        item = await self._queue.get()

        if item is END_OF_QUEUE:
            self.send_closed = True
            self._queue = None
            raise StopAsyncIteration

        self.items_so_far.append(item)
        return item

    class _AsyncIterator(AsyncIterator[T]):
        def __init__(self, broadcastable: "Broadcastable") -> None:
            self._broadcastable = broadcastable
            self._index = 0

        async def __anext__(self) -> T:
            if self._index < len(self._broadcastable.items_so_far):
                item = self._broadcastable.items_so_far[self._index]
            elif self._broadcastable.completed:
                raise StopAsyncIteration
            else:
                async with self._broadcastable._lock:
                    if self._index < len(self._broadcastable.items_so_far):
                        item = self._broadcastable.items_so_far[self._index]
                    else:
                        item = await self._broadcastable._wait_for_next_item()

            self._index += 1
            return item
