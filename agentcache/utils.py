"""Utility functions and classes for the AgentCache framework."""
import asyncio
from typing import Optional, Iterable, List, AsyncIterator, Generic, Union

from agentcache.errors import SendClosedError
from agentcache.typing import IN, OUT


class Sentinel:
    """A sentinel object used pass special values through queues indicating things like "end of queue" etc."""


END_OF_QUEUE = Sentinel()


class Broadcastable(Generic[IN, OUT]):
    """
    A container of items that can be iterated over asynchronously. The support of multiple concurrent consumers is
    seamless. The speed at which items can be sent to the container is independent of the speed at which consumers
    iterate over them. It is not a generator, it is a container that can be iterated over multiple times.

    - If `send_closed` is True, then it is not possible to send new items to the container anymore.
    - If `completed` is True, then all the items are already in the `items_so_far` list and the internal async queue
      has been disposed (all the items were already sent to the container AND at least one consumer already consumed
      them all).
    """

    def __init__(
        self,
        items_so_far: Optional[Iterable[OUT]] = None,
        items_so_far_raw: Optional[Iterable[IN]] = None,
        completed: bool = False,
    ) -> None:
        if items_so_far and items_so_far_raw:
            raise ValueError("Only one of `items_so_far` and `items_so_far_raw` should be provided.")

        self.send_closed: bool = completed

        self.items_so_far: List[OUT] = []
        if items_so_far:
            self.items_so_far = list(items_so_far)
        elif items_so_far_raw:
            self.items_so_far = [self._convert_item(item_raw) for item_raw in items_so_far_raw or ()]

        self._queue = None if completed else asyncio.Queue()
        self._lock = asyncio.Lock()

    def __aiter__(self) -> AsyncIterator[OUT]:
        return self._AsyncIterator(self)

    def __enter__(self) -> "Broadcastable":
        return self

    def __exit__(
        self, exc_type: Optional[Exception], exc_val: Optional[Exception], exc_tb: Optional[Exception]
    ) -> None:
        self.close()

    @property
    def completed(self) -> bool:
        """
        Return True if all the items have been sent to the container and at least one consumer already consumed them
        all.
        """
        return not self._queue

    async def aget_all(self) -> List[OUT]:
        """
        Get all the items in the container. This will block until all the items are available and sending is closed.
        """
        if not self.completed:
            async for _ in self:
                pass
        return self.items_so_far

    def send(self, item: IN) -> None:
        """Send an item to the container."""
        # TODO Oleksandr: should sending be allowed only in the context of a "with" block ?
        if self.send_closed:
            raise SendClosedError("Cannot send items to a closed Broadcastable.")
        self._queue.put_nowait(item)

    def close(self) -> None:
        """Close the container for sending. Has no effect if the container is already closed."""
        if not self.send_closed:
            self.send_closed = True
            self._queue.put_nowait(END_OF_QUEUE)

    async def _await_for_next_item(self) -> OUT:
        if self.completed:
            raise StopAsyncIteration

        item = await self._aget_and_convert_item()

        if item is END_OF_QUEUE:
            self.send_closed = True
            self._queue = None
            # TODO Oleksandr: at this point full Message should be built and stored (MessagePromise subclass)
            raise StopAsyncIteration

        self.items_so_far.append(item)
        return item

    async def _aget_and_convert_item(self) -> Union[OUT, Sentinel]:
        item = await self._aget_item_from_queue()
        if not isinstance(item, Sentinel):
            item = self._convert_item(item)
        return item

    async def _aget_item_from_queue(self) -> Union[IN, Sentinel]:
        return await self._queue.get()

    # noinspection PyMethodMayBeStatic
    def _convert_item(self, item: IN) -> OUT:
        """Convert an item from IN to OUT. Default implementation just returns the item as-is."""
        return item

    class _AsyncIterator(AsyncIterator[OUT]):
        def __init__(self, broadcastable: "Broadcastable") -> None:
            self._broadcastable = broadcastable
            self._index = 0

        async def __anext__(self) -> OUT:
            if self._index < len(self._broadcastable.items_so_far):
                item = self._broadcastable.items_so_far[self._index]
            elif self._broadcastable.completed:
                raise StopAsyncIteration
            else:
                async with self._broadcastable._lock:
                    if self._index < len(self._broadcastable.items_so_far):
                        item = self._broadcastable.items_so_far[self._index]
                    else:
                        item = await self._broadcastable._await_for_next_item()

            self._index += 1
            return item
