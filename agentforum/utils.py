"""Utility functions and classes for the AgentForum framework."""
import asyncio
from types import TracebackType
from typing import Optional, Iterable, List, AsyncIterator, Generic, Union, Type

from agentforum.errors import SendClosedError
from agentforum.typing import IN, OUT


class Sentinel:
    """A sentinel object used pass special values through queues indicating things like "end of queue" etc."""


END_OF_QUEUE = Sentinel()


class AsyncStreamable(Generic[IN, OUT]):
    """
    A stream of items that can be iterated over asynchronously. The support of multiple concurrent consumers is
    seamless. The speed at which items can be sent to the container is independent of the speed at which consumers
    iterate over them. It is not a generator, it is a container that can be iterated over multiple times.

    If `completed` is True, then iterating over this AsyncStreamable till the very end is not going to result in any
    awaiting (all the items were already consumed at least once and are immediately available).
    """

    def __init__(
        self,
        items_so_far: Optional[Iterable[OUT]] = None,
        completed: bool = False,
    ) -> None:
        self._send_closed: bool = completed

        self._items_so_far: List[Union[OUT, BaseException]] = []
        if items_so_far:
            self._items_so_far = list(items_so_far)

        if completed:
            self._queue_in = None
            self._queue_out = None
            self._lock = None
        else:
            self._queue_in = asyncio.Queue()
            self._queue_out = asyncio.Queue()
            self._lock = asyncio.Lock()
            asyncio.create_task(self._amove_items_from_in_to_out())

    @property
    def completed(self) -> bool:
        """
        Returns True if iterating over this AsyncStreamable till the very end is not going to result in any awaiting
        (all the items were already consumed at least once and are immediately available).
        """
        return not self._queue_out

    def __aiter__(self) -> AsyncIterator[OUT]:
        return self._AsyncIterator(self)

    # noinspection PyMethodMayBeStatic
    async def _aconvert_incoming_item(self, incoming_item: IN) -> AsyncIterator[Union[OUT, BaseException]]:
        """
        Convert a single incoming item into ZERO OR MORE outgoing items. The default implementation just yields the
        incoming item as is. This method exists as a separate method in order to be overridden in subclasses if needed.
        """
        yield incoming_item

    async def _amove_items_from_in_to_out(self) -> None:
        while True:
            async for item_out in self._aget_and_convert_incoming_item():
                self._queue_out.put_nowait(item_out)
                if item_out is END_OF_QUEUE:
                    self._queue_in = None
                    return

    async def _aget_and_convert_incoming_item(self) -> AsyncIterator[Union[OUT, Sentinel, BaseException]]:
        item_in = await self._queue_in.get()
        if isinstance(item_in, Sentinel):
            yield item_in  # pass the sentinel through as is
        else:
            try:
                async for item_out in self._aconvert_incoming_item(item_in):
                    yield item_out
            except BaseException as exc:  # pylint: disable=broad-except
                yield exc

    async def _anext_outgoing_item(self) -> OUT:
        if self.completed:
            raise StopAsyncIteration

        item_out = await self._queue_out.get()

        if item_out is END_OF_QUEUE:
            self._queue_out = None
            raise StopAsyncIteration

        self._items_so_far.append(item_out)
        return item_out

    class _Producer:  # pylint: disable=protected-access
        """A context manager that allows sending items to AsyncStreamable."""

        def __init__(self, async_streamable: "AsyncStreamable", suppress_exceptions: bool = False) -> None:
            self._async_streamable = async_streamable
            self._suppress_exceptions = suppress_exceptions

        def send(self, item: Union[IN, BaseException]) -> None:
            """Send an item to AsyncStreamable if it is still open (SendClosedError is raised otherwise)."""
            if self._async_streamable._send_closed:
                raise SendClosedError("Cannot send items to a closed AsyncStreamable.")
            self._async_streamable._queue_in.put_nowait(item)

        def close(self) -> None:
            """Close AsyncStreamable for sending. Has no effect if the container is already closed."""
            if not self._async_streamable._send_closed:
                self._async_streamable._send_closed = True
                self._async_streamable._queue_in.put_nowait(END_OF_QUEUE)

        def __enter__(self) -> "AsyncStreamable._Producer":
            return self

        def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_value: Optional[BaseException],
            traceback: Optional[TracebackType],
        ) -> Optional[bool]:
            is_send_closed_error = isinstance(exc_value, SendClosedError)
            if exc_value and not is_send_closed_error:
                # TODO Oleksandr: is it a good idea to send this exception by context manager automatically ?
                #  check this with the implementation of MessageSequence._MessageProducer.send_zero_or_more_messages()
                self.send(exc_value)
            self.close()
            # we are not suppressing SendClosedError even if self._suppress_exceptions is True
            return self._suppress_exceptions and not is_send_closed_error

    class _AsyncIterator(AsyncIterator[OUT]):
        def __init__(self, async_streamable: "AsyncStreamable") -> None:
            self._async_streamable = async_streamable
            self._index = 0

        async def __anext__(self) -> OUT:
            if self._index < len(self._async_streamable._items_so_far):
                item = self._async_streamable._items_so_far[self._index]
            elif self._async_streamable.completed:
                raise StopAsyncIteration
            else:
                async with self._async_streamable._lock:
                    if self._index < len(self._async_streamable._items_so_far):
                        item = self._async_streamable._items_so_far[self._index]
                    else:
                        item = await self._async_streamable._anext_outgoing_item()

            if isinstance(item, BaseException):
                raise item

            self._index += 1
            return item
