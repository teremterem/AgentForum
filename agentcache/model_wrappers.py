"""
This module contains wrappers for the models defined in agentcache.models. These wrappers are used to add additional
functionality to the models without modifying the models themselves.
"""
from typing import Dict, Any, Optional

from agentcache.models import Message, Freeform, Token
from agentcache.typing import MessageType, IN
from agentcache.utils import Broadcastable


class StreamedMessage(Broadcastable[IN, Token]):  # TODO Oleksandr: come up with a better name for this class ?
    """A message that is streamed token by token instead of being returned all at once."""

    def __init__(self, *args, reply_to: Message, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reply_to = reply_to
        self._metadata: Dict[str, Any] = {}
        self._full_message = None

    async def aget_full_message(self) -> Message:
        """
        Get the full message. This method will "await" until all the tokens are received and then return the complete
        message (async version).
        """
        if not self._full_message:
            tokens = await self.aget_all()
            self._full_message = await self._reply_to.areply(  # TODO Oleksandr: allow _reply_to to be None
                content="".join([token.text for token in tokens]),
                metadata=Freeform(**self._metadata),  # TODO Oleksandr: create a separate function that does this ?
            )
        return self._full_message


class AsyncMessageBundle(Broadcastable[MessageType, MessageType]):
    """
    An asynchronous iterator over a bundle of messages that are being produced by an agent. Because the bundle is
    Broadcastable and relies on an internal async queue, the speed at which messages are produced and sent to the
    bundle is independent of the speed at which consumers iterate over them.
    """

    def __init__(self, *args, bundle_metadata: Optional[Freeform] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bundle_metadata: Freeform = bundle_metadata or Freeform()  # TODO Oleksandr: drop this field ?
