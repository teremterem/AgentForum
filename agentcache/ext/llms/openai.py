"""OpenAI API extension for AgentCache."""
from typing import List, AsyncIterator, Dict, Any, Set, Optional

from agentcache.errors import AgentCacheError
from agentcache.models import Message, StreamedMessage, Token, Metadata
from agentcache.typing import MessageType


async def aopenai_chat_completion(messages: List[MessageType], kwargs: Optional[Metadata] = None) -> MessageType:
    """Chat with OpenAI models (async version). Returns a message or a stream of tokens."""
    import openai  # pylint: disable=import-outside-toplevel

    kwargs = kwargs.model_dump(exclude={"ac_model_"}) if kwargs else {}
    stream = kwargs.pop("stream", False)
    n = kwargs.pop("n", 1)

    if n != 1:
        raise AgentCacheError("Only n=1 is supported by AgentCache for openai.ChatCompletion.acreate()")

    messages = [msg.get_full_message() if isinstance(msg, StreamedMessage) else msg for msg in messages]
    message_dicts = [
        {
            "role": getattr(msg.metadata, "openai_role", "user"),
            "content": msg.content,
        }
        for msg in messages
    ]
    # pprint(message_dicts)
    # print("\n")
    response = await openai.ChatCompletion.acreate(messages=message_dicts, stream=stream, **kwargs)

    if stream:
        # noinspection PyTypeChecker
        return _StreamedMessageAsync(response, messages[-1])

    # pprint(response)
    # print()
    return await messages[-1].areply(
        content=response["choices"][0]["message"]["content"],
        metadata=Metadata(**_build_openai_metadata_dict(response)),
    )


class _StreamedMessageAsync(StreamedMessage):
    """A message that is streamed token by token instead of being returned all at once (async version)."""

    def __init__(self, stream: AsyncIterator[Dict[str, Any]], reply_to: Message):
        self._stream = stream
        self._tokens_raw: List[Dict[str, Any]] = []
        self._metadata: Dict[str, Any] = {}
        self._done = False
        self._full_message = None
        self._reply_to = reply_to
        # TODO Oleksandr: use asyncio.Queue and _Iterator approach just like you did in MessageBundle in order to
        #  ensure that the stream can have multiple consumers

    def get_full_message(self) -> Message:
        raise NotImplementedError("Use aget_full_message()")

    async def aget_full_message(self) -> Message:
        if not self._done:
            # first, make sure that all the tokens are received
            async for _ in self:
                pass

        if not self._full_message:
            self._full_message = await self._reply_to.areply(
                content="".join([self._token_raw_to_text(token_raw) for token_raw in self._tokens_raw]),
                metadata=Metadata(**self._metadata),
            )
        return self._full_message

    def __next__(self) -> Token:
        raise NotImplementedError('Use "async for", anext() or __anext__()')

    async def __anext__(self) -> Token:
        token_text = None
        try:
            while not token_text:
                token_raw = await self._stream.__anext__()
                # pprint(token_raw)
                # print()
                self._tokens_raw.append(token_raw)
                self._metadata.update(
                    {k: v for k, v in _build_openai_metadata_dict(token_raw).items() if v is not None}
                )
                token_text = self._token_raw_to_text(token_raw)
        except StopAsyncIteration:
            self._done = True
            raise

        return Token(text=token_text)

    @staticmethod
    def _token_raw_to_text(token_raw: Dict[str, Any]) -> str:
        return token_raw["choices"][0]["delta"].get("content") or ""


def _build_openai_metadata_dict(openai_response: Dict[str, Any]) -> Dict[str, Any]:
    result = _build_openai_dict(openai_response, skip_keys={"choices", "usage"})
    result.update(_build_openai_dict(openai_response.get("usage", {}), key_suffix="usage"))
    result.update(_build_openai_dict(openai_response["choices"][0], skip_keys={"index", "message", "delta"}))
    result.update(_build_openai_dict(openai_response["choices"][0].get("delta", {}), skip_keys={"content"}))
    result.update(_build_openai_dict(openai_response["choices"][0].get("message", {}), skip_keys={"content"}))
    return result


def _build_openai_dict(
    openai_response: Dict[str, Any], key_suffix: str = "", skip_keys: Set[str] = ()
) -> Dict[str, Any]:
    if key_suffix:
        key_suffix += "_"
    return {f"openai_{key_suffix}{k}": v for k, v in openai_response.items() if k not in skip_keys}
