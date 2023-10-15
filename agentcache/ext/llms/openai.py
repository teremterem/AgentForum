"""OpenAI API extension for AgentCache."""
from typing import List, AsyncIterator, Dict, Any, Set

from agentcache.errors import AgentCacheError, TokenStreamNotFinishedError
from agentcache.models import Message, StreamedMessage, Token, Metadata
from agentcache.typing import MessageType


async def aopenai_chat_completion(messages: List[MessageType], stream: bool = False, n=1, **kwargs) -> MessageType:
    """Chat with OpenAI models (async version). Returns a message or a stream of tokens."""
    import openai  # pylint: disable=import-outside-toplevel

    if n != 1:
        raise AgentCacheError("Only n=1 is supported by AgentCache for openai.ChatCompletion.acreate()")

    message_dicts = [
        {
            "role": getattr(message.metadata, "openai_role", "user"),
            "content": message.content,
        }
        for message in (
            msg_type.get_full_message() if isinstance(msg_type, StreamedMessage) else msg_type for msg_type in messages
        )
    ]
    # pprint(message_dicts)
    # print("\n")
    response = await openai.ChatCompletion.acreate(messages=message_dicts, stream=stream, **kwargs)

    if stream:
        # noinspection PyTypeChecker
        return _StreamedMessageAsync(token async for token in response)

    # pprint(response)
    # print()
    return Message(
        content=response["choices"][0]["message"]["content"],
        metadata=Metadata(**_build_openai_metadata_dict(response)),
    )


class _StreamedMessageAsync(StreamedMessage):
    """A message that is streamed token by token instead of being returned all at once (async version)."""

    def __init__(self, stream: AsyncIterator[Dict[str, Any]]):
        self._stream = stream
        self._tokens_raw: List[Dict[str, Any]] = []
        self._metadata: Dict[str, Any] = {}
        self._done = False
        self._full_message = None
        # TODO Oleksandr: use asyncio.Queue and _Iterator approach just like you did in MessageBundle ?
        #  (in order to ensure that the producer of the tokens is not blocked by the consumer)

    def get_full_message(self) -> Message:
        if not self._done:
            raise TokenStreamNotFinishedError(
                "Token stream in not finished yet. Either finish reading tokens from the stream first or use "
                "aget_full_message() instead of get_full_message() to finish reading the stream automatically."
            )

        if not self._full_message:
            self._full_message = Message(
                content="".join([self._token_raw_to_text(token_raw) for token_raw in self._tokens_raw]),
                metadata=Metadata(**self._metadata),
            )
        return self._full_message

    async def aget_full_message(self) -> Message:
        # first, make sure that all the tokens are received
        async for _ in self:
            pass

        return self.get_full_message()

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
