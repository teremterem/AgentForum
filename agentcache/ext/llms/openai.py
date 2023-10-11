"""OpenAI API extension for AgentCache."""
from typing import List, AsyncIterator, Dict, Any

from agentcache.models import Message, StreamedMessage, Token
from agentcache.typing import MessageType


async def achatgpt(messages: List[Message], stream: bool = False, **kwargs) -> MessageType:
    """Chat with OpenAI models (async version). Returns a message or a stream of tokens."""
    import openai  # pylint: disable=import-outside-toplevel

    message_dicts = [{"role": message.role, "content": message.content} for message in messages]

    response = await openai.ChatCompletion.acreate(messages=message_dicts, stream=stream, **kwargs)

    if stream:
        # noinspection PyTypeChecker
        return _StreamedMessageAsync(token async for token in response)

    return Message(
        # TODO Oleksandr: support all "choices", not only the first one
        content=response["choices"][0]["message"]["content"],
        role=response["choices"][0]["message"]["role"],
    )


class _StreamedMessageAsync(StreamedMessage):
    """A message that is streamed token by token instead of being returned all at once (async version)."""

    def __init__(self, stream: AsyncIterator[Dict[str, Any]]):
        self._stream = stream
        self._tokens_raw: List[Dict[str, Any]] = []
        self._role: str = ""
        self._done = False
        self._full_message = None

    def get_full_message(self) -> Message:
        if not self._done:
            # TODO Oleksandr: use a custom error class
            raise ValueError(
                "Token stream in not finished yet. Either finish reading tokens from the stream first or use "
                "aget_full_message() instead of get_full_message() to finish reading the stream automatically."
            )

        if not self._full_message:
            self._full_message = Message(
                content="".join([self._token_raw_to_text(token_raw) for token_raw in self._tokens_raw]),
                role=self._role,
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
                self._tokens_raw.append(token_raw)

                # TODO Oleksandr: support all "choices", not only the first one
                role = token_raw["choices"][0]["delta"].get("role")
                if role:
                    self._role = role
                token_text = self._token_raw_to_text(token_raw)
        except StopAsyncIteration:
            self._done = True
            raise

        return Token(text=token_text)

    @staticmethod
    def _token_raw_to_text(token_raw: Dict[str, Any]) -> str:
        # TODO Oleksandr: support all "choices", not only the first one
        return token_raw["choices"][0]["delta"].get("content") or ""
