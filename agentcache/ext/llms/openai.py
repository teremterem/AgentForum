"""OpenAI API extension for AgentCache."""
import asyncio
from typing import List, Dict, Any, Set, Union, Optional

from agentcache.errors import AgentCacheError
from agentcache.forum import MessagePromise, Forum
from agentcache.models import Token, Message
from agentcache.utils import Sentinel


async def aopenai_chat_completion(
    forum: Forum,
    prompt: List[Union[MessagePromise, Message]],  # TODO Oleksandr: support more variants ?
    reply_to: Optional[MessagePromise] = None,
    stream: bool = False,
    n: int = 1,
    **kwargs,
) -> MessagePromise:
    """Chat with OpenAI models. Returns a message or a stream of tokens."""
    import openai  # pylint: disable=import-outside-toplevel

    if n != 1:
        raise AgentCacheError("Only n=1 is supported by AgentCache for openai.ChatCompletion.acreate()")

    messages = [await msg.aget_full_message() if isinstance(msg, MessagePromise) else msg for msg in prompt]
    message_dicts = [
        {
            "role": getattr(msg.metadata, "openai_role", "user"),
            "content": msg.content,
        }
        for msg in messages
    ]
    response = await openai.ChatCompletion.acreate(messages=message_dicts, stream=stream, **kwargs)

    if stream:
        message_promise = _OpenAIStreamedMessage(forum=forum, reply_to=reply_to)

        async def _send_tokens() -> None:
            # TODO Oleksandr: what if an exception occurs in this coroutine ?
            #  how to convert it into an ErrorMessage at this point ?
            with message_promise:
                async for token_raw in response:
                    message_promise.send(token_raw)

        asyncio.create_task(_send_tokens())
        return message_promise

    # TODO Oleksandr: cover this case with a unit test ?
    # TODO Oleksandr: don't wait for the response, return an unfulfilled "MessagePromise" instead ?
    return await forum.anew_message(
        forum=forum,
        content=response["choices"][0]["message"]["content"],
        reply_to=reply_to,
        **_build_openai_metadata_dict(response),
    )


class _OpenAIStreamedMessage(MessagePromise[Dict[str, Any]]):
    """A message that is streamed token by token from openai.ChatCompletion.acreate()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokens_raw: List[Dict[str, Any]] = []

    async def _aget_item_from_queue(self) -> Union[Dict[str, Any], Sentinel]:
        while True:
            token_raw = await self._queue.get()
            if isinstance(token_raw, Sentinel):
                # if Sentinel, return it immediately
                return token_raw

            self._tokens_raw.append(token_raw)
            # TODO Oleksandr: postpone compiling metadata until all tokens are collected and the full msg is built
            self._metadata.update({k: v for k, v in _build_openai_metadata_dict(token_raw).items() if v is not None})

            if self._token_raw_to_text(token_raw):
                # we found a token that actually has some text - return it
                return token_raw

    def _convert_item(self, item: Dict[str, Any]) -> Token:
        return Token(text=self._token_raw_to_text(item))

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
