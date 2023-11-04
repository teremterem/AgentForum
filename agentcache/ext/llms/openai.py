"""OpenAI API extension for AgentCache."""
import asyncio
from typing import List, Dict, Any, Set, Union, Optional

from agentcache.errors import AgentCacheError
from agentcache.forum import Forum
from agentcache.models import Token, Message
from agentcache.promises import MessagePromise, StreamedMsgPromise
from agentcache.utils import Sentinel


async def aopenai_chat_completion(  # pylint: disable=too-many-arguments
    forum: Forum,
    prompt: List[Union[MessagePromise, Message]],  # TODO Oleksandr: support more variants ?
    sender_alias: Optional[str] = None,
    in_reply_to: Optional[MessagePromise] = None,
    openai_module: Optional[Any] = None,
    stream: bool = False,
    n: int = 1,
    **kwargs,
) -> MessagePromise:
    """Chat with OpenAI models. Returns a message or a stream of tokens."""
    if not openai_module:
        import openai  # pylint: disable=import-outside-toplevel

        openai_module = openai

    if n != 1:
        raise AgentCacheError("Only n=1 is supported by AgentCache for openai.ChatCompletion.acreate()")

    messages = [await msg.amaterialize() if isinstance(msg, MessagePromise) else msg for msg in prompt]
    # pprint(messages)
    message_dicts = [
        {
            "role": getattr(msg.metadata, "openai_role", "user"),
            "content": msg.content,
        }
        for msg in messages
    ]
    # pprint(message_dicts)
    response = await openai_module.ChatCompletion.acreate(messages=message_dicts, stream=stream, **kwargs)

    if stream:
        message_promise = _OpenAIStreamedMessage(
            forum=forum,
            # TODO Oleksandr: is this a bad place for sender alias resolution ? where to move it ?
            sender_alias=forum.resolve_sender_alias(sender_alias),
            in_reply_to=in_reply_to,
        )

        async def _send_tokens() -> None:
            # TODO Oleksandr: what if an exception occurs in this coroutine ?
            #  how to convert it into an ErrorMessage at this point ?
            with message_promise:
                async for token_raw in response:
                    # noinspection PyProtectedMember
                    message_promise._send(token_raw)  # pylint: disable=protected-access
            # # TODO Oleksandr: do we need the following ?
            # await message_promise.amaterialize()  # let's save the message in the storage

        asyncio.create_task(_send_tokens())
        return message_promise

    # TODO Oleksandr: cover this case with a unit test ?
    # TODO Oleksandr: don't wait for the response, return an unfulfilled "MessagePromise" instead ?
    return forum.new_message_promise(
        content=response["choices"][0]["message"]["content"],
        # TODO Oleksandr: is this a bad place for sender alias resolution ?
        sender_alias=forum.resolve_sender_alias(sender_alias),
        in_reply_to=in_reply_to,
        **_build_openai_metadata_dict(response),
    )


class _OpenAIStreamedMessage(StreamedMsgPromise):
    """A message that is streamed token by token from openai.ChatCompletion.acreate()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokens_raw: List[Dict[str, Any]] = []

    async def _aget_item_from_queue(self) -> Union[Dict[str, Any], Sentinel, BaseException]:
        while True:
            token_raw = await self._queue.get()
            if isinstance(token_raw, Sentinel):
                # if Sentinel, return it immediately
                return token_raw

            self._tokens_raw.append(token_raw)
            # TODO Oleksandr: postpone compiling metadata until all tokens are collected and the full msg is built
            for k, v in _build_openai_metadata_dict(token_raw).items():
                if v is not None:
                    self._metadata[k] = v

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
