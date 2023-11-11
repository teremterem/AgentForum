"""OpenAI API extension for AgentCache."""
import asyncio
from typing import List, Dict, Any, Set, Union, Optional, AsyncIterator

from agentcache.errors import AgentCacheError
from agentcache.forum import Forum, InteractionContext
from agentcache.models import Token, Message
from agentcache.promises import MessagePromise, StreamedMsgPromise


# noinspection PyProtectedMember
async def aopenai_chat_completion(  # pylint: disable=too-many-arguments,protected-access
    forum: Forum,
    prompt: List[Union[MessagePromise, Message]],
    override_sender_alias: Optional[str] = None,
    openai_module: Optional[Any] = None,
    stream: bool = False,
    n: int = 1,
    **kwargs,
) -> MessagePromise:  # TODO Oleksandr: this function doesn't necessarily need to be async
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

    sender_alias = override_sender_alias or InteractionContext.get_current_sender_alias()

    if stream:
        message_promise = _OpenAIStreamedMessage(forum=forum, sender_alias=sender_alias)

        async def _send_tokens() -> None:
            # noinspection PyProtectedMember
            with _OpenAIStreamedMessage._Producer(message_promise) as token_producer:
                response_ = await openai_module.ChatCompletion.acreate(messages=message_dicts, stream=True, **kwargs)
                async for token_raw in response_:
                    token_producer.send(token_raw)
            # # TODO Oleksandr: do we need the following ?
            # await message_promise.amaterialize()  # let's save the message in the storage

        asyncio.create_task(_send_tokens())
        return message_promise

    # TODO Oleksandr: don't wait for the response, return an unfulfilled "MessagePromise" instead ?
    response = await openai_module.ChatCompletion.acreate(messages=message_dicts, stream=False, **kwargs)
    # TODO Oleksandr: fix it - there is no _new_message_promise() method on Forum anymore
    return forum._new_message_promise(
        content=response["choices"][0]["message"]["content"],
        sender_alias=sender_alias,
        **_build_openai_metadata_dict(response),
    )


class _OpenAIStreamedMessage(StreamedMsgPromise[Dict[str, Any]]):
    """A message that is streamed token by token from openai.ChatCompletion.acreate()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokens_raw: List[Dict[str, Any]] = []

    async def _aconvert_incoming_item(
        self, incoming_item: Dict[str, Any]
    ) -> AsyncIterator[Union[Token, BaseException]]:
        self._tokens_raw.append(incoming_item)
        # TODO Oleksandr: postpone compiling metadata until all tokens are collected and the full msg is built ?
        for k, v in _build_openai_metadata_dict(incoming_item).items():
            if v is not None:
                self._metadata[k] = v

        token_text = incoming_item["choices"][0]["delta"].get("content")
        if token_text:
            yield Token(text=token_text)


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
