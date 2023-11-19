"""OpenAI API extension for AgentForum."""
import asyncio
from typing import List, Dict, Any, Set, Union, Optional, AsyncIterator

from pydantic import BaseModel

from agentforum.errors import AgentForumError
from agentforum.models import Message, ContentChunk
from agentforum.promises import MessagePromise, StreamedMessage


def openai_chat_completion(
    # TODO Oleksandr: switch `prompt` to a combination of MessageType and a list of dicts ?
    prompt: List[Union[MessagePromise, Message]],
    async_openai_client: Optional[Any] = None,
    stream: bool = False,
    n: int = 1,
    **kwargs,
) -> StreamedMessage:
    """Chat with OpenAI models."""
    if not async_openai_client:
        from openai import AsyncOpenAI  # pylint: disable=import-outside-toplevel

        # TODO Oleksandr: move client initialization to the module level ?
        async_openai_client = AsyncOpenAI()

    if n != 1:
        raise AgentForumError("Only n=1 is supported by AgentForum for AsyncOpenAI().chat.completions.create()")

    streamed_message = _OpenAIStreamedMessage()

    async def _make_request() -> None:
        # pylint: disable=protected-access
        # noinspection PyProtectedMember
        with _OpenAIStreamedMessage._Producer(streamed_message) as token_producer:
            messages = [await msg.amaterialize() if isinstance(msg, MessagePromise) else msg for msg in prompt]
            message_dicts = [
                {
                    # TODO Oleksandr: introduce a lambda function to derive roles from messages
                    "role": getattr(msg.metadata, "openai_role", "user"),
                    "content": msg.content,
                }
                for msg in messages
            ]

            # noinspection PyTypeChecker
            response = await async_openai_client.chat.completions.create(
                messages=message_dicts, stream=stream, n=n, **kwargs
            )
            if stream:
                async for token_raw in response:
                    token_producer.send(token_raw)
            else:
                # send the whole response as a single "token"
                token_producer.send(response)

    asyncio.create_task(_make_request())

    return streamed_message


class _OpenAIStreamedMessage(StreamedMessage[BaseModel]):
    """A message that is streamed token by token from openai.ChatCompletion.acreate()."""

    async def _aconvert_incoming_item(
        self, incoming_item: BaseModel
    ) -> AsyncIterator[Union[ContentChunk, BaseException]]:
        try:
            token_text = incoming_item.choices[0].delta.content
        except AttributeError:
            token_text = incoming_item.choices[0].message.content

        if token_text:
            yield ContentChunk(text=token_text)

        # TODO Oleksandr: postpone compiling metadata until all tokens are collected and the full message is built
        for k, v in _build_openai_metadata_dict(incoming_item.model_dump()).items():
            if v is not None:
                self._metadata[k] = v


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
