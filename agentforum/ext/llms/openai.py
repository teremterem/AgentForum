"""OpenAI API extension for AgentForum."""
import asyncio
from typing import Dict, Any, Set, Union, Optional, AsyncIterator, Iterable

from pydantic import BaseModel

from agentforum.errors import AgentForumError
from agentforum.models import Message, ContentChunk
from agentforum.promises import MessagePromise, StreamedMessage


def openai_chat_completion(
    # TODO Oleksandr: allow MessageType ? there should be amaterialize_message_type utility function then
    prompt: Iterable[Union[MessagePromise, Message, Dict[str, Any]]],
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
            message_dicts = [await _message_to_openai_dict(msg) for msg in prompt]

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


async def _message_to_openai_dict(message: Union[MessagePromise, Message, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(message, MessagePromise):
        message = await message.amaterialize()
    if isinstance(message, Message):
        # TODO Oleksandr: introduce a lambda function to derive roles from messages ?
        try:
            # TODO Oleksandr: should `openai_role` take precedence over `role` ? should there even be such a thing as
            #  `openai_role` ?
            role = message.openai_role
        except AttributeError:
            role = getattr(message, "role", "user")
        message = {
            "role": role,
            "content": message.content,
        }
    return message


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

        # TODO Oleksandr: postpone compiling metadata until all tokens are collected and the full message is built ?
        self._update_openai_metadata_dict(incoming_item.model_dump())

    def _update_openai_metadata_dict(self, openai_response: Dict[str, Any]) -> None:
        # TODO Oleksandr: put everything under a single "openai" key instead of "openai_*" for each field separately ?
        self._metadata.update(_build_openai_dict(openai_response, skip_keys={"choices", "usage"}))

        self._metadata.update(
            _build_openai_dict(openai_response["choices"][0], skip_keys={"index", "message", "delta", "logprobs"})
        )
        self._metadata.update(
            _build_openai_dict(openai_response["choices"][0].get("delta", {}), skip_keys={"content"})
        )
        self._metadata.update(
            _build_openai_dict(openai_response["choices"][0].get("message", {}), skip_keys={"content"})
        )

        logprobs = openai_response["choices"][0].get("logprobs")
        if logprobs is not None:
            self._metadata.setdefault("openai_logprobs", []).extend(logprobs["content"])

        usage = openai_response.get("usage")
        if usage is not None:
            self._metadata.setdefault("openai_usage", {}).update({k: v for k, v in usage.items() if v is not None})


def _build_openai_dict(openai_response: Dict[str, Any], skip_keys: Set[str] = ()) -> Dict[str, Any]:
    return {f"openai_{k}": v for k, v in openai_response.items() if v is not None and k not in skip_keys}
