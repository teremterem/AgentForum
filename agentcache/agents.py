from agentcache.ext.llms.openai import aopenai_chat_completion
from agentcache.model_wrappers import MessageSequence, StreamedMessage
from agentcache.models import Freeform, _AgentCall
from agentcache.storage import ImmutableStorage


async def acall_agent_draft(forum: ImmutableStorage, request: StreamedMessage, **kwargs) -> MessageSequence:
    agent_call = _AgentCall(
        content="agent_alias",  # TODO Oleksandr: pass an actual agent alias here
        metadata=Freeform(**kwargs),
        prev_msg_hash_key=(await request.aget_full_message()).hash_key,
    )
    await forum.astore_immutable(agent_call)
    return await aagent_caller_draft(forum, agent_call)


async def aagent_caller_draft(forum: ImmutableStorage, agent_call: _AgentCall) -> MessageSequence:
    # TODO Oleksandr: should we do asyncio.create_task() somewhere around here ?
    request = StreamedMessage(
        forum=forum,
        full_message=await forum.aretrieve_immutable(agent_call.request_hash_key),
    )
    response = MessageSequence()
    with response:
        await aagent_draft(request, response, **agent_call.kwargs.as_kwargs)
    return response


async def aagent_draft(request: StreamedMessage, response: MessageSequence, **kwargs) -> None:
    full_chat = await request.aget_full_chat()
    response.send(
        await aopenai_chat_completion(forum=request.forum, prompt=full_chat, reply_to=full_chat[-1], **kwargs)
    )
