from agentcache.ext.llms.openai import aopenai_chat_completion
from agentcache.forum import MessageSequence, StreamedMessage


async def acall_agent_draft(request: StreamedMessage, **kwargs) -> MessageSequence:
    return await aagent_caller_draft(
        await request.forum.anew_agent_call(
            agent_alias="agent_alias",  # TODO Oleksandr: pass an actual agent alias here
            request=request,
            **kwargs,
        ),
    )


async def aagent_caller_draft(agent_call: StreamedMessage) -> MessageSequence:
    # TODO Oleksandr: should we do asyncio.create_task() somewhere around here ?
    request = await agent_call.aget_previous_message()
    response = MessageSequence()
    with response:
        await aagent_draft(request, response, **(await agent_call.aget_metadata()).as_kwargs)
    return response


async def aagent_draft(request: StreamedMessage, response: MessageSequence, **kwargs) -> None:
    full_chat = await request.aget_full_chat()
    response.send(
        await aopenai_chat_completion(forum=request.forum, prompt=full_chat, reply_to=full_chat[-1], **kwargs)
    )
