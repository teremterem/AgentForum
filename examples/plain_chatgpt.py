# pylint: disable=wrong-import-position
"""Chat with OpenAI ChatGPT using the AgentCache library."""
import asyncio

# noinspection PyUnresolvedReferences
import readline  # pylint: disable=unused-import

from dotenv import load_dotenv

load_dotenv()

import promptlayer  # TODO Oleksandr: make this optional

from agentcache.ext.llms.openai import aopenai_chat_completion
from agentcache.forum import Forum, InteractionContext, Conversation
from agentcache.storage import InMemoryStorage

forum = Forum(immutable_storage=InMemoryStorage())


@forum.agent
async def first_openai_agent(ctx: InteractionContext, **kwargs) -> None:
    """The first agent that uses OpenAI ChatGPT. It sends the full chat history to the OpenAI API."""
    full_chat = await ctx.request_messages.amaterialize_full_history()

    first_response = await aopenai_chat_completion(
        forum=ctx.forum, prompt=full_chat, branch_from=full_chat[-1], openai_module=promptlayer.openai, **kwargs
    )
    ctx.respond(first_response)

    # second_response = await aopenai_chat_completion(
    #     forum=request.forum, prompt=full_chat, branch_from=first_response, openai_module=promptlayer.openai, **kwargs
    # )
    # ctx.respond(second_response)


@forum.agent
async def user_proxy_agent(ctx: InteractionContext) -> None:
    """An agent that acts as a proxy between the user and other agents."""
    async for request in ctx.request_messages:
        print("\n\033[1m\033[36mGPT: ", end="", flush=True)
        async for token in request:
            print(token.text, end="", flush=True)
        print("\033[0m")

    user_input = input("\nYOU: ")
    if user_input == "exit":
        raise KeyboardInterrupt
    ctx.respond(user_input)


async def main() -> None:
    """The chat loop."""
    conversation = Conversation(forum)
    assistant_responses = [
        conversation.new_message_promise(
            content="Hi, how are you doing?",
            sender_alias=first_openai_agent.agent_alias,
            openai_role="assistant",
        )
    ]
    try:
        while True:
            assistant_responses = first_openai_agent.quick_call(
                user_proxy_agent.quick_call(assistant_responses),
                # model="gpt-4-1106-preview",
                # model="gpt-4",
                model="gpt-3.5-turbo",
                stream=True,
            )
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    asyncio.run(main())
