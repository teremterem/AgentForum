# pylint: disable=wrong-import-position
"""Chat with OpenAI ChatGPT using the AgentCache library."""
import asyncio

# noinspection PyUnresolvedReferences
import readline  # pylint: disable=unused-import
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import promptlayer  # TODO Oleksandr: make this optional

from agentcache.ext.llms.openai import aopenai_chat_completion
from agentcache.forum import Forum, InteractionContext
from agentcache.promises import MessagePromise
from agentcache.storage import InMemoryStorage

forum = Forum(immutable_storage=InMemoryStorage())


@forum.agent
async def first_openai_agent(request: MessagePromise, ctx: InteractionContext, **kwargs) -> None:
    """The first agent that uses OpenAI ChatGPT. It sends the full chat history to the OpenAI API."""
    full_chat = await request.aget_history()

    first_response = await aopenai_chat_completion(
        forum=request.forum, prompt=full_chat, in_reply_to=full_chat[-1], openai_module=promptlayer.openai, **kwargs
    )
    ctx.respond(first_response)

    # second_response = await aopenai_chat_completion(
    #     forum=request.forum, prompt=full_chat, in_reply_to=first_response, openai_module=promptlayer.openai, **kwargs
    # )
    # ctx.respond(second_response)


@forum.agent
async def user_proxy_agent(request: MessagePromise, ctx: InteractionContext) -> None:
    """An agent that acts as a proxy between the user and other agents."""
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
    latest_message: Optional[MessagePromise] = forum.new_message_promise(
        content="Hi, how are you doing?",
        sender_alias=first_openai_agent.agent_alias,
        openai_role="assistant",
    )
    try:
        while True:
            user_responses = user_proxy_agent.get_responses(latest_message)
            latest_message = await user_responses.aget_concluding_message()

            assistant_responses = first_openai_agent.get_responses(
                latest_message,
                # model="gpt-3.5-turbo-0613",
                model="gpt-3.5-turbo",
                # model="gpt-4-0613",
                # model="gpt-4",
                stream=True,
            )
            latest_message = await assistant_responses.aget_concluding_message()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    asyncio.run(main())
