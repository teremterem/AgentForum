# pylint: disable=wrong-import-position
"""Chat with OpenAI ChatGPT using the AgentCache library."""
import asyncio
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from agentcache.ext.llms.openai import aopenai_chat_completion
from agentcache.forum import MessagePromise, Forum, MessageSequence
from agentcache.storage import InMemoryStorage

forum = Forum(immutable_storage=InMemoryStorage())


@forum.agent
async def first_openai_agent(request: MessagePromise, responses: MessageSequence, **kwargs) -> None:
    """The first agent that uses OpenAI ChatGPT. It sends the full chat history to the OpenAI API."""
    full_chat = await request.aget_history()

    # TODO Oleksandr: try "slipping" PromptLayer in
    first_response = await aopenai_chat_completion(
        forum=request.forum, prompt=full_chat, in_reply_to=full_chat[-1], **kwargs
    )
    # TODO Oleksandr: !!! you broke streaming again !!! introduce the concept of ForwardedMessages !!!
    responses.send((await first_response.amaterialize()).content)

    # second_response = await aopenai_chat_completion(
    #     forum=request.forum, prompt=full_chat, in_reply_to=first_response, **kwargs
    # )
    # responses.send(second_response)


@forum.agent
async def user_proxy_agent(request: MessagePromise, response: MessageSequence) -> None:
    """An agent that acts as a proxy between the user and other agents."""
    print("\nGPT: ", end="", flush=True)
    async for token in request:
        print(token.text, end="", flush=True)
    print()
    user_input = input("\nYOU: ")
    if user_input == "exit":
        raise KeyboardInterrupt
    response.send(user_input)


async def main() -> None:
    """The chat loop."""
    latest_message: Optional[MessagePromise] = forum.new_message_promise(
        content="Hi, how are you doing?",
        sender_alias=first_openai_agent.agent_alias,
        openai_role="assistant",
    )
    try:
        while True:
            user_responses = user_proxy_agent.call(latest_message)
            latest_message = await user_responses.aget_concluding_message()

            assistant_responses = first_openai_agent.call(
                latest_message,
                model="gpt-3.5-turbo-0613",
                # model="gpt-4-0613",
                stream=True,
            )
            latest_message = await assistant_responses.aget_concluding_message()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    asyncio.run(main())
