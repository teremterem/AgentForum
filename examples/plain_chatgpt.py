# pylint: disable=wrong-import-position
"""Chat with OpenAI ChatGPT using the AgentCache library."""
import asyncio
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from agentcache.ext.llms.openai import aopenai_chat_completion
from agentcache.forum import StreamedMessage, Forum, MessageSequence
from agentcache.storage import InMemoryStorage

forum = Forum(immutable_storage=InMemoryStorage())


@forum.agent
async def first_openai_agent(request: StreamedMessage, response: MessageSequence, **kwargs) -> None:
    """The first agent that uses OpenAI ChatGPT. It sends the full chat history to the OpenAI API."""
    full_chat = await request.aget_full_chat()
    response.send(
        # TODO Oleksandr: try "slipping" PromptLayer in
        await aopenai_chat_completion(forum=request.forum, prompt=full_chat, reply_to=full_chat[-1], **kwargs)
    )


async def main() -> None:
    """The chat loop."""
    latest_message: Optional[StreamedMessage] = None
    try:
        while True:
            user_input = input("\nYOU: ")
            if user_input == "exit":
                raise KeyboardInterrupt

            # TODO Oleksandr: move this inside the acall_agent_draft() ?
            latest_message = await forum.anew_message(content=user_input, reply_to=latest_message)

            responses = await first_openai_agent.acall(
                request=latest_message,
                model="gpt-3.5-turbo-0613",
                stream=True,
            )
            async for streamed_message in responses:
                print("\nGPT: ", end="", flush=True)
                async for token in streamed_message:
                    print(token.text, end="", flush=True)
                print()
            latest_message = await responses.aget_concluding_message()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    asyncio.run(main())
