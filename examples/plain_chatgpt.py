# pylint: disable=wrong-import-position
"""Chat with OpenAI ChatGPT using the AgentCache library."""
import asyncio
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from agentcache.agents import acall_agent_draft
from agentcache.model_wrappers import StreamedMessage
from agentcache.models import Message
from agentcache.storage import InMemoryStorage


async def main() -> None:
    """The chat loop."""
    forum = InMemoryStorage()
    latest_message: Optional[StreamedMessage] = None
    try:
        while True:
            user_input = input("\nYOU: ")
            if user_input == "exit":
                raise KeyboardInterrupt

            latest_message = StreamedMessage(  # TODO Oleksandr: move this to the framework level
                forum=forum,
                full_message=Message(content=user_input),
                reply_to=latest_message,
            )
            responses = await acall_agent_draft(
                request=latest_message,
                model="gpt-3.5-turbo-0613",
                stream=True,
            )

            async for streamed_message in responses:
                print("\nGPT: ", end="", flush=True)
                async for token in streamed_message:
                    print(token.text, end="", flush=True)
                print()
            latest_message = (await responses.aget_all())[-1]  # TODO Oleksandr: aget_concluding_message()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    asyncio.run(main())
