# pylint: disable=wrong-import-position
"""Chat with OpenAI ChatGPT using the AgentCache library."""
import asyncio

from dotenv import load_dotenv

load_dotenv()

from agentcache.agents import afirst_agent
from agentcache.model_wrappers import MessageSequence
from agentcache.models import Freeform
from agentcache.storage import InMemoryStorage


async def main() -> None:
    """The chat loop."""
    forum = InMemoryStorage()
    message = None
    try:
        while True:
            user_input = input("\nYOU: ")
            if user_input == "exit":
                raise KeyboardInterrupt

            if not message:
                message = await message_tree.anew_message(content=user_input)
            else:
                message = await message.areply(content=user_input)
            requests = MessageSequence(  # TODO Oleksandr: move this to the framework level
                sequence_metadata=Freeform(
                    model="gpt-3.5-turbo-0613",
                    stream=True,
                ),
                items_so_far=[message],
                completed=True,
            )
            responses = await afirst_agent(requests)

            async for streamed_message in responses:
                print("\nGPT: ", end="", flush=True)
                async for token in streamed_message:
                    print(token.text, end="", flush=True)
                print()
            message = await (await responses.aget_all())[-1].aget_full_message()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    asyncio.run(main())
