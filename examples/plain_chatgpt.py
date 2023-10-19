# pylint: disable=wrong-import-position
"""Chat with OpenAI ChatGPT using the AgentCache library."""
import asyncio

from dotenv import load_dotenv

load_dotenv()

from agentcache.agents import afirst_agent
from agentcache.message_tree import MessageTree
from agentcache.model_wrappers import AsyncMessageBundle
from agentcache.models import Freeform
from agentcache.storage import InMemoryStorage


async def main() -> None:
    """The chat loop."""
    message_tree = MessageTree(immutable_storage=InMemoryStorage())
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
            request_bundle = AsyncMessageBundle(  # TODO Oleksandr: move this to the framework level
                bundle_metadata=Freeform(
                    model="gpt-3.5-turbo-0613",
                    stream=True,
                ),
                items_so_far=[message],
                completed=True,
            )
            response_bundle = await afirst_agent(request_bundle)

            async for streamed_message in response_bundle:
                print("\nGPT: ", end="", flush=True)
                async for token in streamed_message:
                    print(token.text, end="", flush=True)
                print()
            message = await (await response_bundle.aget_all())[-1].aget_full_message()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    asyncio.run(main())
