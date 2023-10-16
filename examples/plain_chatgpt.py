# pylint: disable=wrong-import-position
"""Chat with OpenAI ChatGPT using the AgentCache library."""
import asyncio

from dotenv import load_dotenv

from agentcache.models import MessageBundle, Metadata, Message

load_dotenv()

from agentcache.agents import AgentFirstDraft


async def main() -> None:
    """The chat loop."""
    agent = AgentFirstDraft()
    try:
        while True:
            user_input = input("\nYOU: ")
            if user_input == "exit":
                raise KeyboardInterrupt

            request_bundle = MessageBundle(  # TODO Oleksandr: move this to the framework level
                bundle_metadata=Metadata(
                    model="gpt-3.5-turbo-0613",
                    stream=True,
                ),
                messages_so_far=[Message(content=user_input)],
                complete=True,
            )
            response = await agent.arun(request_bundle)

            async for message in response:
                print("\nGPT: ", end="", flush=True)
                async for token in message:
                    print(token.text, end="", flush=True)
                print()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    asyncio.run(main())
