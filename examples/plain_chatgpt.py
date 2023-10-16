# pylint: disable=wrong-import-position
"""Chat with OpenAI ChatGPT using the AgentCache library."""
import asyncio

from dotenv import load_dotenv

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

            response = await agent.arun(user_input, model="gpt-3.5-turbo-0613", stream=True)

            print("\nGPT: ", end="", flush=True)
            async for token in response:
                print(token.text, end="", flush=True)
            print()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    asyncio.run(main())
