# pylint: disable=wrong-import-position
"""Delete me after testing."""
import asyncio

from dotenv import load_dotenv

load_dotenv()

from agentcache.models import Message
from agentcache.ext.llms.openai import achatgpt


async def main():
    """Delete me after testing."""
    messages = []
    while True:
        print()
        messages.append(
            Message(
                role="user",
                content=input("YOU: "),
            )
        )
        print()
        print("GPT: ", end="", flush=True)
        response = await achatgpt(messages=messages, model="gpt-3.5-turbo-0613", stream=True)
        async for token in response:
            print(token.text, end="", flush=True)
        print()
        messages.append(
            Message(
                role="assistant",
                content=response.get_full_message().content,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
