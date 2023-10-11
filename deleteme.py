# pylint: disable=wrong-import-position
"""Delete me after testing."""
import asyncio
from pprint import pprint

from dotenv import load_dotenv

load_dotenv()

from agentcache.llms.openai import achatgpt


async def main():
    """Delete me after testing."""
    messages = [
        {
            "role": "user",
            "content": "Hello, World!",
        },
    ]
    pprint(await achatgpt(messages=messages, model="gpt-3.5-turbo-0613"))


if __name__ == "__main__":
    asyncio.run(main())
