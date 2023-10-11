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
        response = await achatgpt(messages=messages, model="gpt-3.5-turbo-0613", stream=True)
        # print()
        # pprint(response)
        print()
        print("GPT: ", end="", flush=True)
        response_msg = []
        async for token in response:
            token_str = token["choices"][0]["delta"].get("content", "")
            print(token_str, end="", flush=True)
            # print()
            # pprint(token)
            response_msg.append(token_str)
        print()
        messages.append(
            Message(
                role="assistant",
                content="".join(response_msg),
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
