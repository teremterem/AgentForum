# pylint: disable=wrong-import-position
"""Chat with OpenAI ChatGPT using the AgentCache library."""
import asyncio

from dotenv import load_dotenv

load_dotenv()

from agentcache.models import Message
from agentcache.ext.llms.openai import achatgpt


async def main(print_user_input_again: bool = False) -> None:
    """The chat loop."""
    messages = []
    while True:
        print()
        prompt_text = "YOU: "
        user_input = input(prompt_text)
        if user_input == "exit":
            break
        if print_user_input_again:
            # helps to visualise it in a jupyter notebook, because in a notebook the input happens in a popup
            print()
            print(prompt_text + user_input)
        messages.append(
            Message(
                role="user",
                content=user_input,
            )
        )
        print()
        print("GPT: ", end="", flush=True)
        response = await achatgpt(messages=messages, model="gpt-3.5-turbo-0613", stream=True)
        async for token in response:
            print(token.text, end="", flush=True)
        messages.append(response.get_full_message())
        print()


if __name__ == "__main__":
    asyncio.run(main())
