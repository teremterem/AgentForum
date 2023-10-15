# pylint: disable=wrong-import-position
"""Chat with OpenAI ChatGPT using the AgentCache library."""
import asyncio

from dotenv import load_dotenv

load_dotenv()

from agentcache.agents import AgentFirstDraft


async def main(echo_input: bool = False) -> None:
    """The chat loop."""
    agent = AgentFirstDraft()
    try:
        while True:
            print()
            user_input = custom_input("YOU: ", echo_input=echo_input)
            print()
            print("GPT: ", end="", flush=True)
            response = await agent.arun(user_input, model="gpt-3.5-turbo-0613", stream=True)
            async for token in response:
                print(token.text, end="", flush=True)
            print()
    except KeyboardInterrupt:
        pass
    print()


def custom_input(prompt_text: str, echo_input: bool = False) -> str:
    """Custom input function that allows to exit the chat by typing "exit"."""
    user_input = input(prompt_text)
    if user_input == "exit":
        raise KeyboardInterrupt("exit")
    if echo_input:
        # helps to visualise it in a jupyter notebook, because in a notebook the input happens in a popup
        print()
        print(prompt_text + user_input)
    return user_input


if __name__ == "__main__":
    asyncio.run(main())
