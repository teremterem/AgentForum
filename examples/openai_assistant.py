# pylint: disable=wrong-import-position,import-outside-toplevel
"""Chat with OpenAI ChatGPT using the AgentForum library."""
import asyncio

# noinspection PyUnresolvedReferences
import readline  # pylint: disable=unused-import
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI
from openai.types.beta import Thread, Assistant

from agentforum.forum import Forum, InteractionContext, ConversationTracker
from agentforum.storage import InMemoryStorage

forum = Forum(immutable_storage=InMemoryStorage())

async_openai_client = AsyncOpenAI()

conversation: ConversationTracker
thread: Thread
assistant: Assistant
latest_oai_msg_id: Optional[str] = None


@forum.agent
async def openai_assistant(ctx: InteractionContext) -> None:
    """The first agent that uses OpenAI ChatGPT. It sends the full chat history to the OpenAI API."""
    # TODO Oleksandr: resolve OpenAI Thread based on the descriptor of the current ConversationTracker
    global latest_oai_msg_id
    for msg in await ctx.request_messages.amaterialize_all():
        openai_msg = await async_openai_client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=msg.content,
        )
        latest_oai_msg_id = openai_msg.id
    run = await async_openai_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        # instructions="Please address the user as Jane Doe. The user has a premium account."
    )
    for _ in range(100):
        await asyncio.sleep(1)
        run = await async_openai_client.beta.threads.runs.retrieve(run_id=run.id, thread_id=thread.id)
        if run.status == "completed":
            assistant_messages = await async_openai_client.beta.threads.messages.list(
                thread_id=thread.id, before=latest_oai_msg_id
            )
            latest_oai_msg_id = assistant_messages.last_id
            # print()
            # print()
            # pprint(assistant_messages.model_dump())
            # print()
            # print()
            for assistant_message in reversed(assistant_messages.data):
                ctx.respond(assistant_message.content[0].text.value)
            return

    ctx.respond("Request timed out.")


@forum.agent
async def user_proxy_agent(ctx: InteractionContext) -> None:
    """An agent that acts as a proxy between the user and other agents."""
    async for request in ctx.request_messages:
        print("\n\033[1m\033[36mGPT: ", end="", flush=True)
        async for token in request:
            print(token.text, end="", flush=True)
        print("\033[0m")

    user_input = input("\nYOU: ")
    if user_input == "exit":
        raise KeyboardInterrupt
    ctx.respond(user_input)


async def main() -> None:
    """The chat loop."""
    global conversation, thread, assistant

    conversation = ConversationTracker(forum)
    thread = await async_openai_client.beta.threads.create()
    assistant = await async_openai_client.beta.assistants.retrieve("asst_Gy7dZUh9IYNLfz8xOtUaJSbC")

    assistant_responses = []
    try:
        while True:
            user_requests = user_proxy_agent.quick_call(assistant_responses)

            # the following line is needed in order to wait until the previous back-and-forth is processed
            # (otherwise back-and-forth-s will be perpetually scheduled but never executed)
            # TODO Oleksandr: how to turn this hack into something more elegant ?
            await user_requests.amaterialize_all()

            assistant_responses = openai_assistant.quick_call(user_requests)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    asyncio.run(main())
