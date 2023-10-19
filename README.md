# 🌟🤖 AgentCache 🕵🌟

It is highly encouraged that the agents are stateless (rely only on the incoming messages and the chat history)
because this simplifies "message replay" from arbitrary point in the agent interaction history. _(What about the
possibility for an Agent to have memory that is not implemented as plain chat history, though?)_

## 💡 Philosophy

- Inversion of Control for agents.
- Agents are as stateless as possible.
- Agents can receive multiple messages before responding, as well as respond with multiple messages.
- Caching of agent responses, which is necessary for effective Inversion of Control and also simplifies debugging /
  experimenting with agents' internal logic by making message replay inside a complex chain of interactions possible.
- LLM token streaming is supported no matter how many nested levels of agents there are.
- Messages are represented as immutable objects.

## 🔧 Implementation details

This framework is async-first, and uses the [asyncio](https://docs.python.org/3/library/asyncio.html) library to
achieve concurrency. The classes also have synchronous versions of the methods here and there, but the full potential
of the framework is only unlocked when using the async methods. It is designed to support **Python 3.8 or higher**.
