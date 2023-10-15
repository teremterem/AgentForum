# 🕵️ AgentCache

It is highly encouraged that the agents are stateless (rely only on the incoming messages and the chat history)
because this simplifies "message replay" from arbitrary point in the agent interaction history. _(What about the
possibility for an Agent to have memory that is not implemented as plain chat history, though?)_

## 🔧 Implementation details

This framework is async-first, and uses the [asyncio](https://docs.python.org/3/library/asyncio.html) library to
achieve concurrency. The classes also have synchronous versions of the methods here and there, but the full potential
of the framework is only unlocked when using the async methods. It is designed to support Python 3.8 or higher.
