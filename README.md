# 🏛 AgentForum

An asynchronous framework for building LLM-based multi-agent systems in Python, with a focus on:

- **Message trees which highly resemble commits in git to facilitate backtracking.** Messages are immutable and are
  identified by hashes based on their content and their history (a hash of the previous message). A single hash
  represents state of a conversation at a given point in time.
- **Message promises and the possibility of token streaming.** `MessagePromise` objects are returned before the
  messages are fully generated. Token streaming is supported regardless of the number of nested levels of agents.

*NOTE: While this framework is designed with LLMs in mind, it can be used with any kind of agents.*

## 🧩 Features

- **Message forwarding.** Messages can be forwarded to become a part of different message branches or
  trees (much like it is done in messaging apps for humans).

## 💡 Philosophy

- **Agents run as concurrently as possible.** When an agent is called with `.call()` or `.quick_call()` a task is
  scheduled and `AgentCall` or `AsyncMessageSequence` objects are returned immediately. The actual processing will
  happen upon a task switch.
- **Stateless agents are encouraged.** This is not a hard requirement, but ideally the agents should produce the same
  output when they are given the same input (the same sequence of messages from a message branch).

## 🔧 Implementation details

This framework supports **Python 3.9 or higher** and uses [asyncio](https://docs.python.org/3/library/asyncio.html)
under the hood.

## 🌱 Future plans

- **Message replies.** Technically this will be very similar to message forwarding. The difference will mostly be
  semantic (in the spirit of mimicking messaging apps for humans).
- **Exceptions as part of the message tree.** In the future, exceptions raised by agents will be represented as a
  special type of messages and will be part of the message branches those agents were on when they were raised.
- **Optional caching of agent responses (enabled by default).** When the same sequence of messages is sent to an agent,
  the framework will respond with the same sequence of response messages without actually calling the agent.
- **Cancellation of agent execution.** In the future, it will be possible to cancel the execution of agents before
  they finish.

***⚠️ NOTE: These plans are tentative and may change in the future. ⚠️***
