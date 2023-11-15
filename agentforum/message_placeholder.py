from agentforum.forum import Forum
from agentforum.models import Message, MessageParams
from agentforum.promises import MessagePromise, DetachedMsgPromise


class MessagePlaceholder:
    def __init__(self, forum: Forum, msg_params: MessageParams, do_not_forward_if_possible: bool = True) -> None:
        self.forum = forum
        self.msg_params = msg_params
        self.do_not_forward_if_possible = do_not_forward_if_possible

    def _materialize_message_promise(self) -> "MessagePromise":
        if isinstance(self.msg_params.content, str):
            forward_of = None
            content = self.msg_params.content
        elif isinstance(self.msg_params.content, Message):
            forward_of = MessagePromise(forum=self.forum, materialized_msg=self.msg_params.content)
            # TODO Oleksandr: should we store the materialized_msg ?
            #  (the promise will not store it since it is already "materialized")
            #  or do we trust that something else already stored it ?
            content = ""  # this is a hack (the content will actually be taken from the forwarded message)
        elif isinstance(self.msg_params.content, MessagePromise):  # TODO Oleksandr: MessagePlaceholder
            forward_of = self.msg_params.content
            content = ""  # this is a hack (the content will actually be taken from the forwarded message)
        else:
            raise ValueError(f"Unexpected message content type: {type(self.msg_params.content)}")

        msg_promise = DetachedMsgPromise(
            forum=self.forum,
            branch_from=self._latest_msg_promise,
            forward_of=forward_of,
            detached_msg=Message(
                content=content,
                sender_alias=self.msg_params.sender_alias,
                metadata=self.msg_params.metadata,
            ),
        )
        self._latest_msg_promise = msg_promise
        return msg_promise

    async def aappend_zero_or_more_messages(
        self, content: MessageType, sender_alias: str, do_not_forward_if_possible: bool = True, **metadata
    ) -> AsyncIterator[MessagePlaceholder]:
        # if do_not_forward_if_possible and not self.has_prior_history and not metadata:
        #     # if there is no prior history (and no extra metadata) we can just append the original message
        #     # (or sequence of messages) instead of creating message forwards
        #     # TODO Oleksandr: move this logic into the future MessagePlaceholder class, because right now it works in
        #     #  quite an unpredictable and hard to comprehend way
        #
        #     if isinstance(content, MessageSequence):
        #         async for msg_promise in content:
        #             self._latest_msg_promise = msg_promise
        #             # TODO Oleksandr: other parallel tasks may submit messages to the same conversation which will
        #             #  mess it up (because we are not doing forwards here) - how to protect from this ?
        #             yield msg_promise
        #         return
        #     if isinstance(content, MessagePromise):
        #         self._latest_msg_promise = content
        #         yield content
        #         return
        #     if isinstance(content, Message):
        #         self._latest_msg_promise = MessagePromise(forum=self.forum, materialized_msg=content)
        #         yield self._latest_msg_promise
        #         return

        # if it's not a plain string then it should be forwarded (either because prior history in this conversation
        # should be maintained or because there is extra metadata)

        if isinstance(content, (str, Message, MessagePromise)):
            yield self.new_message_promise(
                content=content,
                sender_alias=sender_alias,
                **metadata,
            )

        elif hasattr(content, "__iter__"):
            for msg in content:
                async for msg_promise in self.aappend_zero_or_more_messages(
                    content=msg,
                    sender_alias=sender_alias,
                    **metadata,
                ):
                    yield msg_promise
        elif hasattr(content, "__aiter__"):
            async for msg in content:
                async for msg_promise in self.aappend_zero_or_more_messages(
                    content=msg,
                    sender_alias=sender_alias,
                    **metadata,
                ):
                    yield msg_promise
        else:
            raise ValueError(f"Unexpected message content type: {type(content)}")
