"""Utility functions and classes for the AgentCache framework."""


class Sentinel:
    """A sentinel object used pass special values through queues indicating things like "end of queue" etc."""


END_OF_QUEUE = Sentinel()
