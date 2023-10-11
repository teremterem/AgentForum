"""TODO Oleksandr"""


async def achatgpt(**kwargs):
    """TODO Oleksandr"""
    import openai  # pylint: disable=import-outside-toplevel

    async for token in await openai.ChatCompletion.acreate(**kwargs, stream=True):
        yield token
