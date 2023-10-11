"""TODO Oleksandr"""


async def achatgpt(**kwargs):
    """TODO Oleksandr"""
    import openai  # pylint: disable=import-outside-toplevel

    return await openai.ChatCompletion.acreate(**kwargs)
