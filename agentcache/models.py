"""TODO oleksandr"""
from pydantic import BaseModel


class Cacheable(BaseModel):
    """A base class for objects that can be cached."""

    class Config:
        """Pydantic config."""

        frozen = True
