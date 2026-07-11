from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

_user_locks: dict[str, asyncio.Lock] = {}


def serialized_user_interaction(
    func: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Keep automatic prompts for one user from racing each other in this bot process."""

    @wraps(func)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        user = kwargs.get("user")
        user_id = getattr(user, "id", None)
        if user_id is None:
            return await func(*args, **kwargs)
        lock = _user_locks.setdefault(str(user_id), asyncio.Lock())
        async with lock:
            return await func(*args, **kwargs)

    return wrapped
