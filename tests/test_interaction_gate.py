from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

from mental_state_bot.services.interaction_gate import serialized_user_interaction


async def test_user_interaction_gate_serializes_same_user() -> None:
    events: list[str] = []
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    user = SimpleNamespace(id=uuid4())

    @serialized_user_interaction
    async def action(*, user, label: str) -> None:
        events.append(f"start:{label}")
        if label == "first":
            first_entered.set()
            await release_first.wait()
        events.append(f"end:{label}")

    first = asyncio.create_task(action(user=user, label="first"))
    await first_entered.wait()
    second = asyncio.create_task(action(user=user, label="second"))
    await asyncio.sleep(0)

    assert events == ["start:first"]
    release_first.set()
    await asyncio.gather(first, second)
    assert events == ["start:first", "end:first", "start:second", "end:second"]
