from __future__ import annotations

from datetime import date, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import User, UserSettings
from mental_state_bot.time_utils import journal_date, zoneinfo


async def current_journal_date(
    session: AsyncSession,
    *,
    user: User,
    user_settings: UserSettings,
    now: datetime | None = None,
) -> date:
    local_now = (now or datetime.now(tz=zoneinfo("UTC"))).astimezone(zoneinfo(user.timezone))
    calendar_date = local_now.date()
    candidate = journal_date(user.timezone, active_start=user_settings.active_start, now=local_now)
    if candidate == calendar_date:
        return candidate

    previous_day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=candidate)
    if previous_day is not None and previous_day.ended_at is not None:
        return calendar_date
    return candidate
