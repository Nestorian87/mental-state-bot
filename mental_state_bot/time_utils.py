from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


def zoneinfo(timezone: str) -> ZoneInfo:
    return ZoneInfo(timezone)


def utc_now() -> datetime:
    return datetime.now(tz=zoneinfo("UTC"))


def local_now(timezone: str) -> datetime:
    return utc_now().astimezone(zoneinfo(timezone))


def local_date(timezone: str) -> date:
    return local_now(timezone).date()


def journal_date(timezone: str, *, active_start: str, now: datetime | None = None) -> date:
    local = (now or utc_now()).astimezone(zoneinfo(timezone))
    target_date = local.date()
    if local.time() < parse_hhmm(active_start):
        return target_date - timedelta(days=1)
    return target_date


def parse_hhmm(value: str) -> time:
    hour, minute = value.split(":", maxsplit=1)
    return time(hour=int(hour), minute=int(minute))
