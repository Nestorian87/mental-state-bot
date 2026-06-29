from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo


def utc_now() -> datetime:
    return datetime.now(tz=ZoneInfo("UTC"))


def local_now(timezone: str) -> datetime:
    return utc_now().astimezone(ZoneInfo(timezone))


def local_date(timezone: str) -> date:
    return local_now(timezone).date()


def parse_hhmm(value: str) -> time:
    hour, minute = value.split(":", maxsplit=1)
    return time(hour=int(hour), minute=int(minute))
