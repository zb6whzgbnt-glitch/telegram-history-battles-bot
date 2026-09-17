from __future__ import annotations
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from telegram.ext import Application

from .config import SEND_HOURS, TZ_NAME
from . import store

log = logging.getLogger("history_bot.scheduler")

async def _job(app: Application) -> None:
    ids = store.subscriber_ids()
    if not ids:
        log.info("no subscribers, skip slot")
        return
    from .pipeline import send_one_to
    import asyncio
    title = None
    for attempt in range(1, 4):
        try:
            title = await send_one_to(ids, bot=app.bot)
            if title:
                break
            log.warning("slot attempt %s produced no story; retrying", attempt)
        except Exception:
            log.exception("scheduled send failed attempt=%s", attempt)
        if attempt < 3:
            await asyncio.sleep(20 * attempt)
    log.info("slot sent story=%s users=%s", title, len(ids))

def attach_scheduler(app: Application) -> None:
    tz = ZoneInfo(TZ_NAME)
    scheduler = AsyncIOScheduler(timezone=tz)

    async def start_sched() -> None:
        if scheduler.running:
            return
        for hour, minute in SEND_HOURS:
            scheduler.add_job(
                _job,
                CronTrigger(hour=hour, minute=minute, timezone=tz),
                args=[app],
                id=f"battle-{hour:02d}{minute:02d}",
                replace_existing=True,
            )
        scheduler.start()
        log.info("scheduler started tz=%s slots=%s now=%s", TZ_NAME, SEND_HOURS, datetime.now(tz).isoformat())

    async def stop_sched() -> None:
        if scheduler.running:
            scheduler.shutdown(wait=False)

    app.post_init = _chain(app.post_init, start_sched)
    app.post_shutdown = _chain(app.post_shutdown, stop_sched)

def _chain(existing, extra):
    async def combined(*args, **kwargs):
        if existing:
            await existing(*args, **kwargs)
        await extra()
    return combined
