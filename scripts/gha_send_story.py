"""GitHub Actions entry: process pending Telegram commands briefly, then send one story."""
from __future__ import annotations

import asyncio
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gha_send")


async def process_commands(timeout_s: float = 25.0) -> None:
    """Drain pending updates so /start and /stop work without a 24/7 poller."""
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes

    from history_bot import store
    from history_bot.bot import WELCOME
    from history_bot.config import TELEGRAM_BOT_TOKEN

    store.init()

    async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        store.subscribe(update.effective_user.id)
        await update.message.reply_text(WELCOME)

    async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        store.unsubscribe(update.effective_user.id)
        await update.message.reply_text("توقفت. إذا تبي ترجع، أرسل /start")

    async def cmd_now(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        await update.message.reply_text("جاري تجهيز قصة… قد يستغرق دقيقة أو أكثر.")
        from history_bot.pipeline import send_one_to

        title = await send_one_to([update.effective_user.id])
        if not title:
            await update.message.reply_text("تعذر تجهيز القصة الآن. جرّب بعد قليل.")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("now", cmd_now))

    await app.initialize()
    await app.start()
    try:
        # Process backlog then listen briefly for near-real-time commands
        await app.updater.start_polling(drop_pending_updates=False, allowed_updates=["message"])
        await asyncio.sleep(timeout_s)
        await app.updater.stop()
    finally:
        await app.stop()
        await app.shutdown()


async def send_scheduled() -> str | None:
    from history_bot import store
    from history_bot.pipeline import send_one_to

    store.init()
    ids = store.subscriber_ids()
    extra = os.environ.get("SUBSCRIBER_CHAT_IDS", "").strip()
    if extra:
        for part in extra.split(","):
            part = part.strip()
            if part.isdigit():
                store.subscribe(int(part))
        ids = store.subscriber_ids()
    if not ids:
        log.warning("no subscribers; skip send (users must /start near a scheduled run, or set SUBSCRIBER_CHAT_IDS)")
        return None
    return await send_one_to(ids)


def main() -> None:
    mode = (sys.argv[1] if len(sys.argv) > 1 else "all").strip()
    if mode in ("commands", "all"):
        asyncio.run(process_commands(timeout_s=float(os.environ.get("CMD_POLL_SECONDS", "25"))))
    if mode in ("send", "all"):
        title = asyncio.run(send_scheduled())
        print(title or "nothing sent")


if __name__ == "__main__":
    main()
