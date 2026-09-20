from __future__ import annotations
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from . import store
from .config import TELEGRAM_BOT_TOKEN, TTS_VOICE

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("history_bot")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

WELCOME = (
    "حياك الله.\n\n"
    "أنا بوت قصص الأطفال: كل ثلاث ساعات قصة عربية مشوّقة بطابع أصيل (مغامرة وتشويق خفيف).\n\n"
    "التنويع: حيوانات، أساطير، قصص بشرية، وقصص تاريخية للأطفال.\n"
    "كل 4 قصص: 3 للأعمار 8+ (أطول قليلاً) وقصة لمن دون السابعة.\n"
    "عنوان كل قصة = اسم بطلها أو بطلتها (عربي)، بدون تكرار الأسماء.\n"
    f"النص فصحى مع تشكيل كامل، يُرسل كملف .txt ثم صوت الراوية {TTS_VOICE} "
    "(أنثى — نبرة هادئة مستقرة، مخارج واضحة، سرعة معتدلة ثابتة)، طول الصوت تقريباً 3–6 دقائق، بدون موسيقى.\n"
    "جرّب: /now — للإيقاف: /stop"
)

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
    await update.message.reply_text("جاري تجهيز قصة أطفال معرّبة… قد يستغرق دقيقة أو أكثر.")
    try:
        from .pipeline import send_one_to
        title = await send_one_to([update.effective_user.id])
        if not title:
            await update.message.reply_text("تعذر تجهيز القصة الآن. جرّب بعد قليل.")
    except Exception:
        log.exception("send_one failed")
        await update.message.reply_text("تعذر تجهيز القصة الآن. جرّب بعد قليل.")

def build_app() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN missing")
    store.init()
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("now", cmd_now))
    return app

def main() -> None:
    app = build_app()
    try:
        from .scheduler import attach_scheduler
        attach_scheduler(app)
    except Exception:
        log.exception("scheduler not attached yet")
    log.info("kids stories bot polling")
    app.run_polling(drop_pending_updates=True, allowed_updates=["message"])

if __name__ == "__main__":
    main()
