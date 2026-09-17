from __future__ import annotations
import asyncio
import logging
from io import BytesIO

from telegram import Bot
from telegram.error import Forbidden, TelegramError

from . import store
from .config import FEMALE_VOICE, MALE_VOICE, TELEGRAM_BOT_TOKEN
from .gemini_client import arabize_story, synthesize_long
from .stories import pick_unused_story
from .tts import pcm_to_ogg

log = logging.getLogger("history_bot.pipeline")

GENRE_AR = {
    "animals": "حيوانات",
    "legend": "أسطورة",
    "human": "بشرية",
    "history": "تاريخية للأطفال",
}

def _chunk_text(text: str, limit: int = 3900) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    buf = ""
    for para in text.split("\n"):
        candidate = (buf + "\n" + para).strip() if buf else para
        if len(candidate) <= limit:
            buf = candidate
        else:
            if buf:
                parts.append(buf)
            if len(para) <= limit:
                buf = para
            else:
                for i in range(0, len(para), limit):
                    parts.append(para[i : i + limit])
                buf = ""
    if buf:
        parts.append(buf)
    return parts

def _prepare_episode_sync() -> tuple[dict, str, str, bytes, str, dict] | None:
    profile = store.peek_episode_profile()
    story = pick_unused_story(
        store.sent_ids(),
        age_band=profile["age_band"],
        genre=profile["genre"],
        sent_titles=store.sent_title_keys(),
    )
    if not story:
        return None
    # Reserve original English title BEFORE arabize/TTS so it never repeats
    store.reserve_story(story["id"], story.get("title_en", ""))
    gender = store.next_voice_gender()
    title_ar, body_ar = arabize_story(story, age_band=profile["age_band"], genre=profile["genre"])
    pcm, rate = synthesize_long(body_ar, gender)
    ogg = pcm_to_ogg(pcm, rate)
    return story, title_ar, body_ar, ogg, gender, profile

async def send_one_to(user_ids: list[int], bot: Bot | None = None) -> str | None:
    try:
        prepared = await asyncio.wait_for(asyncio.to_thread(_prepare_episode_sync), timeout=700)
    except Exception:
        log.exception("prepare story episode failed")
        return None
    if not prepared:
        return None
    story, title_ar, body_ar, ogg, gender, profile = prepared
    voice_name = FEMALE_VOICE if gender == "female" else MALE_VOICE
    age_ar = "دون 7" if profile["age_band"] == "under7" else "8+"

    own = bot is None
    if bot is None:
        bot = Bot(TELEGRAM_BOT_TOKEN)
        await bot.initialize()

    delivered = 0
    header = (
        f"📖 {title_ar}\n"
        f"🏷️ النوع: {GENRE_AR.get(profile['genre'], profile['genre'])} | العمر: {age_ar}\n"
        f"🎙️ الراوي: {'امرأة' if gender == 'female' else 'رجل'} ({voice_name})\n\n"
    )
    body_chunks = _chunk_text(header + body_ar)
    caption = title_ar[:1000]

    try:
        for uid in user_ids:
            try:
                for chunk in body_chunks:
                    await bot.send_message(chat_id=uid, text=chunk)
                await bot.send_voice(
                    chat_id=uid,
                    voice=BytesIO(ogg),
                    caption=caption,
                    filename=f"{story['id']}.ogg",
                )
                store.log_send(story["id"], uid, True)
                delivered += 1
            except Forbidden:
                store.log_send(story["id"], uid, False, "blocked")
                store.drop_subscriber(uid)
            except TelegramError as exc:
                store.log_send(story["id"], uid, False, str(exc))
                log.warning("send failed user=%s err=%s", uid, exc)
        if delivered:
            store.mark_sent(story["id"], title_en=story.get("title_en", ""))
            store.advance_episode_profile()
            return title_ar
        return None
    finally:
        if own:
            await bot.shutdown()
