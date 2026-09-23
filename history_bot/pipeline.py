from __future__ import annotations
import asyncio
import logging
import re
from io import BytesIO

from telegram import Bot
from telegram.error import Forbidden, TelegramError

from . import store
from .config import TELEGRAM_BOT_TOKEN, TTS_VOICE
from .gemini_client import arabize_story, synthesize_long, _disambiguate_hero_title
from .stories import pick_unused_story
from .tts import pcm_to_ogg

log = logging.getLogger("history_bot.pipeline")

GENRE_AR = {
    "animals": "حيوانات",
    "legend": "أسطورة",
    "human": "بشرية",
    "history": "تاريخية للأطفال",
}


def _safe_filename(title: str, story_id: str) -> str:
    base = re.sub(r"[^\w\u0600-\u06FF\-]+", "_", title, flags=re.UNICODE).strip("_")
    if not base:
        base = story_id
    return f"{base[:80]}.txt"


def _prepare_episode_sync() -> tuple[dict, str, str, bytes, dict] | None:
    """Pick/finalize up to 3 source stories; never ship local-template fallback."""
    from .gemini_client import StoryQualityError

    profile = store.peek_episode_profile()
    sent = set(store.sent_ids())
    sent_titles = set(store.sent_title_keys())
    attempted: set[str] = set()
    last_err: Exception | None = None

    for attempt in range(1, 4):
        story = pick_unused_story(
            sent | attempted,
            age_band=profile["age_band"],
            genre=profile["genre"],
            sent_titles=sent_titles,
        )
        if not story:
            log.error("no unused quality story left (attempt %s/3)", attempt)
            break

        attempted.add(story["id"])
        # Reserve original English title BEFORE arabize/TTS so it never repeats
        store.reserve_story(story["id"], story.get("title_en", ""))
        log.info(
            "prepare episode attempt=%s/3 source=%s title=%s words=%s",
            attempt,
            story.get("source"),
            (story.get("title_en") or "")[:80],
            story.get("words"),
        )
        used_heroes = store.used_hero_keys(limit=400)
        try:
            title_ar, body_ar = arabize_story(
                story,
                age_band=profile["age_band"],
                genre=profile["genre"],
                used_heroes=used_heroes,
            )
        except (StoryQualityError, RuntimeError) as exc:
            last_err = exc
            log.warning(
                "discard source after finalize/quality failure attempt=%s/3 id=%s: %s",
                attempt,
                story["id"],
                str(exc)[:200],
            )
            continue
        except Exception as exc:
            last_err = exc
            log.exception(
                "discard source after unexpected arabize failure attempt=%s/3 id=%s",
                attempt,
                story["id"],
            )
            continue

        # Reserve Arabic hero/title name so concurrent runs cannot collide
        if not store.reserve_hero(title_ar, story["id"]):
            blocked = store.used_hero_normalized_keys()
            for _ in range(6):
                blocked = blocked | {store.normalize_hero_name(title_ar)}
                title_ar = _disambiguate_hero_title(title_ar, body_ar, blocked)
                if store.reserve_hero(title_ar, story["id"]):
                    break
            else:
                log.error("hero reserve failed after disambiguation title=%s", title_ar[:60])
        body_words = len((body_ar.split("المصدر:")[0] if "المصدر:" in body_ar else body_ar).split())
        log.info(
            "episode ready hero_title=%s body_words~%s voice=%s attempt=%s",
            title_ar[:80],
            body_words,
            TTS_VOICE,
            attempt,
        )
        # Always Aoede — never alternate voices mid-story or across episodes
        pcm, rate = synthesize_long(body_ar, TTS_VOICE)
        ogg = pcm_to_ogg(pcm, rate)
        return story, title_ar, body_ar, ogg, profile

    log.error(
        "prepare episode failed after %s attempts last=%s",
        len(attempted),
        str(last_err)[:180] if last_err else "none",
    )
    return None


async def send_one_to(user_ids: list[int], bot: Bot | None = None) -> str | None:
    try:
        prepared = await asyncio.wait_for(asyncio.to_thread(_prepare_episode_sync), timeout=700)
    except Exception:
        log.exception("prepare story episode failed")
        return None
    if not prepared:
        return None
    story, title_ar, body_ar, ogg, profile = prepared
    age_ar = "دون 7" if profile["age_band"] == "under7" else "8+"

    own = bot is None
    if bot is None:
        bot = Bot(TELEGRAM_BOT_TOKEN)
        await bot.initialize()

    delivered = 0
    # Full story as .txt document (with tashkeel); short caption = title
    doc_text = (
        f"{title_ar}\n"
        f"النوع: {GENRE_AR.get(profile['genre'], profile['genre'])} | العمر: {age_ar}\n"
        f"الراوي: {TTS_VOICE} (أنثى — راوية قصص أطفال هادئة ومستقرة)\n\n"
        f"{body_ar}\n"
    )
    filename = _safe_filename(title_ar, story["id"])
    caption = title_ar[:1024]

    try:
        for uid in user_ids:
            try:
                doc_buf = BytesIO(doc_text.encode("utf-8"))
                doc_buf.name = filename
                await bot.send_document(
                    chat_id=uid,
                    document=doc_buf,
                    filename=filename,
                    caption=caption,
                )
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
            store.mark_hero_sent(title_ar, story["id"])
            store.advance_episode_profile()
            return title_ar
        return None
    finally:
        if own:
            await bot.shutdown()
