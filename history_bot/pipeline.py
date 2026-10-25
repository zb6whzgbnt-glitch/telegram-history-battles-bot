from __future__ import annotations
import asyncio
import logging
import re
from io import BytesIO

from telegram import Bot
from telegram.error import Forbidden, TelegramError

from . import store
from .config import TELEGRAM_BOT_TOKEN, TTS_VOICE
from .gemini_client import arabize_story, synthesize_parts
from .stories import pick_unused_story, tale_type_label
from .tts import pcm_to_ogg
from .story_images import build_scenes_zip, images_enabled

log = logging.getLogger("history_bot.pipeline")

_AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def _safe_filename(title: str, story_id: str) -> str:
    base = re.sub(r"[^\w\u0600-\u06FF\-]+", "_", title, flags=re.UNICODE).strip("_")
    if not base:
        base = story_id
    return f"{base[:80]}.txt"


def _prepare_episode_sync() -> tuple[dict, str, str, list[bytes], dict, dict] | None:
    """Pick/translate up to 3 source stories faithfully; never ship a local template."""
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
        # Reserve original title BEFORE translation/TTS so it never repeats
        store.reserve_story(story["id"], story.get("title_en", ""))
        log.info(
            "prepare episode attempt=%s/3 source=%s title=%s words=%s tale_type=%s",
            attempt,
            story.get("source"),
            (story.get("title_en") or "")[:80],
            story.get("words"),
            story.get("tale_type"),
        )
        # Previously used Arabic character names: only to vary names, never titles.
        used_names = store.used_hero_keys(limit=300)
        try:
            title_ar, body_ar, meta = arabize_story(
                story,
                age_band=profile["age_band"],
                genre=profile["genre"],
                used_heroes=used_names,
            )
        except (StoryQualityError, RuntimeError) as exc:
            last_err = exc
            log.warning(
                "discard source after translation/fidelity failure attempt=%s/3 id=%s: %s",
                attempt,
                story["id"],
                str(exc)[:300],
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

        for name in (meta.get("names") or [])[:8]:
            store.reserve_hero(name, story["id"])
        log.info(
            "episode ready title_ar=%s original_title=%s src_words=%s ar_words=%s ratio=%s voice=%s",
            title_ar[:80],
            (story.get("title_en") or "")[:80],
            meta.get("source_words"),
            meta.get("arabic_words"),
            meta.get("ratio"),
            TTS_VOICE,
        )
        # Always Aoede; long stories → several ordered voice parts (never shortened)
        voices = synthesize_parts(body_ar, TTS_VOICE)
        oggs = [pcm_to_ogg(pcm, rate) for pcm, rate in voices]
        return story, title_ar, body_ar, oggs, profile, meta

    log.error(
        "prepare episode failed after %s attempts last=%s",
        len(attempted),
        str(last_err)[:180] if last_err else "none",
    )
    return None


def build_card(story: dict, title_ar: str, body_ar: str, age_band: str) -> str:
    age_ar = "دون 7" if age_band == "under7" else "8+"
    genre_ar = tale_type_label(story)
    src_line = f"المصدر: {story.get('source', '')}"
    if story.get("title_en"):
        src_line += f" — العنوان الأصلي: {story['title_en']}"
    if story.get("author"):
        src_line += f" — المؤلف: {story['author']}"
    body = body_ar.split("المصدر:")[0].rstrip() if "المصدر:" in body_ar else body_ar.rstrip()
    return (
        f"{title_ar}\n"
        f"النوع: {genre_ar} | العمر: {age_ar}\n"
        f"{src_line}\n"
        f"الراوي: {TTS_VOICE} (أنثى — راوية قصص أطفال هادئة ومستقرة)\n\n"
        f"{body}\n"
    )


async def send_one_to(user_ids: list[int], bot: Bot | None = None) -> str | None:
    try:
        prepared = await asyncio.wait_for(asyncio.to_thread(_prepare_episode_sync), timeout=2100)
    except Exception:
        log.exception("prepare story episode failed")
        return None
    if not prepared:
        return None
    story, title_ar, body_ar, oggs, profile, meta = prepared

    own = bot is None
    if bot is None:
        bot = Bot(TELEGRAM_BOT_TOKEN)
        await bot.initialize()

    delivered = 0
    # Full story as .txt document (with tashkeel); caption = translated title
    doc_text = build_card(story, title_ar, body_ar, profile["age_band"])
    filename = _safe_filename(title_ar, story["id"])
    caption = title_ar[:1024]
    log.info("card genre=%s source=%s original_title=%s voices=%s",
             tale_type_label(story), story.get("source"), story.get("title_en"), len(oggs))

    zip_bytes: bytes | None = None
    zip_meta: dict = {}
    zip_name = f"{story['id']}_scenes.zip"
    if images_enabled():
        try:
            zip_bytes, _paths, zip_meta = await asyncio.to_thread(
                build_scenes_zip, title_ar, body_ar
            )
            log.info("story images ready meta=%s", zip_meta)
        except Exception:
            log.exception("story images generation failed; sending text+voice only")
            zip_bytes = None

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
                for i, ogg in enumerate(oggs, 1):
                    vcap = caption if len(oggs) == 1 else (
                        f"{caption} — الجزء {str(i).translate(_AR_DIGITS)} من {str(len(oggs)).translate(_AR_DIGITS)}"
                    )
                    await bot.send_voice(
                        chat_id=uid,
                        voice=BytesIO(ogg),
                        caption=vcap[:1024],
                        filename=f"{story['id']}_{i}.ogg",
                    )
                if zip_bytes:
                    zbuf = BytesIO(zip_bytes)
                    zbuf.name = zip_name
                    await bot.send_document(
                        chat_id=uid,
                        document=zbuf,
                        filename=zip_name,
                        caption=(f"مشاهد القصة ({zip_meta.get('scenes', '?')} صورة) — 16:9")[:1024],
                    )
                    log.info("sent scenes zip user=%s scenes=%s kb=%.0f",
                             uid, zip_meta.get("scenes"), len(zip_bytes) / 1024)
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
            for name in (meta.get("names") or [])[:8]:
                store.mark_hero_sent(name, story["id"])
            store.advance_episode_profile()
            return title_ar
        return None
    finally:
        if own:
            await bot.shutdown()
