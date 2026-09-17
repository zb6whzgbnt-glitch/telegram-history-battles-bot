from __future__ import annotations

import logging
import random
import re
from typing import Any

import requests
import yt_dlp

from .stories import _fits_age, _guess_genre, _id, _word_count, normalize_title

log = logging.getLogger("history_bot.youtube")

# handle, lang, default genre hint
CHANNELS: list[dict[str, str]] = [
    # English
    {"handle": "StorylineOnline", "lang": "en", "genre": "human"},
    {"handle": "CBeebies", "lang": "en", "genre": "human"},
    {"handle": "KidTimeStoryTime", "lang": "en", "genre": "legend"},
    {"handle": "vooks", "lang": "en", "genre": "human"},
    {"handle": "StoryTimefromSpace", "lang": "en", "genre": "human"},
    {"handle": "supersimplesongs", "lang": "en", "genre": "animals"},
    {"handle": "FairyTalesandStoriesforKids", "lang": "en", "genre": "legend"},
    {"handle": "LittleFoxKids", "lang": "en", "genre": "human"},
    {"handle": "peppapig", "lang": "en", "genre": "human"},
    {"handle": "oxbridgebaby", "lang": "en", "genre": "human"},
    # French
    {"handle": "LesPtitszAmis", "lang": "fr", "genre": "human"},
    {"handle": "ContesDeFeesFrancais", "lang": "fr", "genre": "legend"},
    {"handle": "MondedesTitounis", "lang": "fr", "genre": "human"},
    {"handle": "tchoupi", "lang": "fr", "genre": "human"},
    {"handle": "trotro", "lang": "fr", "genre": "animals"},
    {"handle": "comptines", "lang": "fr", "genre": "human"},
    {"handle": "petitoursbrun", "lang": "fr", "genre": "animals"},
    {"handle": "HelloMaestro", "lang": "fr", "genre": "history"},
    {"handle": "LeLoupOfficiel", "lang": "fr", "genre": "animals"},
    {"handle": "SimonSuperLapin", "lang": "fr", "genre": "animals"},
    # Spanish
    {"handle": "CuentosInfantiles", "lang": "es", "genre": "legend"},
    {"handle": "elreinoinfantil", "lang": "es", "genre": "human"},
    {"handle": "CuentosdeHadasEspanoles", "lang": "es", "genre": "legend"},
    {"handle": "toycantando", "lang": "es", "genre": "human"},
    {"handle": "pocoyotv", "lang": "es", "genre": "human"},
    {"handle": "lunacreciente", "lang": "es", "genre": "human"},
    {"handle": "SmileandLearnEspanol", "lang": "es", "genre": "human"},
    {"handle": "PinkfongES", "lang": "es", "genre": "animals"},
    {"handle": "CuentosyCanciones", "lang": "es", "genre": "human"},
    {"handle": "PeppaPigEspanolLatino", "lang": "es", "genre": "human"},
]

_YTDLP_BASE = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "ignoreerrors": True,
    "extractor_args": {"youtube": {"player_client": ["android"]}},
}


def _channel_videos_url(handle: str) -> str:
    h = handle.lstrip("@")
    return f"https://www.youtube.com/@{h}/videos"


def list_channel_entries(handle: str, limit: int = 25) -> list[dict[str, Any]]:
    opts = {
        **_YTDLP_BASE,
        "extract_flat": True,
        "playlistend": limit,
    }
    url = _channel_videos_url(handle)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        log.exception("channel list failed @%s", handle)
        return []
    out = []
    for e in (info or {}).get("entries") or []:
        if not e:
            continue
        vid = e.get("id") or e.get("url")
        title = (e.get("title") or "").strip()
        if not vid or not title:
            continue
        if isinstance(vid, str) and vid.startswith("http"):
            m = re.search(r"[?&]v=([\w-]{6,})", vid) or re.search(r"/shorts/([\w-]{6,})", vid)
            vid = m.group(1) if m else vid
        dur = e.get("duration")
        try:
            dur_i = int(dur) if dur is not None else None
        except Exception:
            dur_i = None
        out.append({"video_id": vid, "title": title, "duration": dur_i})
    return out


def _pick_track(info: dict, prefer_langs: list[str]) -> tuple[str, dict] | None:
    subs = info.get("subtitles") or {}
    autos = info.get("automatic_captions") or {}
    for lang in prefer_langs:
        for bag in (subs, autos):
            tracks = bag.get(lang) or []
            if not tracks:
                # prefix match e.g. en-US
                for k, v in bag.items():
                    if k.lower().startswith(lang.lower()) and v:
                        tracks = v
                        break
            if not tracks:
                continue
            for prefer in ("json3", "srv3", "vtt", "srt", "ttml", "srv1", "srv2"):
                for t in tracks:
                    if t.get("ext") == prefer and t.get("url"):
                        return lang, t
            if tracks[0].get("url"):
                return lang, tracks[0]
    return None


def _parse_caption_payload(ext: str, raw: str) -> str:
    if ext == "json3":
        import json
        data = json.loads(raw)
        parts: list[str] = []
        for ev in data.get("events") or []:
            for seg in ev.get("segs") or []:
                t = seg.get("utf8")
                if t and t.strip() and t != "\n":
                    parts.append(t.replace("\n", " ").strip())
        return re.sub(r"\s+", " ", " ".join(parts)).strip()
    # crude VTT/SRT strip
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        if re.match(r"^\d+$", line):
            continue
        if re.search(r"\d{2}:\d{2}:\d{2}", line) or "-->" in line:
            continue
        line = re.sub(r"<[^>]+>", "", line)
        if line:
            lines.append(line)
    # dedupe consecutive duplicates common in auto captions
    out: list[str] = []
    for ln in lines:
        if not out or out[-1] != ln:
            out.append(ln)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def fetch_caption_text(video_id: str, lang: str) -> tuple[str, str] | None:
    """Return (text, lang_used) or None."""
    prefer = [lang, "en", "fr", "es"]
    # unique preserve order
    seen = set()
    prefer_langs = []
    for x in prefer:
        if x not in seen:
            seen.add(x)
            prefer_langs.append(x)
    opts = dict(_YTDLP_BASE)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    except Exception:
        log.exception("video extract failed %s", video_id)
        return None
    if not info:
        return None
    picked = _pick_track(info, prefer_langs)
    if not picked:
        # fallback: description as weak seed
        desc = (info.get("description") or "").strip()
        if len(desc.split()) >= 40:
            return desc[:4000], lang
        return None
    used_lang, track = picked
    try:
        r = requests.get(track["url"], timeout=45)
        r.raise_for_status()
        raw = r.text
    except Exception:
        log.exception("caption download failed %s", video_id)
        return None
    body = _parse_caption_payload(track.get("ext") or "", raw)
    if not body:
        return None
    return body, used_lang


def build_story_from_entry(ch: dict[str, str], entry: dict[str, Any]) -> dict | None:
    title = entry["title"]
    vid = entry["video_id"]
    dur = entry.get("duration")
    # skip very long videos (hard to keep arabized audio under 7 min)
    if isinstance(dur, int) and dur > 900:
        return None
    if isinstance(dur, int) and dur < 45:
        return None
    cap = fetch_caption_text(vid, ch["lang"])
    if not cap:
        return None
    body, used_lang = cap
    words = _word_count(body)
    if words < 60 or words > 2200:
        return None
    handle = ch["handle"].lstrip("@")
    source = f"YouTube @{handle}"
    url = f"https://www.youtube.com/watch?v={vid}"
    genre = _guess_genre(title, body) if random.random() < 0.5 else ch.get("genre", "human")
    return {
        "id": _id(f"yt-{handle}", title),
        "title_en": title,  # original title (may be FR/ES); used for dedupe
        "source": source,
        "url": url,
        "text_en": body,  # original transcript language
        "words": words,
        "genre": genre,
        "lang": used_lang,
        "channel": handle,
        "video_id": vid,
    }


def collect_youtube_pool(per_channel: int = 12, max_stories: int = 40) -> list[dict]:
    channels = list(CHANNELS)
    random.shuffle(channels)
    pool: list[dict] = []
    seen_titles: set[str] = set()
    for ch in channels:
        if len(pool) >= max_stories:
            break
        entries = list_channel_entries(ch["handle"], limit=per_channel)
        random.shuffle(entries)
        for entry in entries[:8]:
            if len(pool) >= max_stories:
                break
            key = normalize_title(entry.get("title", ""))
            if key and key in seen_titles:
                continue
            try:
                story = build_story_from_entry(ch, entry)
            except Exception:
                log.exception("build story failed")
                story = None
            if not story:
                continue
            key2 = normalize_title(story["title_en"])
            if key2 in seen_titles:
                continue
            seen_titles.add(key2)
            pool.append(story)
            # one successful caption per channel pass keeps runtime sane
            break
    return pool
