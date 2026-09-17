from __future__ import annotations
import hashlib
import logging
import random
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .config import USER_AGENT

log = logging.getLogger("history_bot.stories")
HEADERS = {"User-Agent": USER_AGENT}

GENRE_HINTS = {
    "animals": [
        "lion", "mouse", "fox", "wolf", "crow", "rabbit", "cat", "dog", "bird",
        "animal", "fable", "tortoise", "hare", "monkey", "bear", "sheep", "goat",
        "أسد", "ثعلب", "فأر", "غراب",
    ],
    "legend": [
        "legend", "myth", "fairy", "magic", "dragon", "genie", "ogre", "princess",
        "arabian", "nights", "enchant", "giant", "wizard", "kingdom",
    ],
    "human": [
        "boy", "girl", "child", "children", "friend", "school", "brother", "sister",
        "prince", "princess", "farmer", "merchant", "orphan",
    ],
    "history": [
        "history", "historical", "king", "queen", "caliph", "egypt", "baghdad",
        "war", "hero", "ancient", "pharaoh", "travel", "explorer", "ibn",
    ],
}

def normalize_title(title: str) -> str:
    """Stable key from original English title (cross-source dedupe)."""
    t = (title or "").lower().strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # drop leading articles
    for art in ("the ", "a ", "an "):
        if t.startswith(art):
            t = t[len(art):]
            break
    return t


def _id(source: str, title: str) -> str:
    raw = f"{source}:{title}".strip().lower()
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

def title_id(title: str) -> str:
    key = normalize_title(title)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] if key else ""


def _get(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def _word_count(text: str) -> int:
    return len(text.split())

def _guess_genre(title: str, text: str) -> str:
    blob = f"{title}\n{text}".lower()
    scores = {g: 0 for g in GENRE_HINTS}
    for g, words in GENRE_HINTS.items():
        for w in words:
            if w in blob:
                scores[g] += 1
    # Aesop/Gutenberg animal bias
    if "lion" in blob or "fox" in blob or "wolf" in blob or "mouse" in blob:
        scores["animals"] += 2
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return random.choice(list(GENRE_HINTS))
    return best

def _fits_age(words: int, age_band: str) -> bool:
    if age_band == "under7":
        return 60 <= words <= 450
    # over8: longer
    return 180 <= words <= 900

def fetch_bedtime_candidates(limit: int = 30) -> list[dict]:
    html = _get("https://bedtimeshortstories.com/")
    soup = BeautifulSoup(html, "lxml")
    out, seen = [], set()
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        if href.startswith("/"):
            href = urljoin("https://bedtimeshortstories.com/", href)
        if "bedtimeshortstories.com" not in href:
            continue
        if any(x in href for x in ("/category/", "/tag/", "/author/", "/page/", "hindi.", "/about", "/contact", "/privacy", "easter", "jesus", "st-patrick", "christmas")):
            continue
        title = a.get_text(" ", strip=True)
        if len(title) < 8 or len(title) > 120:
            continue
        if href in seen:
            continue
        seen.add(href)
        out.append({"source": "bedtimeshortstories", "url": href, "title_en": title})
        if len(out) >= limit:
            break
    return out

def fetch_bedtime_story(meta: dict) -> dict | None:
    html = _get(meta["url"])
    soup = BeautifulSoup(html, "lxml")
    article = soup.select_one("article") or soup.select_one("div.entry-content") or soup.select_one("div.td-post-content")
    if not article:
        return None
    for bad in article.select("script, style, nav, aside, form, .sharedaddy, .jp-relatedposts"):
        bad.decompose()
    paras = []
    for p in article.find_all(["p", "h2", "h3"]):
        t = p.get_text(" ", strip=True)
        if not t:
            continue
        low = t.lower()
        if any(x in low for x in ("subscribe", "facebook", "copyright", "leave a reply", "related posts")):
            continue
        paras.append(t)
    body = _clean("\n\n".join(paras))
    words = _word_count(body)
    if words < 60 or words > 1600:
        return None
    title = meta["title_en"]
    h1 = soup.select_one("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    return {
        "id": _id("bedtime", title),
        "title_en": title,
        "source": "Bedtime Short Stories",
        "source_url": meta["url"],
        "text_en": body,
        "genre": _guess_genre(title, body),
        "words": words,
    }

def fetch_gutenberg_aesop(limit: int = 50) -> list[dict]:
    text = _get("https://www.gutenberg.org/files/21/21-0.txt")
    start = text.find("AESOP")
    if start < 0:
        start = 0
    chunk = text[start:]
    parts = re.split(r"\r?\n\r?\n(?=[A-Z][A-Za-z' ,\-]{3,60}\r?\n)", chunk)
    stories = []
    for part in parts:
        lines = [ln.strip() for ln in part.strip().splitlines() if ln.strip()]
        if len(lines) < 3:
            continue
        title = lines[0]
        if title.upper().startswith("AESOP") or title.upper().startswith("LIFE OF"):
            continue
        if title.startswith("***") or "GUTENBERG" in title.upper():
            continue
        body = _clean("\n".join(lines[1:]))
        words = _word_count(body)
        if words < 40 or words > 900:
            continue
        if not re.match(r"^[A-Za-z]", title):
            continue
        stories.append({
            "id": _id("gutenberg-aesop", title),
            "title_en": title,
            "source": "Project Gutenberg (Aesop's Fables)",
            "source_url": "https://www.gutenberg.org/ebooks/21",
            "text_en": body,
            "genre": "animals",
            "words": words,
        })
        if len(stories) >= limit:
            break
    return stories

def fetch_gutenberg_andersen(limit: int = 20) -> list[dict]:
    # Andersen fairy tales — good for legend/human mix
    try:
        text = _get("https://www.gutenberg.org/files/27200/27200-0.txt")
    except Exception:
        return []
    # crude split on numbered/titled stories
    parts = re.split(r"\r?\n\r?\n(?=[A-Z][A-Za-z0-9' ,\-]{4,70}\r?\n\r?\n)", text)
    stories = []
    for part in parts:
        lines = [ln.strip() for ln in part.strip().splitlines() if ln.strip()]
        if len(lines) < 8:
            continue
        title = lines[0]
        if "GUTENBERG" in title.upper() or title.startswith("***"):
            continue
        body = _clean("\n".join(lines[1:]))
        words = _word_count(body)
        if words < 200 or words > 1600:
            continue
        stories.append({
            "id": _id("gutenberg-andersen", title),
            "title_en": title,
            "source": "Project Gutenberg (Andersen)",
            "source_url": "https://www.gutenberg.org/ebooks/27200",
            "text_en": body,
            "genre": _guess_genre(title, body) if _guess_genre(title, body) != "animals" else "legend",
            "words": words,
        })
        if len(stories) >= limit:
            break
    return stories

def fetch_storynory_candidates(limit: int = 15) -> list[dict]:
    for feed in ("https://www.storynory.com/feed/", "https://www.storynory.com/feeds/stories"):
        try:
            xml = _get(feed)
            break
        except Exception:
            xml = ""
    if not xml:
        return []
    soup = BeautifulSoup(xml, "lxml-xml")
    out = []
    for item in soup.select("item")[:limit]:
        title = (item.title.get_text(strip=True) if item.title else "") or ""
        link = (item.link.get_text(strip=True) if item.link else "") or ""
        desc = ""
        if item.find("content:encoded"):
            desc = item.find("content:encoded").get_text()
        elif item.description:
            desc = item.description.get_text()
        body = _clean(BeautifulSoup(desc, "lxml").get_text("\n", strip=True))
        words = _word_count(body)
        if words < 80 or words > 1600 or not link:
            continue
        out.append({
            "id": _id("storynory", title),
            "title_en": title,
            "source": "Storynory",
            "source_url": link,
            "text_en": body,
            "genre": _guess_genre(title, body),
            "words": words,
        })
    return out

def fetch_storyberries_candidates(limit: int = 8) -> list[dict]:
    try:
        html = _get("https://www.storyberries.com/")
    except Exception:
        return []
    if len(html) < 1000:
        return []
    soup = BeautifulSoup(html, "lxml")
    metas = []
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        if href.startswith("/"):
            href = urljoin("https://www.storyberries.com/", href)
        title = a.get_text(" ", strip=True)
        if "storyberries.com" in href and len(title) > 10 and "/category/" not in href:
            metas.append({"url": href, "title_en": title})
        if len(metas) >= limit:
            break
    stories = []
    for meta in metas[:5]:
        try:
            page = _get(meta["url"])
            soup = BeautifulSoup(page, "lxml")
            art = soup.select_one("article") or soup.select_one("div.entry-content")
            if not art:
                continue
            body = _clean(art.get_text("\n", strip=True))
            words = _word_count(body)
            if words < 80 or words > 1600:
                continue
            stories.append({
                "id": _id("storyberries", meta["title_en"]),
                "title_en": meta["title_en"],
                "source": "Storyberries",
                "source_url": meta["url"],
                "text_en": body,
                "genre": _guess_genre(meta["title_en"], body),
                "words": words,
            })
        except Exception:
            log.exception("storyberries story failed")
    return stories


def _collect_pool() -> list[dict]:
    """Best free text-only children's story sources (no YouTube)."""
    pool: list[dict] = []
    fetchers = [
        fetch_storyberries_candidates,
        fetch_storynory_candidates,
        fetch_gutenberg_aesop,
        fetch_gutenberg_andersen,
        lambda: [s for s in (fetch_bedtime_story(m) for m in fetch_bedtime_candidates(30)) if s],
    ]
    random.shuffle(fetchers)
    for fn in fetchers:
        try:
            items = fn()
            pool.extend([x for x in items if x])
        except Exception:
            log.exception("text collector failed: %s", getattr(fn, "__name__", fn))
    seen_ids, seen_titles, uniq = set(), set(), []
    for s in pool:
        key = normalize_title(s.get("title_en", ""))
        if s["id"] in seen_ids:
            continue
        if key and key in seen_titles:
            continue
        seen_ids.add(s["id"])
        if key:
            seen_titles.add(key)
        uniq.append(s)
    return uniq


def pick_unused_story(
    sent: set[str],
    age_band: str = "over8",
    genre: str = "animals",
    sent_titles: set[str] | None = None,
) -> dict | None:
    used_titles = sent_titles or set()
    pool = []
    for s in _collect_pool():
        if s["id"] in sent:
            continue
        key = normalize_title(s.get("title_en", ""))
        if key and key in used_titles:
            continue
        # Prefer mid-length English source texts that arabize well to 3–6 min
        w = s.get("words", 0)
        if w < 80 or w > 1400:
            continue
        pool.append(s)
    if not pool:
        return None

    def score(s: dict) -> int:
        sc = 0
        if s.get("genre") == genre:
            sc += 5
        # Prefer adventure/legend tone for engagement
        if s.get("genre") in ("legend", "history"):
            sc += 2
        if _fits_age(s.get("words", 0), age_band):
            sc += 4
        elif age_band == "over8" and s.get("words", 0) >= 180:
            sc += 2
        elif age_band == "under7" and s.get("words", 0) <= 500:
            sc += 2
        src = (s.get("source") or "").lower()
        if any(x in src for x in ("storyberries", "storynory", "bedtime", "gutenberg")):
            sc += 2
        # Prefer titles hinting adventure/mystery
        title = (s.get("title_en") or "").lower()
        if any(x in title for x in ("adventure", "mystery", "dark", "shadow", "forest", "cave", "ghost", "dragon", "quest", "secret", "night")):
            sc += 3
        return sc

    ranked = sorted(pool, key=score, reverse=True)
    top = [s for s in ranked if score(s) >= 4] or ranked[:12]
    return random.choice(top)
