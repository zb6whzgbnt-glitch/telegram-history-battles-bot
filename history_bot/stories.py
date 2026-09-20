from __future__ import annotations

import hashlib
import logging
import random
import re
from collections import defaultdict
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .config import USER_AGENT

log = logging.getLogger("history_bot.stories")
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8,fr;q=0.6,es;q=0.6,ru;q=0.6",
}

# Preferred primary sources (language tags for diversity rotation).
SOURCE_LANG = {
    "SurLaLune Fairy Tales": "en",
    "Baldwin Project / Main Lesson": "en",
    "Project Gutenberg (Aesop's Fables)": "en",
    "Project Gutenberg (Andersen)": "en",
    "Project Gutenberg Australia": "en",
    "Il était une histoire": "fr",
    "Contes.biz": "fr",
    "РуСказки / rustih": "ru",
    "Лукошко сказок": "ru",
    "Mundo Primaria": "es",
    "Cervantes Virtual (LIJ)": "es",
    "Trove (NLA)": "en",
}

GENRE_HINTS = {
    "animals": [
        "lion", "mouse", "fox", "wolf", "crow", "rabbit", "cat", "dog", "bird",
        "animal", "fable", "tortoise", "hare", "monkey", "bear", "sheep", "goat",
        "أسد", "ثعلب", "فأر", "غراب", "лиса", "волк", "медвед", "zorro", "lobo",
        "renard", "loup", "ours",
    ],
    "legend": [
        "legend", "myth", "fairy", "magic", "dragon", "genie", "ogre", "princess",
        "arabian", "nights", "enchant", "giant", "wizard", "kingdom", "сказк",
        "conte", "légende", "leyenda", "encant", "hadа",
    ],
    "human": [
        "boy", "girl", "child", "children", "friend", "school", "brother", "sister",
        "prince", "princess", "farmer", "merchant", "orphan", "niño", "niña",
        "garçon", "fille", "мальчик", "девоч",
    ],
    "history": [
        "history", "historical", "king", "queen", "caliph", "egypt", "baghdad",
        "war", "hero", "ancient", "pharaoh", "travel", "explorer", "ibn",
        "histoire", "historia", "цар",
    ],
}

# Skip non-narrative / low-quality title cues.
_SKIP_TITLE_RE = re.compile(
    r"(?i)\b("
    r"song|songs|poem|poems|poetry|rhyme|rhymes|nursery\s*rhyme|lullaby|"
    r"comptine|chansons?|po[eé]sie|berceuse|"
    r"canci[oó]n|canciones|poema|poes[ií]a|"
    r"песня|песни|стих|стихотворен|потешк|колыбельн|"
    r"crossword|quiz|worksheet|coloring|colouring|recipe|riddle|riddles|"
    r"contents|index|introduction|preface|bibliography|copyright|"
    r"subscribe|newsletter|category|tag\b|author\b|"
    r"видеокниг|словарь|конкурс|загадк|докучн|"
    r"cuentos\s+con\s+moraleja|leer\s+y\s+escuchar|"
    r"mitos\s+y\s+leyendas|recursos\s+educativos"
    r")\b"
)

_SKIP_BODY_RE = re.compile(
    r"(?i)("
    r"cookie\s*policy|enable\s*javascript|just a moment|checking your browser|"
    r"cloudflare|access denied|captcha|anubis|"
    r"togel|slot\s*gacor|jagoledak"
    r")"
)

_NAV_JUNK_RE = re.compile(
    r"(?i)^(home|accueil|inicio|главная|menu|search|buscar|recherche|next|prev|"
    r"suivant|anterior|siguiente|more|lire la suite)$"
)


def normalize_title(title: str) -> str:
    """Stable key from original title (cross-source dedupe, any script)."""
    t = (title or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    # Drop common leading articles across languages.
    for art in (
        "the ", "a ", "an ",
        "le ", "la ", "les ", "l'", "un ", "une ",
        "el ", "los ", "las ", "una ",
        "der ", "die ", "das ",
    ):
        if t.startswith(art):
            t = t[len(art):]
            break
    # Keep letters (any language) and digits; collapse other punctuation.
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _id(source: str, title: str) -> str:
    raw = f"{source}:{title}".strip().lower()
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def title_id(title: str) -> str:
    key = normalize_title(title)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] if key else ""


def _get(url: str, timeout: int = 28) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    # Soft-detect bot walls / parked domains.
    low = r.text[:4000].lower()
    if any(x in low for x in ("just a moment", "making sure you're not a bot", "cf-challenge", "anubis_challenge")):
        raise RuntimeError(f"bot-wall at {url}")
    if "tremainsvillehall" in (r.url or "").lower() or "jagoledak" in low or "slot gacor" in low:
        raise RuntimeError(f"hijacked/parked domain for {url}")
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Drop isolated footnote digits left in mid-sentence: "donkey, 2 which"
    text = re.sub(r"([A-Za-zÀ-ÿ])\s+\d{1,2}\s+(?=[a-zà-ÿ])", r"\1 ", text)
    return text.strip()


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _guess_genre(title: str, text: str) -> str:
    blob = f"{title}\n{text}".lower()
    scores = {g: 0 for g in GENRE_HINTS}
    for g, words in GENRE_HINTS.items():
        for w in words:
            if w in blob:
                scores[g] += 1
    if any(x in blob for x in ("lion", "fox", "wolf", "mouse", "лиса", "волк", "zorro", "renard")):
        scores["animals"] += 2
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return random.choice(list(GENRE_HINTS))
    return best


def _fits_age(words: int, age_band: str) -> bool:
    if age_band == "under7":
        return 60 <= words <= 450
    return 180 <= words <= 900


def _looks_like_ocr_junk(text: str) -> bool:
    if not text or len(text) < 40:
        return True
    letters = sum(1 for c in text if c.isalpha())
    if letters < 40:
        return True
    ratio = letters / max(len(text), 1)
    if ratio < 0.45:
        return True
    if text.count("�") > 5 or text.count("\ufffd") > 5:
        return True
    # Many single-char "words" → broken OCR / nav crumbs
    words = text.split()
    if words and sum(1 for w in words if len(w) == 1) / len(words) > 0.35:
        return True
    return False


def _is_quality_story(title: str, body: str, min_words: int = 80, max_words: int = 1600) -> bool:
    title = (title or "").strip()
    body = (body or "").strip()
    if len(title) < 3 or len(title) > 160:
        return False
    if _SKIP_TITLE_RE.search(title) or _NAV_JUNK_RE.match(title):
        return False
    if _SKIP_BODY_RE.search(body[:800]):
        return False
    words = _word_count(body)
    if words < min_words or words > max_words:
        return False
    if _looks_like_ocr_junk(body):
        return False
    # Prefer narrative: at least a couple of sentence-like breaks or dialogue
    if body.count(".") + body.count("!") + body.count("?") + body.count("。") + body.count("…") < 2:
        if words < 120:
            return False
    return True


def _story(
    source_key: str,
    source_label: str,
    title: str,
    url: str,
    body: str,
    genre: str | None = None,
) -> dict | None:
    body = _clean(body)
    title = _clean(title) or "Untitled"
    if not _is_quality_story(title, body):
        return None
    return {
        "id": _id(source_key, title),
        "title_en": title,
        "source": source_label,
        "source_url": url,
        "text_en": body,
        "genre": genre or _guess_genre(title, body),
        "words": _word_count(body),
    }


def _extract_article_text(soup: BeautifulSoup, selectors: list[str] | None = None) -> str:
    selectors = selectors or [
        "article",
        "div.entry-content",
        "div.post-content",
        "div.td-post-content",
        "div.content",
        "div#content",
        "main",
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if not el:
            continue
        for bad in el.select(
            "script, style, nav, aside, form, footer, header, .sharedaddy, "
            ".jp-relatedposts, .comments, noscript, iframe, .dflip, [class*=flipbook]"
        ):
            bad.decompose()
        paras = []
        for p in el.find_all(["p", "h2", "h3"]):
            t = p.get_text(" ", strip=True)
            if not t or len(t) < 15:
                continue
            low = t.lower()
            if any(
                x in low
                for x in (
                    "subscribe", "facebook", "copyright", "leave a reply",
                    "related posts", "cookie", "compartir", "suscrib",
                )
            ):
                continue
            paras.append(t)
        if paras:
            return _clean("\n\n".join(paras))
        txt = _clean(el.get_text("\n", strip=True))
        if _word_count(txt) >= 80:
            return txt
    return ""


# ---------------------------------------------------------------------------
# Primary scrapers
# ---------------------------------------------------------------------------

def fetch_surlalune(limit: int = 18) -> list[dict]:
    """SurLaLune annotated fairy tales (public-domain tale texts)."""
    out: list[dict] = []
    indexes = [
        "https://www.surlalunefairytales.com/annotated-a-g.html",
        "https://www.surlalunefairytales.com/annotated-h-r.html",
        "https://www.surlalunefairytales.com/annotated-s-z.html",
    ]
    metas: list[tuple[str, str]] = []
    seen = set()
    for idx in indexes:
        try:
            html = _get(idx)
        except Exception:
            log.exception("surlalune index failed: %s", idx)
            continue
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("a[href]"):
            href = a.get("href") or ""
            title = a.get_text(" ", strip=True).rstrip(" >").strip()
            if not title or len(title) < 4:
                continue
            full = urljoin(idx, href)
            if "surlalunefairytales.com" not in full:
                continue
            if not full.endswith("-tale.html") and "/tales/" not in full:
                # Prefer *-tale.html pages
                if "-tale.html" not in full:
                    continue
            if full in seen:
                continue
            seen.add(full)
            metas.append((title, full))
    random.shuffle(metas)
    for title, url in metas[: max(limit * 2, 12)]:
        if len(out) >= limit:
            break
        try:
            html = _get(url)
            soup = BeautifulSoup(html, "lxml")
            h1 = soup.select_one("h1")
            if h1 and h1.get_text(strip=True):
                title = h1.get_text(" ", strip=True)
                title = re.sub(r"\s*[|\u2014\-].*SurLaLune.*$", "", title, flags=re.I).strip()
                title = re.sub(r"\s*Annotated Tale\s*$", "", title, flags=re.I).strip()
            paras = []
            for p in soup.find_all("p"):
                t = p.get_text(" ", strip=True)
                if not t or len(t) < 40:
                    continue
                low = t.lower()
                if any(
                    x in low
                    for x in (
                        "annotated version", "recommend reading", "return to",
                        "surlaune", "surlalune", "bookstore", "click here",
                    )
                ):
                    continue
                paras.append(t)
            body = _clean("\n\n".join(paras))
            # Trim trailing annotation dump if huge
            if _word_count(body) > 1400:
                parts = body.split("\n\n")
                acc = []
                for part in parts:
                    acc.append(part)
                    if _word_count("\n\n".join(acc)) >= 900:
                        break
                body = "\n\n".join(acc)
            s = _story("surlalune", "SurLaLune Fairy Tales", title, url, body, genre="legend")
            if s:
                out.append(s)
        except Exception:
            log.exception("surlalune story failed: %s", url)
    return out


def fetch_mainlesson(limit: int = 16) -> list[dict]:
    """Baldwin Project / Gateway to the Classics via mainlesson gateway."""
    books = [
        ("jacobs", "english", "Baldwin Project / Main Lesson"),
        ("jacobs", "moreenglish", "Baldwin Project / Main Lesson"),
        ("aesop", "fables", "Baldwin Project / Main Lesson"),
    ]
    out: list[dict] = []
    metas: list[tuple[str, str, str]] = []
    for author, book, label in books:
        contents = (
            f"http://www.gatewaytotheclassics.com/browse/display.php"
            f"?author={author}&book={book}&story=_contents"
        )
        try:
            html = _get(contents)
        except Exception:
            log.exception("mainlesson contents failed: %s", contents)
            continue
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("a[href*='story=']"):
            href = a.get("href") or ""
            title = a.get_text(" ", strip=True)
            if not title or len(title) < 3:
                continue
            full = urljoin(contents, href)
            if any(x in full for x in ("_contents", "_alphaindex", "_front", "_cover")):
                continue
            metas.append((title, full, label))
    random.shuffle(metas)
    for title, url, label in metas[: max(limit * 3, 20)]:
        if len(out) >= limit:
            break
        try:
            html = _get(url)
            soup = BeautifulSoup(html, "lxml")
            for bad in soup.select("script, style"):
                bad.decompose()
            body = ""
            # Prefer the largest low-link table cell (classic layout).
            best_td = ""
            for td in soup.find_all("td"):
                if len(td.find_all("a")) >= 12:
                    continue
                txt = td.get_text("\n", strip=True)
                w = _word_count(txt)
                if 80 <= w <= 2000 and w > _word_count(best_td):
                    best_td = txt
            if best_td:
                # Drop leading title line if duplicated
                lines = best_td.splitlines()
                if lines and normalize_title(lines[0]) == normalize_title(title):
                    best_td = "\n".join(lines[1:])
                body = _clean(best_td)
            if not body:
                paras = [
                    p.get_text(" ", strip=True)
                    for p in soup.find_all("p")
                    if len(p.get_text(strip=True)) > 40
                ]
                body = _clean("\n\n".join(paras))
            genre = "animals" if "aesop" in url or "fable" in title.lower() else "legend"
            s = _story("mainlesson", label, title, url, body, genre=genre)
            if s:
                out.append(s)
        except Exception:
            log.exception("mainlesson story failed: %s", url)
    return out


def fetch_iletaitunehistoire(limit: int = 12) -> list[dict]:
    """French children's tales from iletaitunehistoire.com."""
    indexes = [
        "https://www.iletaitunehistoire.com/genres/contes-et-legendes",
        "https://www.iletaitunehistoire.com/genres/albums-et-histoires",
    ]
    out: list[dict] = []
    metas: list[tuple[str, str]] = []
    seen = set()
    for idx in indexes:
        try:
            html = _get(idx)
        except Exception:
            log.exception("iletait index failed: %s", idx)
            continue
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("a[href*='/lire/']"):
            href = (a.get("href") or "").split("#")[0]
            if not href:
                continue
            full = urljoin(idx, href)
            if full in seen:
                continue
            raw = a.get_text(" ", strip=True)
            if not raw or len(raw) < 8:
                continue
            if any(x in raw for x in ("Noter", "Voir l", "Lancer", "Ma note")):
                continue
            # Card text often: "Title First sentence… Genre"
            title = re.split(
                r"\s+(?:Contes et légendes|Albums et histoires|Fables|Documentaires)\b",
                raw,
                maxsplit=1,
            )[0].strip()
            # Heuristic: title is first ~6 words if sentence continues
            if len(title) > 90:
                title = " ".join(title.split()[:8])
            seen.add(full)
            metas.append((title, full))
    random.shuffle(metas)
    for title, url in metas[: max(limit * 2, 10)]:
        if len(out) >= limit:
            break
        try:
            html = _get(url)
            soup = BeautifulSoup(html, "lxml")
            if soup.title and soup.title.get_text(strip=True):
                t = soup.title.get_text(strip=True)
                t = re.sub(r"\s*[-–|].*Il était une histoire.*$", "", t, flags=re.I).strip()
                if t:
                    title = t
            # First reading tab / document body
            body = ""
            doc = soup.select_one("#document")
            if doc:
                panes = doc.select(".tab-pane")
                target = None
                for pane in panes:
                    pid = (pane.get("id") or "") + " " + " ".join(pane.get("class") or [])
                    if "audio" in pid.lower() or "ecouter" in pid.lower() or "jeu" in pid.lower():
                        continue
                    txt = pane.get_text("\n", strip=True)
                    if _word_count(txt) >= 80:
                        target = pane
                        break
                if target is None and panes:
                    target = panes[0]
                if target is not None:
                    for bad in target.select("script, style, .idocXNote, button, .gamecard"):
                        bad.decompose()
                    raw = target.get_text("\n", strip=True)
                    raw = re.sub(r"^L'histoire\s*", "", raw)
                    # Drop leading decorative single capital letter lines (A\nmmamellen)
                    raw = re.sub(r"(?m)^([A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜ])\n(?=[a-zàâäçéèêëîïôöùûü])", r"\1", raw)
                    body = _clean(raw)
            if not body:
                body = _extract_article_text(soup)
            s = _story("iletait", "Il était une histoire", title, url, body, genre="legend")
            if s:
                out.append(s)
        except Exception:
            log.exception("iletait story failed: %s", url)
    return out


def fetch_contes_biz(limit: int = 8) -> list[dict]:
    """Contes.biz — soft-fail if domain hijacked/parked."""
    try:
        html = _get("https://www.contes.biz/")
    except Exception as exc:
        log.warning("contes.biz unavailable (soft-fail): %s", exc)
        return []
    # If we somehow get a real page, try to list contes links.
    soup = BeautifulSoup(html, "lxml")
    out: list[dict] = []
    metas = []
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        title = a.get_text(" ", strip=True)
        full = urljoin("https://www.contes.biz/", href)
        if "contes.biz" not in full or len(title) < 5:
            continue
        if any(x in full for x in ("/conte", "/histoire", "/textes", ".html")):
            metas.append((title, full))
    random.shuffle(metas)
    for title, url in metas[:limit]:
        try:
            page = _get(url)
            soup = BeautifulSoup(page, "lxml")
            body = _extract_article_text(soup) or _clean(
                "\n\n".join(p.get_text(" ", strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40)
            )
            s = _story("contesbiz", "Contes.biz", title, url, body, genre="legend")
            if s:
                out.append(s)
        except Exception:
            log.exception("contes.biz story failed: %s", url)
    return out


def fetch_gutenberg_au(limit: int = 12) -> list[dict]:
    """Project Gutenberg Australia — children's/folklore HTML ebooks."""
    ebooks = [
        ("https://gutenberg.net.au/ebooks21/2100071h.html", "Fairy Tales from Australia"),
        ("https://gutenberg.net.au/ebooks21/2100111h.html", "Fairy Tales, Fables, and Legends"),
    ]
    out: list[dict] = []
    for url, book_title in ebooks:
        try:
            html = _get(url)
        except Exception:
            log.exception("gutenberg.au ebook failed: %s", url)
            continue
        soup = BeautifulSoup(html, "lxml")
        for bad in soup.select("script, style"):
            bad.decompose()
        # Split on h2 chapter headings (skip CONTENTS / THE END)
        headings = soup.find_all("h2")
        if len(headings) < 2:
            # Fallback: whole-page chunks are too big; skip
            continue
        for h2 in headings:
            title = h2.get_text(" ", strip=True)
            if not title or title.upper() in {"CONTENTS", "THE END", "INDEX", "PREFACE"}:
                continue
            if _SKIP_TITLE_RE.search(title):
                continue
            parts = []
            for sib in h2.next_siblings:
                if getattr(sib, "name", None) in ("h1", "h2"):
                    break
                if getattr(sib, "get_text", None):
                    t = sib.get_text(" ", strip=True)
                    if t and len(t) > 20:
                        parts.append(t)
            body = _clean("\n\n".join(parts))
            s = _story(
                "gutenberg-au",
                "Project Gutenberg Australia",
                title,
                url,
                body,
                genre=_guess_genre(title, body),
            )
            if s:
                out.append(s)
            if len(out) >= limit:
                return out
    random.shuffle(out)
    return out[:limit]


def fetch_trove(limit: int = 5) -> list[dict]:
    """Trove NLA — only free public-domain text; soft-fail on bot wall/paywall."""
    try:
        # Lightweight probe; Anubis/Cloudflare often blocks automation.
        _get(
            "https://trove.nla.gov.au/search/category/books"
            "?keyword=fairy%20tales%20children&l-availability=y"
        )
    except Exception as exc:
        log.warning("trove.nla.gov.au unavailable (soft-fail): %s", exc)
        return []
    # If somehow reachable without wall, we still avoid scraping paywalled viewer.
    log.info("trove reachable but no free plain-text extractor configured; skipping")
    return []


def fetch_ruskazki(limit: int = 14) -> list[dict]:
    """Russian fairy tales (skazki.rustih.ru redirects to ruskazki.com)."""
    indexes = [
        "https://ruskazki.com/russkie-narodnye-skazki/",
        "https://skazki.rustih.ru/",
        "https://ruskazki.com/avtorskie-skazki/gans-xristian-andersen-skazki/",
    ]
    out: list[dict] = []
    metas: list[tuple[str, str]] = []
    seen = set()
    for idx in indexes:
        try:
            html = _get(idx)
        except Exception:
            log.exception("ruskazki index failed: %s", idx)
            continue
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("h2 a[href], h3 a[href], .entry-title a[href], article h2 a[href]"):
            href = a.get("href") or ""
            title = a.get_text(" ", strip=True)
            if not title or len(title) < 3:
                continue
            full = urljoin(idx, href)
            host = urlparse(full).netloc
            if "ruskazki.com" not in host and "rustih.ru" not in host:
                continue
            if full.rstrip("/").count("/") < 3:
                continue
            if any(x in full for x in ("/page/", "/avtory", "/tag/", "/category/")):
                continue
            if full in seen:
                continue
            seen.add(full)
            metas.append((title, full))
    random.shuffle(metas)
    for title, url in metas[: max(limit * 2, 12)]:
        if len(out) >= limit:
            break
        if _SKIP_TITLE_RE.search(title) or "былина" in title.lower() or "стихотворен" in title.lower():
            continue
        # Skip bare author names / author index titles.
        if (
            re.match(
                r"^(?:Самуил|Александр|Владимир|Михаил|Андерсен|Гримм|Пушкин|Толстой|Маршак|Пришвин)\b",
                title,
            )
            and "—" not in title
            and "–" not in title
            and "сказк" not in title.lower()
            and len(title.split()) <= 3
        ):
            continue
        try:
            html = _get(url)
            soup = BeautifulSoup(html, "lxml")
            if soup.title and soup.title.get_text(strip=True):
                t = soup.title.get_text(strip=True)
                t = re.sub(r"\s*[:\-–|].*(читать|РуСказ|сказк).*$", "", t, flags=re.I).strip()
                if t and len(t) < 120:
                    title = t
            # Skip author archive / list pages
            path = urlparse(url).path.strip("/")
            if path in {"avtorskie-skazki", "russkie-narodnye-skazki"} or path.endswith("/page"):
                continue
            art = soup.select_one("article") or soup.select_one(".entry-content")
            body = ""
            if art:
                for bad in art.select(
                    "script, style, nav, aside, .sharedaddy, form, .font-size, "
                    ".entry-meta, .post-meta, button"
                ):
                    bad.decompose()
                paras = []
                for p in art.find_all("p"):
                    t = p.get_text(" ", strip=True)
                    if not t or len(t) < 20:
                        continue
                    low = t.lower()
                    if any(x in low for x in ("уменьшить шрифт", "увеличить шрифт", "подпис", "вконтакте", "facebook")):
                        continue
                    paras.append(t)
                body = _clean("\n\n".join(paras)) if paras else _clean(art.get_text("\n", strip=True))
            # Reject author bios / index pages mistaken for stories
            if re.search(r"(?i)^(андерсен|гримм|пушкин|маршак|пришвин|толстой)\b", title.strip()) and "сказк" not in title.lower():
                continue
            if body and body.lower().count("сказк") == 0 and "жил" not in body.lower()[:200] and "once" not in body.lower()[:80]:
                # still allow if long narrative paragraphs exist
                if body.count("\n\n") < 2 and _word_count(body) < 250:
                    continue
            s = _story("ruskazki", "РуСказки / rustih", title, url, body, genre="legend")
            if s:
                out.append(s)
        except Exception:
            log.exception("ruskazki story failed: %s", url)
    return out


def fetch_lukoshko(limit: int = 12) -> list[dict]:
    """Лукошко сказок — Russian folk tales."""
    lists = [
        "https://lukoshko.net/storyList/russkie-narodnye-skazki.htm",
        "https://lukoshko.net/storyList/skazki-narodov-vostoka.htm",
        "https://lukoshko.net/storyList/ukrainskie-narodnye-skazki.htm",
    ]
    out: list[dict] = []
    metas: list[tuple[str, str]] = []
    seen = set()
    for idx in lists:
        try:
            html = _get(idx)
        except Exception:
            log.exception("lukoshko list failed: %s", idx)
            continue
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("a[href*='/story/'], a[href*='/content/']"):
            href = a.get("href") or ""
            title = a.get_text(" ", strip=True)
            full = urljoin(idx, href)
            if full in seen or not title or len(title) < 3:
                continue
            if "storyList" in full or "gallery" in full:
                continue
            if any(x in title for x in ("Видео", "Словарь", "Конкурс", "сайт")):
                continue
            if not (full.endswith(".htm") or full.endswith(".html")):
                continue
            seen.add(full)
            metas.append((title, full))
    random.shuffle(metas)
    for title, url in metas[: max(limit * 2, 10)]:
        if len(out) >= limit:
            break
        try:
            html = _get(url)
            soup = BeautifulSoup(html, "lxml")
            if soup.title and soup.title.get_text(strip=True):
                t = soup.title.get_text(strip=True)
                t = re.sub(r"\s*[.\-–|].*(сказка|Лукошко).*$", "", t, flags=re.I).strip()
                if t:
                    title = t
            content = soup.select_one("div.content") or soup.select_one("div#content")
            body = ""
            if content:
                for bad in content.select("script, style, nav, iframe, form"):
                    bad.decompose()
                # Prefer paragraphs; else full text minus nav-heavy blocks
                paras = [
                    p.get_text(" ", strip=True)
                    for p in content.find_all("p")
                    if len(p.get_text(strip=True)) > 25
                ]
                if paras:
                    body = _clean("\n\n".join(paras))
                else:
                    body = _clean(content.get_text("\n", strip=True))
                    # Strip title line
                    lines = body.splitlines()
                    if lines and normalize_title(lines[0]) == normalize_title(title):
                        body = "\n".join(lines[1:]).strip()
            s = _story("lukoshko", "Лукошко сказок", title, url, body, genre="legend")
            if s:
                out.append(s)
        except Exception:
            log.exception("lukoshko story failed: %s", url)
    return out


def fetch_mundoprimaria(limit: int = 12) -> list[dict]:
    """Spanish short children's tales from Mundo Primaria."""
    indexes = [
        "https://www.mundoprimaria.com/cuentos-infantiles-cortos",
        "https://www.mundoprimaria.com/fabulas-para-ninos",
    ]
    out: list[dict] = []
    metas: list[tuple[str, str]] = []
    seen = set()
    for idx in indexes:
        try:
            html = _get(idx)
        except Exception:
            log.exception("mundoprimaria index failed: %s", idx)
            continue
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("a[href]"):
            href = a.get("href") or ""
            title = a.get_text(" ", strip=True)
            full = urljoin(idx, href)
            if "mundoprimaria.com" not in full or not title or len(title) < 4:
                continue
            path = urlparse(full).path.rstrip("/")
            # Story pages live under the index path as an extra slug
            if path == urlparse(idx).path.rstrip("/"):
                continue
            if not (
                "/cuentos-infantiles-cortos/" in full
                or "/fabulas-para-ninos/" in full
            ):
                continue
            if any(x in full for x in ("/page/", "/author", "/tag/", "/category", "#")):
                continue
            if full in seen:
                continue
            seen.add(full)
            metas.append((title, full))
    random.shuffle(metas)
    for title, url in metas[: max(limit * 2, 10)]:
        if len(out) >= limit:
            break
        try:
            html = _get(url)
            soup = BeautifulSoup(html, "lxml")
            if soup.title and soup.title.get_text(strip=True):
                t = soup.title.get_text(strip=True)
                t = re.sub(r"\s*[|\-–◀▷].*$", "", t).strip()
                t = re.sub(r"^\s*[◀▷]\s*", "", t).strip()
                if t:
                    title = t
            title = re.sub(r"\s*\(Leer y Escuchar\)\s*$", "", title, flags=re.I).strip()
            title = re.sub(r"^\s*Cuento\s+", "", title, flags=re.I).strip()
            # Skip category/index-like pages
            if re.search(r"(?i)^(cuentos|f[aá]bulas|leyendas)\b", title) and len(title.split()) <= 4:
                continue
            for bad in soup.select(
                "script, style, nav, aside, footer, header, iframe, "
                ".dflip, [class*=flip], form, .sharedaddy"
            ):
                bad.decompose()
            paras = []
            for p in soup.find_all("p"):
                t = p.get_text(" ", strip=True)
                if not t or len(t) < 25:
                    continue
                low = t.lower()
                if any(
                    x in low
                    for x in (
                        "cookie", "suscrib", "facebook", "instagram", "whatsapp",
                        "mundoprimaria", "descarga", "flipbook", "compartir",
                        "política de privacidad", "derechos reservados",
                    )
                ):
                    continue
                paras.append(t)
            body = _clean("\n\n".join(paras))
            genre = "animals" if "fabula" in url or "fábula" in title.lower() else "legend"
            s = _story("mundoprimaria", "Mundo Primaria", title, url, body, genre=genre)
            if s:
                out.append(s)
        except Exception:
            log.exception("mundoprimaria story failed: %s", url)
    return out


def fetch_cervantes(limit: int = 8) -> list[dict]:
    """Cervantes Virtual LIJ portal — soft-fail on Cloudflare/403."""
    idx = "https://www.cervantesvirtual.com/portales/biblioteca_literatura_infantil_juvenil/cuentos_infantiles/"
    try:
        html = _get(idx)
    except Exception as exc:
        log.warning("cervantesvirtual unavailable (soft-fail): %s", exc)
        return []
    soup = BeautifulSoup(html, "lxml")
    out: list[dict] = []
    metas: list[tuple[str, str]] = []
    seen = set()
    for a in soup.select("a[href*='obra-visor'], a[href*='/obra/']"):
        href = a.get("href") or ""
        title = a.get_text(" ", strip=True)
        title = re.sub(r"Información detallada.*$", "", title).strip()
        title = re.sub(r"^html", "", title).strip()
        if not title or len(title) < 5:
            continue
        full = urljoin(idx, href)
        if "/obras/autor/" in full:
            continue
        # Prefer HTML visor when available
        if "/obra/" in full and "obra-visor" not in full:
            # try construct visor later
            pass
        if full in seen:
            continue
        seen.add(full)
        metas.append((title, full))
    random.shuffle(metas)
    for title, url in metas[: max(limit * 2, 8)]:
        if len(out) >= limit:
            break
        try:
            visor = url
            if "/obra/" in url and "obra-visor" not in url:
                # Heuristic visor URL
                slug = url.rstrip("/").split("/obra/")[-1]
                visor = (
                    "https://www.cervantesvirtual.com/portales/"
                    f"biblioteca_literatura_infantil_juvenil/obra-visor/{slug}/html/"
                )
            html = _get(visor)
            soup = BeautifulSoup(html, "lxml")
            body = _extract_article_text(soup)
            if not body:
                paras = [
                    p.get_text(" ", strip=True)
                    for p in soup.find_all("p")
                    if len(p.get_text(strip=True)) > 40
                ]
                body = _clean("\n\n".join(paras))
            s = _story("cervantes", "Cervantes Virtual (LIJ)", title, visor, body, genre="legend")
            if s:
                out.append(s)
        except Exception as exc:
            log.warning("cervantes story soft-fail %s: %s", url, exc)
    return out


# ---------------------------------------------------------------------------
# Secondary / fallback Gutenberg.org scrapers (kept, demoted in scoring)
# ---------------------------------------------------------------------------

def fetch_gutenberg_aesop(limit: int = 12) -> list[dict]:
    try:
        text = _get("https://www.gutenberg.org/files/21/21-0.txt")
    except Exception:
        log.exception("gutenberg aesop failed")
        return []
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
        if not re.match(r"^[A-Za-z]", title):
            continue
        body = _clean("\n".join(lines[1:]))
        s = _story(
            "gutenberg-aesop",
            "Project Gutenberg (Aesop's Fables)",
            title,
            "https://www.gutenberg.org/ebooks/21",
            body,
            genre="animals",
            )
        if s:
            stories.append(s)
        if len(stories) >= limit:
            break
    return stories


def fetch_gutenberg_andersen(limit: int = 12) -> list[dict]:
    try:
        text = _get("https://www.gutenberg.org/cache/epub/27200/pg27200.txt")
    except Exception:
        log.exception("gutenberg andersen failed")
        return []
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
        # Andersen often long — allow slightly higher min for quality
        if _word_count(body) < 200:
            continue
        genre = _guess_genre(title, body)
        if genre == "animals":
            genre = "legend"
        s = _story(
            "gutenberg-andersen",
            "Project Gutenberg (Andersen)",
            title,
            "https://www.gutenberg.org/ebooks/27200",
            body,
            genre=genre,
        )
        if s:
            stories.append(s)
        if len(stories) >= limit:
            break
    return stories


# ---------------------------------------------------------------------------
# Pool + pick
# ---------------------------------------------------------------------------

def _collect_pool() -> list[dict]:
    """Fetch from primary multilingual sources; rotate for diversity."""
    # Primary pool first (prefer these), then secondary Gutenberg.org.
    primary = [
        ("en", fetch_surlalune),
        ("en", fetch_mainlesson),
        ("en", fetch_gutenberg_au),
        ("en", fetch_trove),
        ("fr", fetch_iletaitunehistoire),
        ("fr", fetch_contes_biz),
        ("ru", fetch_ruskazki),
        ("ru", fetch_lukoshko),
        ("es", fetch_mundoprimaria),
        ("es", fetch_cervantes),
    ]
    secondary = [
        ("en", fetch_gutenberg_aesop),
        ("en", fetch_gutenberg_andersen),
    ]

    by_lang: dict[str, list] = defaultdict(list)
    for lang, fn in primary:
        by_lang[lang].append(fn)
    for lang, fn in secondary:
        by_lang[lang].append(fn)

    langs = list(by_lang.keys())
    random.shuffle(langs)

    pool: list[dict] = []
    # Round-robin across languages so one language cannot dominate.
    # Cap per fetcher soft-fails individually.
    for lang in langs:
        fns = by_lang[lang][:]
        random.shuffle(fns)
        for fn in fns:
            try:
                items = fn()
                pool.extend([x for x in items if x])
                log.info("collector %s/%s -> %s stories", lang, getattr(fn, "__name__", fn), len(items or []))
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
        w = s.get("words", 0)
        if w < 80 or w > 1400:
            continue
        pool.append(s)
    if not pool:
        return None

    primary_sources = (
        "surlalune", "main lesson", "baldwin", "il était", "iletait",
        "contes.biz", "gutenberg australia", "русказ", "rustih", "лукошко",
        "mundo primaria", "cervantes", "trove",
    )

    def score(s: dict) -> int:
        sc = 0
        if s.get("genre") == genre:
            sc += 5
        if s.get("genre") in ("legend", "history"):
            sc += 2
        if _fits_age(s.get("words", 0), age_band):
            sc += 4
        elif age_band == "over8" and s.get("words", 0) >= 180:
            sc += 2
        elif age_band == "under7" and s.get("words", 0) <= 500:
            sc += 2
        src = (s.get("source") or "").lower()
        if any(x in src for x in primary_sources):
            sc += 6  # strongly prefer new primary pool
        elif "gutenberg" in src:
            sc += 1  # demote classic gutenberg.org vs new sources
        title = (s.get("title_en") or "").lower()
        if any(
            x in title
            for x in (
                "adventure", "mystery", "dark", "shadow", "forest", "cave",
                "ghost", "dragon", "quest", "secret", "night", "волк", "лес",
                "замок", "bosque", "dragón", "secret", "forêt", "dragon",
            )
        ):
            sc += 3
        return sc

    ranked = sorted(pool, key=score, reverse=True)
    top = [s for s in ranked if score(s) >= 4] or ranked[:12]
    # Light language diversity: if many from same source, sample among top.
    return random.choice(top)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pool = _collect_pool()
    by_src: dict[str, list[str]] = defaultdict(list)
    for s in pool:
        by_src[s["source"]].append(s["title_en"])
    print(f"POOL_SIZE={len(pool)}")
    for src, titles in sorted(by_src.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f"\n## {src} ({len(titles)})")
        for t in titles[:5]:
            print(f"  - {t}")
