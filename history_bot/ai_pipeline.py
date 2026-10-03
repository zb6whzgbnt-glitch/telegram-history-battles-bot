"""Source prep: clean scraped text WITHOUT changing the story.

FIDELITY POLICY: stories are transferred exactly as in the source. Prep only
removes scraping junk (nav, ads, footnote numbers). It never rewrites,
summarizes, or drafts a loose Arabic retelling. The previous "rewrite" and
"Arabic draft" steps were removed because they abridged stories.

Clean chain (soft-fail): Talking Mini Gemini → main Gemini → Cohere →
Cloudflare Llama → keep scraped (BeautifulSoup) text.
A cleaned text is accepted only if it keeps >= CLEAN_MIN_RATIO of the words.

Faithful Arabic translation with full tashkeel happens in
gemini_client.translate_faithful. Aoede TTS prefers GEMINI_API_KEY.
"""
from __future__ import annotations

import logging
import re
from typing import Callable

import requests

from .config import (
    COHERE_API_KEY,
    COHERE_MODEL,
    GEMINI_API_KEY,
    GEMINI_STORY_API_KEY,
    TALKING_MINI_API_KEY,
    TEXT_MODEL,
)

log = logging.getLogger("history_bot.ai_pipeline")

# Cloudflare Workers AI (Llama) — needs CLOUDFLARE_ACCOUNT_ID + token.
# Soft-skipped when account id is missing (documented in prepare logs).
CLOUDFLARE_API_TOKEN = ""
CLOUDFLARE_ACCOUNT_ID = ""
try:
    from .config import CLOUDFLARE_ACCOUNT_ID as _CF_ACCT  # type: ignore
    from .config import CLOUDFLARE_API_TOKEN as _CF_TOK  # type: ignore

    CLOUDFLARE_ACCOUNT_ID = _CF_ACCT
    CLOUDFLARE_API_TOKEN = _CF_TOK
except Exception:
    pass

# Cache genai clients per API key so we do not rebuild every call.
_gemini_clients: dict[str, object] = {}


def _clip(text: str, n: int = 40000) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n] + "\n…"


def _gemini_key_for_label(key_label: str) -> tuple[str | None, str]:
    """Return (api_key, log_label) for prep. Labels: talking-mini | main."""
    if key_label == "talking-mini":
        key = TALKING_MINI_API_KEY
        return (key or None), "gemini-talking-mini"
    # main / story: prefer GEMINI_API_KEY, else story key
    key = GEMINI_API_KEY or GEMINI_STORY_API_KEY
    return (key or None), "gemini-main"


def gemini_prep_chat(
    prompt: str,
    system: str = "",
    key_label: str = "talking-mini",
    max_tokens: int = 2500,
) -> str | None:
    """Generate prep text via google.genai. Soft-fail (incl. 403 PERMISSION_DENIED).

    key_label: 'talking-mini' or 'main' — logged as model=gemini-talking-mini / model=gemini-main.
    """
    api_key, model_log = _gemini_key_for_label(key_label)
    if not api_key:
        log.info("model=%s skip: no API key", model_log)
        return None
    try:
        from google import genai
        from google.genai import types

        client = _gemini_clients.get(api_key)
        if client is None:
            client = genai.Client(api_key=api_key)
            _gemini_clients[api_key] = client

        contents = prompt
        config = None
        if system:
            config = types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            )
        else:
            config = types.GenerateContentConfig(max_output_tokens=max_tokens)

        resp = client.models.generate_content(
            model=TEXT_MODEL,
            contents=contents,
            config=config,
        )
        text = (getattr(resp, "text", None) or "").strip()
        if text:
            log.info("model=%s/%s ok chars=%s", model_log, TEXT_MODEL, len(text))
            return text
        log.warning("model=%s empty response", model_log)
        return None
    except Exception as exc:
        msg = str(exc)[:220]
        # Soft-fail: 403 / PERMISSION_DENIED / denied access / quota / etc.
        log.warning("model=%s fail (soft): %s", model_log, msg)
        return None


def cohere_chat(prompt: str, system: str = "", max_tokens: int = 2500) -> str | None:
    if not COHERE_API_KEY:
        log.info("model=cohere skip: no COHERE_API_KEY")
        return None
    try:
        import cohere

        client = cohere.ClientV2(api_key=COHERE_API_KEY)
        messages = []
        if system:
            # Cohere v2: system as first user preamble is fine; also supports system role on some models
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {"model": COHERE_MODEL, "messages": messages}
        try:
            resp = client.chat(**kwargs)
        except Exception:
            # Some accounts reject system role — fold into user message.
            messages = [{"role": "user", "content": (system + "\n\n" + prompt).strip()}]
            resp = client.chat(model=COHERE_MODEL, messages=messages)
        parts = getattr(getattr(resp, "message", None), "content", None) or []
        text = "".join(getattr(p, "text", "") or "" for p in parts).strip()
        if text:
            log.info("model=cohere/%s ok chars=%s", COHERE_MODEL, len(text))
            return text
        log.warning("model=cohere empty response")
        return None
    except Exception as exc:
        log.warning("model=cohere fail: %s", str(exc)[:200])
        return None


def cloudflare_llama_chat(prompt: str, system: str = "", max_tokens: int = 2000) -> str | None:
    """Optional Llama via Cloudflare Workers AI — soft-skip without account id."""
    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        log.info(
            "model=cloudflare-llama skip: need CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID "
            "(token alone is not enough)"
        )
        return None
    try:
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}"
            "/ai/run/@cf/meta/llama-3.1-8b-instruct"
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
            json={"messages": messages, "max_tokens": max_tokens},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        text = (
            (data.get("result") or {}).get("response")
            or (data.get("result") or {}).get("text")
            or ""
        )
        text = str(text).strip()
        if text:
            log.info("model=cloudflare-llama ok chars=%s", len(text))
            return text
        return None
    except Exception as exc:
        log.warning("model=cloudflare-llama fail: %s", str(exc)[:200])
        return None


def _first_ok(fns: list[Callable[[], str | None]]) -> tuple[str | None, str]:
    for fn in fns:
        name = getattr(fn, "__name__", "model")
        try:
            out = fn()
        except Exception as exc:
            log.warning("%s raised: %s", name, str(exc)[:160])
            continue
        if out and out.strip():
            return out.strip(), name
    return None, "none"


def _prep_chain(
    prompt: str,
    system: str,
    *,
    prefer: str = "talking-mini",
    max_tokens: int = 2500,
) -> tuple[str | None, str]:
    """Ordered soft-fail chain for one prep step.

    prefer='talking-mini' → mini then main; prefer='main' → main then mini.
    Then Cohere → Cloudflare Llama.
    """
    if prefer == "main":
        first, second = "main", "talking-mini"
        n1, n2 = "gemini_main", "gemini_talking_mini"
    else:
        first, second = "talking-mini", "main"
        n1, n2 = "gemini_talking_mini", "gemini_main"

    def gemini_first():
        return gemini_prep_chat(prompt, system=system, key_label=first, max_tokens=max_tokens)

    gemini_first.__name__ = n1

    def gemini_second():
        return gemini_prep_chat(prompt, system=system, key_label=second, max_tokens=max_tokens)

    gemini_second.__name__ = n2

    def cohere():
        return cohere_chat(prompt, system=system, max_tokens=max_tokens)

    def cloudflare_llama():
        return cloudflare_llama_chat(prompt, system=system, max_tokens=min(max_tokens, 3000))

    return _first_ok([gemini_first, gemini_second, cohere, cloudflare_llama])


def _parse_title_body(text: str, fallback_title: str) -> tuple[str, str]:
    text = (text or "").strip()
    if not text:
        return fallback_title, ""
    # Prefer TITLE:/BODY: markers
    m = re.search(
        r"(?is)\bTITLE\s*:\s*(.+?)\s*\n\s*BODY\s*:\s*(.+)$",
        text,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Arabic markers
    m = re.search(r"(?is)^\s*العنوان\s*:\s*(.+?)\s*\n\s*(?:النص|المسودة)\s*:\s*(.+)$", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2 and len(lines[0]) < 120:
        return lines[0], "\n".join(lines[1:]).strip()
    return fallback_title, text


CLEAN_MIN_RATIO = 0.9


def extract_clean_story(story: dict) -> dict:
    """AI-clean scraped title+body without touching the narrative.

    Accepted only if the cleaned body keeps >= CLEAN_MIN_RATIO of the words;
    otherwise the scraped text is kept verbatim.
    """
    title = story.get("title_en") or ""
    body = story.get("text_en") or ""
    raw_words = len(body.split())
    system = (
        "You clean scraped text of a children's folk/fairy tale. Remove ONLY website junk: "
        "navigation, ads, UI chrome, share buttons, copyright footers, editor notes, and "
        "footnote/annotation reference numbers glued to words. "
        "Do NOT remove, shorten, merge, reorder, summarize, or reword ANY sentence of the tale. "
        "Keep every repetition, refrain, rhyme/verse line (one per line), dialogue line and the "
        "exact original ending. Keep the original language and spelling. "
        "Output EXACTLY:\nTITLE: ...\nBODY: ...\n"
    )
    prompt = (
        f"Source: {story.get('source', '')}\nURL: {story.get('source_url', '')}\n\n"
        f"Raw title:\n{title}\n\nRaw body:\n{_clip(body)}\n"
    )
    max_tokens = max(4000, int(raw_words * 3))
    text, used = _prep_chain(prompt, system, prefer="talking-mini", max_tokens=max_tokens)
    out = dict(story)
    out["source_words_raw"] = raw_words
    if not text:
        log.info("extract: keep scraped text verbatim (no AI clean) words=%s", raw_words)
        out["extract_model"] = "beautifulsoup"
        return out
    t, b = _parse_title_body(text, title)
    new_words = len((b or "").split())
    if b and raw_words and new_words >= CLEAN_MIN_RATIO * raw_words and new_words <= raw_words * 1.05:
        out["title_en"] = t or title
        out["text_en"] = b
        out["words"] = new_words
        out["extract_model"] = used
        log.info("extract: cleaned via %s words=%s (raw=%s)", used, new_words, raw_words)
    else:
        out["extract_model"] = f"{used}-rejected+beautifulsoup"
        log.info(
            "extract: AI clean rejected (words %s vs raw %s; must keep >=%.0f%%); keeping scraped text",
            new_words, raw_words, CLEAN_MIN_RATIO * 100,
        )
    return out


def prepare_story_for_gemini(story: dict, age_band: str = "over8", genre: str = "legend") -> dict:
    """Clean-only prep (no rewrite / no Arabic draft) so the source stays complete."""
    log.info(
        "ai_pipeline start source=%s title=%s words=%s talking_mini=%s gemini_main=%s cohere=%s cf_llama=%s",
        story.get("source"),
        (story.get("title_en") or "")[:60],
        story.get("words"),
        bool(TALKING_MINI_API_KEY),
        bool(GEMINI_API_KEY or GEMINI_STORY_API_KEY),
        bool(COHERE_API_KEY),
        bool(CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID),
    )
    s = extract_clean_story(story)
    # Fidelity: never hand finalize a loose Arabic draft; translate from the source.
    s.pop("arabic_draft", None)
    s.pop("arabic_draft_title", None)
    log.info("ai_pipeline done extract=%s words=%s (rewrite/draft disabled for fidelity)",
             s.get("extract_model"), s.get("words"))
    return s
