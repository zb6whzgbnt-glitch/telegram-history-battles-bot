"""Multi-model story prep: DeepSeek + Cohere extract/clean/translate; Gemini is NOT used here.

Gemini's only role remains final kids retelling with full tashkeel + Aoede TTS
(see gemini_client.finalize_kids_story / synthesize_long).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable

import requests

from .config import (
    COHERE_API_KEY,
    COHERE_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
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


def _clip(text: str, n: int = 12000) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n] + "\n…"


def deepseek_chat(prompt: str, system: str = "", max_tokens: int = 2500) -> str | None:
    if not DEEPSEEK_API_KEY:
        log.info("model=deepseek skip: no DEEPSEEK_API_KEY")
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.4,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            log.info("model=deepseek/%s ok chars=%s", DEEPSEEK_MODEL, len(text))
            return text
        log.warning("model=deepseek empty response")
        return None
    except Exception as exc:
        log.warning("model=deepseek fail: %s", str(exc)[:200])
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


def extract_clean_story(story: dict) -> dict:
    """AI-clean scraped title+body (DeepSeek primary, Cohere fallback)."""
    title = story.get("title_en") or ""
    body = story.get("text_en") or ""
    system = (
        "You clean children's folk/fairy tale text. Remove navigation, ads, UI chrome, "
        "footnotes, 'subscribe', font-size widgets, and OCR junk. Keep the complete narrative. "
        "Output EXACTLY:\nTITLE: ...\nBODY: ...\n"
        "Keep the original language. Do not invent a new plot."
    )
    prompt = (
        f"Source: {story.get('source', '')}\nURL: {story.get('source_url', '')}\n\n"
        f"Raw title:\n{title}\n\nRaw body:\n{_clip(body)}\n"
    )

    def deepseek():
        return deepseek_chat(prompt, system=system)

    def cohere():
        return cohere_chat(prompt, system=system)

    def cloudflare_llama():
        return cloudflare_llama_chat(prompt, system=system)

    text, used = _first_ok([deepseek, cohere, cloudflare_llama])
    out = dict(story)
    if not text:
        log.info("extract: BeautifulSoup fallback (no AI clean)")
        out["extract_model"] = "beautifulsoup"
        return out
    t, b = _parse_title_body(text, title)
    if b and len(b.split()) >= 40:
        out["title_en"] = t or title
        out["text_en"] = b
        out["words"] = len(b.split())
        out["extract_model"] = used
        log.info("extract: cleaned via %s words=%s", used, out["words"])
    else:
        out["extract_model"] = f"{used}+beautifulsoup"
        log.info("extract: AI too short; keeping BS text")
    return out


def rewrite_source_quality(story: dict) -> dict:
    """Optional light rewrite in the source language for clarity (not arabization)."""
    title = story.get("title_en") or ""
    body = story.get("text_en") or ""
    system = (
        "You are an editor of classic children's tales. Lightly normalize spelling and "
        "paragraphs in the SAME language as the input. Do not modernize culture, do not "
        "shorten below ~80% length, do not add morals lectures. "
        "Output EXACTLY:\nTITLE: ...\nBODY: ..."
    )
    prompt = f"TITLE: {title}\n\nBODY:\n{_clip(body)}\n"

    def cohere():
        return cohere_chat(prompt, system=system, max_tokens=3000)

    def deepseek():
        return deepseek_chat(prompt, system=system, max_tokens=3000)

    text, used = _first_ok([cohere, deepseek])
    out = dict(story)
    if not text:
        out["rewrite_model"] = "skip"
        return out
    t, b = _parse_title_body(text, title)
    if b and len(b.split()) >= 60:
        out["title_en"] = t or title
        out["text_en"] = b
        out["words"] = len(b.split())
        out["rewrite_model"] = used
        log.info("rewrite: via %s words=%s", used, out["words"])
    else:
        out["rewrite_model"] = "skip"
    return out


def translate_arabic_draft(story: dict, age_band: str = "over8", genre: str = "legend") -> dict:
    """Translate/adapt to a clear Arabic draft (NO tashkeel) — Gemini will finalize later."""
    title = story.get("title_en") or ""
    body = story.get("text_en") or ""
    age = "under 7" if age_band == "under7" else "8+"
    system = (
        "You translate children's stories into simple Modern Standard Arabic (فصحى بسيطة). "
        "Arabize character and place names to familiar Arabic ones. "
        "Do NOT add full tashkeel/diacritics. Do NOT write English. "
        "Keep a complete short draft suitable for later dramatic retelling. "
        "Output EXACTLY:\nالعنوان: ...\nالنص: ..."
    )
    prompt = (
        f"Age band: {age}. Genre hint: {genre}.\n"
        f"Original title: {title}\nSource: {story.get('source', '')}\n\n"
        f"Story text:\n{_clip(body)}\n"
    )

    def deepseek():
        return deepseek_chat(prompt, system=system, max_tokens=2800)

    def cohere():
        return cohere_chat(prompt, system=system, max_tokens=2800)

    def cloudflare_llama():
        return cloudflare_llama_chat(prompt, system=system)

    text, used = _first_ok([deepseek, cohere, cloudflare_llama])
    out = dict(story)
    if not text:
        out["translate_model"] = "none"
        out["arabic_draft"] = ""
        log.info("translate: no AI draft (Gemini will arabize from source text)")
        return out
    # Parse Arabic draft
    t, b = _parse_title_body(text, title)
    if "العنوان:" in text.split("\n", 1)[0]:
        first, _, rest = text.partition("\n")
        t = first.split(":", 1)[-1].strip()
        rest = rest.strip()
        if rest.startswith("النص:"):
            b = rest.split(":", 1)[-1].strip()
        else:
            b = rest
    # Must contain Arabic letters
    if b and any("\u0600" <= ch <= "\u06FF" for ch in b):
        out["arabic_draft_title"] = t
        out["arabic_draft"] = b
        out["translate_model"] = used
        log.info("translate: draft via %s arabic_words~%s", used, len(b.split()))
    else:
        out["translate_model"] = "none"
        out["arabic_draft"] = ""
        log.info("translate: AI output not Arabic; skipping draft")
    return out


def prepare_story_for_gemini(story: dict, age_band: str = "over8", genre: str = "legend") -> dict:
    """Full pre-Gemini chain with soft fallbacks. Logs which models ran."""
    log.info(
        "ai_pipeline start source=%s title=%s deepseek=%s cohere=%s cf_llama=%s",
        story.get("source"),
        (story.get("title_en") or "")[:60],
        bool(DEEPSEEK_API_KEY),
        bool(COHERE_API_KEY),
        bool(CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID),
    )
    s = extract_clean_story(story)
    s = rewrite_source_quality(s)
    s = translate_arabic_draft(s, age_band=age_band, genre=genre)
    log.info(
        "ai_pipeline done extract=%s rewrite=%s translate=%s draft=%s",
        s.get("extract_model"),
        s.get("rewrite_model"),
        s.get("translate_model"),
        "yes" if s.get("arabic_draft") else "no",
    )
    return s
