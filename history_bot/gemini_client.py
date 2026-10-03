from __future__ import annotations
import json
import logging
import os
import re
import random
import time
from google import genai
from google.genai import types

from .config import (
    GEMINI_API_KEY,
    GEMINI_STORY_API_KEY,
    TALKING_MINI_API_KEY,
    TEXT_MODEL,
    TTS_MODEL,
    TTS_VOICE,
)

log = logging.getLogger("history_bot.gemini")
_client = None

# Fixed TTS delivery instruction — same on every call (single or chunked).
TTS_INSTRUCTION = (
    "اقرأ النص بنبرة راوية قصص أطفال، بصوت هادئ ومستقر مع لمسة تشويق خفيفة، "
    "ومخارج حروف واضحة وسرعة معتدلة وثابتة تماماً من البداية للنهاية."
)

_TASHKEEL_RE = re.compile(r"[\u064B-\u065F\u0670]")


def strip_tashkeel(text: str) -> str:
    """Remove Arabic diacritics; keep letters and punctuation only.

    Kept for optional tooling; arabize + TTS now KEEP full tashkeel.
    """
    return _TASHKEEL_RE.sub("", text)


def client() -> genai.Client:
    """Client for Gemini TTS (and general). Prefer main GEMINI_API_KEY."""
    global _client
    if _client is None:
        key = GEMINI_API_KEY or TALKING_MINI_API_KEY or GEMINI_STORY_API_KEY
        if not key:
            raise RuntimeError("GEMINI_API_KEY missing")
        _client = genai.Client(api_key=key)
    return _client


_clients_by_key: dict[str, genai.Client] = {}


def _client_for_key(api_key: str) -> genai.Client:
    c = _clients_by_key.get(api_key)
    if c is None:
        c = genai.Client(api_key=api_key)
        _clients_by_key[api_key] = c
    return c


def _finalize_key_candidates() -> list[tuple[str, str]]:
    """Ordered finalize keys: story (dedicated narration) → main → talking-mini.

    Prefer GEMINI_STORY_API_KEY first for kids finalize text. Dedupe by value so a
    shared verify/talking key is not tried twice.
    """
    ordered = [
        ("gemini-story", GEMINI_STORY_API_KEY),
        ("gemini-main", GEMINI_API_KEY),
        ("gemini-talking-mini", TALKING_MINI_API_KEY),
    ]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, key in ordered:
        key = (key or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((label, key))
    return out


def _text_model_candidates() -> list[str]:
    """Prefer stronger/newer flash models first; keep TEXT_MODEL in the mix."""
    models: list[str] = []
    for m in ("gemini-3.6-flash", "gemini-flash-latest", TEXT_MODEL, "gemini-3.8-flash"):
        m = (m or "").strip()
        if m and m not in models:
            models.append(m)
    return models


def _is_permanent_fail(msg: str) -> bool:
    u = msg.upper()
    permanent = any(
        x in u
        for x in (
            "403",
            "404",
            "PERMISSION_DENIED",
            "NOT_FOUND",
            "API_KEY_INVALID",
            "CONSUMER_INVALID",
            "PROJECT_DENIED",
        )
    )
    if not permanent:
        return False
    # Still retry if the same message also looks transient.
    return not any(x in u for x in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))


def _is_retryable(msg: str) -> bool:
    return any(
        x in msg
        for x in (
            "503",
            "UNAVAILABLE",
            "429",
            "RESOURCE_EXHAUSTED",
            "high demand",
            "timed out",
            "Timeout",
        )
    )


def _call_with_retry(fn, tries: int = 5, base: float = 2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            msg = str(e)
            if _is_permanent_fail(msg) or not _is_retryable(msg) or i == tries - 1:
                raise
            sleep_s = base * (2 ** i) + random.uniform(0, 1.5)
            log.warning("gemini retry %s/%s after %.1fs: %s", i + 1, tries, sleep_s, msg[:160])
            time.sleep(sleep_s)
    raise last  # type: ignore[misc]


def story_client() -> genai.Client:
    """Best-effort single client for callers that still expect story_client().

    Finalize itself iterates keys; this prefers story then main then talking-mini.
    """
    cands = _finalize_key_candidates()
    if not cands:
        raise RuntimeError("GEMINI_API_KEY missing")
    return _client_for_key(cands[0][1])


# ---------------------------------------------------------------------------
# FIDELITY POLICY (hard requirement from the user)
#   Stories are transferred EXACTLY as in the source: complete, same events,
#   same order, same repetitions/refrains, same dialogue, same ending, same
#   authorial tone. The ONLY change allowed is arabization of personal names,
#   places, foods and cultural items. No abridging, no softened endings, no
#   invented dialogue/suspense, no word-budget truncation.
# ---------------------------------------------------------------------------

FIDELITY_MIN_RATIO = float(os.environ.get("FIDELITY_MIN_RATIO", "0.7"))
# English uses many separate articles/prepositions that Arabic attaches to words,
# so a COMPLETE faithful translation lands around 0.70–0.80 Arabic words per
# English word (measured: Gingerbread Man 495 → 355 = 0.72). Use a slightly
# lower floor for English so complete translations are not rejected; the
# Gemini fidelity audit (verify_fidelity) is the semantic completeness check.
FIDELITY_MIN_RATIO_EN = float(os.environ.get("FIDELITY_MIN_RATIO_EN", "0.62"))


def min_ratio_for(story: dict) -> float:
    try:
        from .stories import SOURCE_LANG

        lang = SOURCE_LANG.get(story.get("source", ""), "")
    except Exception:
        lang = ""
    # en/fr/es: separate articles/prepositions → lower natural Arabic ratio (Icare fr: 0.73)
    return FIDELITY_MIN_RATIO_EN if lang in ("en", "fr", "es") else FIDELITY_MIN_RATIO
SINGLE_PASS_MAX_WORDS = 900   # longer sources are translated in sequential parts
CHUNK_WORDS = 450             # target source words per translated part


def _arabic_words(text: str) -> list[str]:
    return [w for w in text.split() if any("\u0600" <= ch <= "\u06FF" for ch in w)]


def arabic_word_count(text: str) -> int:
    return len(_arabic_words(_story_body_main(text)))


def tashkeel_density(text: str) -> float:
    """Diacritics per Arabic letter (full tashkeel is roughly 0.6–1.0)."""
    letters = sum(1 for ch in text if "\u0621" <= ch <= "\u064A")
    marks = len(_TASHKEEL_RE.findall(text))
    return marks / max(letters, 1)


def _story_body_main(body: str) -> str:
    main = body or ""
    if "المصدر:" in main:
        main = main.split("المصدر:")[0]
    return main.strip()


def _generate_text(prompt: str, purpose: str, tries: int = 4) -> tuple[str, str]:
    """Run one text prompt across finalize keys/models. Returns (text, 'key/model')."""
    key_cands = _finalize_key_candidates()
    model_cands = _text_model_candidates()
    if not key_cands:
        raise RuntimeError(f"Gemini {purpose}: no API keys configured")
    last_err: Exception | None = None
    for key_label, api_key in key_cands:
        for model_name in model_cands:
            try:
                def _gen(k=api_key, m=model_name):
                    return _client_for_key(k).models.generate_content(model=m, contents=prompt)

                resp = _call_with_retry(_gen, tries=tries, base=2.0)
                out = (resp.text or "").strip()
                if not out:
                    raise RuntimeError(f"empty {purpose} output")
                return out, f"{key_label}/{model_name}"
            except Exception as exc:
                last_err = exc
                msg = str(exc)
                log.warning("%s soft-fail key=%s model=%s: %s", purpose, key_label, model_name, msg[:200])
                if any(x in msg.upper() for x in ("403", "PERMISSION_DENIED", "API_KEY_INVALID", "CONSUMER_INVALID")):
                    break  # key denied → next key
                continue
    raise RuntimeError(f"Gemini {purpose} failed for all keys/models: {last_err}") from last_err


# Fingerprints of the deprecated local template — reject if they appear in output.
_LOCAL_TEMPLATE_FINGERPRINTS = (
    "فِي زَمَانٍ قَدِيمٍ، انْطَلَقَتْ",
    "فِي قَرْيَةٍ عَرَبِيَّةٍ جَمِيلَةٍ، عَاشَتْ",
    "سَاعَدَتْهُمَا",
    "الشَّجَاعَةَ الْحَقِيقِيَّةَ تَكُونُ فِي خِدْمَةِ الْآخَرِينَ",
    "تَعَلَّمَ الْجَمِيعُ أَنَّ اللُّطْفَ وَالصَّبْرَ يُضِيئَانِ الدَّرْبَ",
    "وَاجَهَا عَطَشاً وَرِيحاً شَدِيدَةً",
)


class StoryQualityError(RuntimeError):
    """Raised when a translated story fails fidelity/quality gates (do not send)."""


def looks_like_local_template(title: str, body: str) -> bool:
    blob = f"{title or ''}\n{body or ''}"
    if any(fp in blob for fp in _LOCAL_TEMPLATE_FINGERPRINTS):
        return True
    if "حِكَايَةُ" in blob and "الصَّحْرَاءِ وَالْوَاحَاتِ" in blob:
        return True
    return False


def fidelity_ratio(body_ar: str, source_words: int) -> float:
    return arabic_word_count(body_ar) / max(int(source_words or 0), 1)


def validate_story_quality(title: str, body: str, source_words: int, min_ratio: float | None = None) -> list[str]:
    """Return failing reason codes (empty = OK). Length is judged vs the SOURCE."""
    reasons: list[str] = []
    floor = FIDELITY_MIN_RATIO if min_ratio is None else min_ratio
    ratio = fidelity_ratio(body, source_words)
    if source_words and ratio < floor:
        reasons.append(f"too_short_vs_source:{ratio:.2f}<{floor}")
    if looks_like_local_template(title, body):
        reasons.append("local_template_fingerprint")
    if not (title or "").strip():
        reasons.append("empty_title")
    if arabic_word_count(body) < 30:
        reasons.append("too_little_arabic")
    return reasons


_FIDELITY_RULES = (
    "قواعد الأمانة (إلزامية):\n"
    "1) ترجم النص كاملاً جملةً جملةً بالترتيب نفسه. لا تختصر، ولا تلخّص، ولا تحذف أي حدث أو مشهد أو وصف أو جملة.\n"
    "2) احتفظ بكل تكرار ولازمة وأنشودة كما هي في الأصل وبالعدد نفسه من المرات (في الحكايات التراكمية تتراكم اللازمة "
    "سطراً بعد سطر — اكتبها كاملة في كل مرة، كل سطر من الأنشودة في سطر مستقل).\n"
    "3) احتفظ بكل حوار كما هو، ولا تخترع حواراً أو أسئلة للمستمع أو تشويقاً أو مشاهد أو عبرة غير موجودة في الأصل.\n"
    "4) احتفظ بالنهاية الأصلية تماماً كما هي حتى لو كانت حزينة أو قاسية (مثلاً إن أُكل البطل يبقى مأكولاً). "
    "ممنوع تلطيف النهاية أو إنقاذ الشخصية أو تغيير المصير.\n"
    "5) حافظ على أسلوب المؤلف ونبرته وبنية فقراته وطول جمله قدر ما تسمح العربية.\n"
    "6) التعريب فقط: استبدل أسماء الأشخاص بأسماء عربية، والأماكن بأماكن عربية، والأطعمة والأدوات والعادات "
    "بما يقابلها في الثقافة العربية — بشرط أن يؤدي البديل الدور نفسه في الحكاية دون تغيير أي حدث. "
    "أسماء الحيوانات العامة (الثعلب، البقرة...) تُترجم كما هي.\n"
    "7) اكتب بالعربية الفصحى السليمة مع تشكيل كامل على كل كلمة (فتحة، ضمة، كسرة، سكون، تنوين، شدة).\n"
    "8) لا إيموجي، ولا عناوين فرعية، ولا شروح أو ملاحظات من عندك، ولا كلمات إنجليزية.\n"
)


def _names_block(used_names: list[str] | None) -> str:
    shown: list[str] = []
    for h in (used_names or [])[:300]:
        clean = strip_tashkeel(str(h)).strip()
        if clean and clean not in shown:
            shown.append(clean)
    if not shown:
        return ""
    return (
        "- عند اختيار أسماء عربية للشخصيات، تجنّب هذه الأسماء المستخدمة في قصص سابقة لتتنوّع الأسماء: "
        + "، ".join(shown) + ".\n"
    )


def _split_source_parts(text: str, target_words: int = CHUNK_WORDS) -> list[str]:
    """Split source into sequential parts on paragraph (then sentence) boundaries."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    units: list[str] = []
    for para in paras:
        if len(para.split()) <= target_words:
            units.append(para)
            continue
        sents = re.split(r"(?<=[.!?…»\"])\s+", para)
        buf = ""
        for sent in sents:
            cand = f"{buf} {sent}".strip()
            if buf and len(cand.split()) > target_words:
                units.append(buf)
                buf = sent
            else:
                buf = cand
        if buf:
            units.append(buf)
    parts: list[str] = []
    buf = ""
    for u in units:
        cand = f"{buf}\n\n{u}".strip() if buf else u
        if buf and len(cand.split()) > target_words:
            parts.append(buf)
            buf = u
        else:
            buf = cand
    if buf:
        parts.append(buf)
    return parts or [text]


def _parse_header(out: str) -> tuple[str, list[str], str]:
    """Parse 'العنوان: …' / 'الأسماء: …' header lines; return (title, names, body)."""
    title = ""
    names: list[str] = []
    lines = out.strip().splitlines()
    body_start = 0
    for i, ln in enumerate(lines[:6]):
        clean = ln.strip().lstrip("#*").strip()
        bare = strip_tashkeel(clean)
        if bare.startswith("العنوان"):
            title = clean.split(":", 1)[-1].strip().strip("*").strip()
            body_start = i + 1
        elif bare.startswith("الأسماء") or bare.startswith("الاسماء"):
            raw = clean.split(":", 1)[-1]
            names = [n.strip(" .،,") for n in re.split(r"[،,;؛]", raw) if n.strip(" .،,")]
            names = [n for n in names if strip_tashkeel(n) not in ("لا يوجد", "لا توجد", "لا شيء", "-")]
            body_start = i + 1
    body = "\n".join(lines[body_start:]).strip()
    if strip_tashkeel(body).startswith("النص:"):
        body = body.split(":", 1)[-1].strip()
    return title, names, body


def _translate_part(
    story: dict,
    part_text: str,
    idx: int,
    total: int,
    used_names: list[str] | None,
    title_ar: str,
    names_ar: list[str],
    prev_tail: str,
) -> tuple[str, list[str], str]:
    first = idx == 1
    header = (
        "أنت مترجم أدبي أمين متخصص في نقل حكايات الأطفال العالمية إلى العربية الفصحى. "
        "مهمتك ترجمة أمينة كاملة مع تعريب الأسماء والأماكن والثقافة فقط — لا إعادة كتابة.\n\n"
        + _FIDELITY_RULES
    )
    if first:
        fmt = (
            "\nصيغة الإخراج:\n"
            "- السطر الأول: العنوان: ثم ترجمة عربية أمينة للعنوان الأصلي مشكّلة (ترجم العنوان كما هو؛ "
            "استبدل فقط أي اسم شخص فيه بالاسم العربي نفسه المستخدم في النص — لا تجعل العنوان اسم البطل إن لم يكن كذلك في الأصل).\n"
            "- السطر الثاني: الأسماء: ثم الأسماء العربية التي اخترتها للشخصيات مفصولة بفاصلة (أو: لا يوجد).\n"
            "- ثم سطر فارغ ثم الترجمة المشكّلة كاملة.\n"
        )
        fmt += _names_block(used_names)
    else:
        fmt = (
            "\nهذا جزء لاحق من الحكاية نفسها. ترجم هذا الجزء فقط بالكامل، متابعاً مباشرة من حيث انتهى الجزء السابق، "
            "ولا تُعد ترجمة ما سبق ولا تضف خاتمة من عندك.\n"
            f"- العنوان العربي المعتمد: {title_ar}\n"
            + (f"- استخدم الأسماء العربية نفسها المعتمدة: {'، '.join(names_ar)}\n" if names_ar else "")
            + (f"- آخر ما تُرجم من الجزء السابق (للسياق فقط، لا تكرره):\n{prev_tail}\n" if prev_tail else "")
            + "- صيغة الإخراج: نص الترجمة المشكّلة فقط بلا عنوان ولا مقدمات.\n"
        )
    part_label = f"الجزء {idx} من {total}" if total > 1 else "النص الكامل"
    prompt = (
        header
        + fmt
        + f"\nالعنوان الأصلي: {story.get('title_en', '')}\n"
        + f"المصدر: {story.get('source', '')}\n\n"
        + f"{part_label} (اللغة الأصلية):\n{part_text}\n"
    )
    out, used = _generate_text(prompt, purpose=f"translate part {idx}/{total}")
    if first:
        t, names, body = _parse_header(out)
    else:
        t, names, body = "", [], _parse_header(out)[2] or out
    log.info(
        "translate part %s/%s via %s src_words=%s ar_words=%s",
        idx, total, used, len(part_text.split()), len(_arabic_words(body)),
    )
    return t, names, body


def translate_faithful(
    story: dict,
    used_names: list[str] | None = None,
    force_chunked: bool = False,
    chunk_words: int = CHUNK_WORDS,
) -> tuple[str, list[str], str]:
    """Faithful complete Arabic translation (full tashkeel). Long sources → sequential parts."""
    src = (story.get("text_en") or "").strip()
    words = len(src.split())
    if force_chunked or words > SINGLE_PASS_MAX_WORDS:
        parts = _split_source_parts(src, chunk_words)
    else:
        parts = [src]
    title_ar, names_ar, bodies = "", [], []
    for i, part in enumerate(parts, start=1):
        prev_tail = " ".join(bodies[-1].split()[-60:]) if bodies else ""
        t, names, body = _translate_part(
            story, part, i, len(parts), used_names, title_ar, names_ar, prev_tail
        )
        if i == 1:
            title_ar, names_ar = t, names
        part_ratio = len(_arabic_words(body)) / max(len(part.split()), 1)
        if part_ratio < min_ratio_for(story):
            log.warning("part %s/%s short (ratio %.2f); retrying this part once", i, len(parts), part_ratio)
            t2, names2, body2 = _translate_part(
                story, part, i, len(parts), used_names, title_ar, names_ar, prev_tail
            )
            if len(_arabic_words(body2)) > len(_arabic_words(body)):
                body = body2
                if i == 1:
                    title_ar, names_ar = t2 or title_ar, names2 or names_ar
        bodies.append(body.strip())
    return title_ar, names_ar, "\n\n".join(bodies)


def verify_fidelity(story: dict, body_ar: str) -> dict | None:
    """Ask Gemini to compare source vs translation. None if the check could not run."""
    prompt = (
        "You are a strict translation auditor. Compare the ORIGINAL story with its ARABIC translation. "
        "Arabization of names, places, foods and cultural items is ALLOWED and is not an error. "
        "Report as errors: any omitted event/scene/sentence, omitted or shortened repetitions/refrains, "
        "added events/dialogue/suspense not in the original, and any changed or softened ending.\n"
        "Reply with ONLY a JSON object: "
        '{"complete": true|false, "ending_preserved": true|false, "refrains_preserved": true|false, '
        '"missing": ["..."], "added": ["..."], "ending_original": "...", "ending_translation": "..."}\n\n'
        f"ORIGINAL ({story.get('title_en', '')}):\n{story.get('text_en', '')}\n\n"
        f"ARABIC TRANSLATION:\n{strip_tashkeel(_story_body_main(body_ar))}\n"
    )
    try:
        out, used = _generate_text(prompt, purpose="fidelity verify", tries=3)
    except Exception as exc:
        log.warning("fidelity verify unavailable: %s", str(exc)[:160])
        return None
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        log.warning("fidelity verify: no JSON in reply")
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        log.warning("fidelity verify: bad JSON")
        return None
    data["_via"] = used
    return data


def _verify_ok(v: dict | None) -> bool:
    if v is None:
        return True  # verifier unavailable → rely on the length-ratio gate
    return bool(v.get("complete", True)) and bool(v.get("ending_preserved", True))


def tashkeel_pass(body: str) -> str:
    """Language-only pass: add missing diacritics. Must not change any word.

    Rejected unless the undiacritized text is unchanged (within 2% words).
    No plot, suspense, dialogue or length changes are allowed here.
    """
    parts = _split_tts_chunks(body, limit=3500)
    out_parts: list[str] = []
    for i, part in enumerate(parts, 1):
        prompt = (
            "أضف التشكيل الكامل إلى النص العربي التالي وصحّح الأخطاء الإملائية والنحوية الطفيفة فقط. "
            "ممنوع منعاً باتاً حذف أي كلمة أو جملة أو إضافة أي محتوى أو حوار أو تشويق أو تغيير أي حدث أو ترتيب. "
            "أعد النص نفسه مشكّلاً فقط، بلا مقدمات.\n\n" + part
        )
        try:
            out, _ = _generate_text(prompt, purpose=f"tashkeel {i}/{len(parts)}", tries=3)
        except Exception as exc:
            log.warning("tashkeel pass skipped part %s: %s", i, str(exc)[:120])
            out_parts.append(part)
            continue
        a = len(strip_tashkeel(part).split())
        b = len(strip_tashkeel(out).split())
        if a and abs(b - a) / a <= 0.02:
            out_parts.append(out.strip())
        else:
            log.warning("tashkeel pass rejected part %s (words %s→%s)", i, a, b)
            out_parts.append(part)
    return "\n\n".join(out_parts)


def arabize_story(
    story: dict,
    age_band: str = "over8",
    genre: str = "legend",
    used_heroes: list[str] | None = None,
) -> tuple[str, str, dict]:
    """Clean source → faithful translation → fidelity gates (+retry chunked) → tashkeel.

    Returns (title_ar, body_ar, meta). title_ar is a faithful translation of the
    ORIGINAL title (personal names arabized). Raises StoryQualityError if the
    translation cannot be made complete — caller picks another source.
    """
    try:
        from .ai_pipeline import prepare_story_for_gemini

        story = prepare_story_for_gemini(story, age_band=age_band, genre=genre)
    except Exception:
        log.exception("ai_pipeline prepare failed; continuing with scraped story text")

    src_words = len((story.get("text_en") or "").split())
    if not src_words:
        raise StoryQualityError("empty source text")
    floor = min_ratio_for(story)
    meta: dict = {"source_words": src_words, "attempts": [], "min_ratio": floor}

    best: tuple[str, list[str], str] | None = None
    verdict: dict | None = None
    plans = [
        ("auto", False, CHUNK_WORDS),
        ("chunked", True, CHUNK_WORDS),
        ("chunked-small", True, max(150, CHUNK_WORDS // 2)),
    ]
    for label, chunked, cw in plans:
        try:
            title, names, body = translate_faithful(
                story, used_names=used_heroes, force_chunked=chunked, chunk_words=cw
            )
        except RuntimeError as exc:
            meta["attempts"].append({"plan": label, "error": str(exc)[:160]})
            log.warning("translate plan=%s failed: %s", label, str(exc)[:160])
            continue
        if not title:
            title = story.get("title_en", "")
        reasons = validate_story_quality(title, body, src_words, min_ratio=floor)
        ratio = fidelity_ratio(body, src_words)
        attempt = {"plan": label, "ratio": round(ratio, 2), "reasons": reasons}
        log.info(
            "fidelity plan=%s src_words=%s ar_words=%s ratio=%.2f reasons=%s",
            label, src_words, arabic_word_count(body), ratio, reasons,
        )
        if reasons:
            meta["attempts"].append(attempt)
            if best is None or arabic_word_count(body) > arabic_word_count(best[2]):
                best = (title, names, body)
            continue
        verdict = verify_fidelity(story, body)
        attempt["verify"] = (
            {k: verdict.get(k) for k in ("complete", "ending_preserved", "refrains_preserved", "missing", "added")}
            if verdict else None
        )
        meta["attempts"].append(attempt)
        log.info("fidelity verify plan=%s verdict=%s", label, attempt["verify"])
        if _verify_ok(verdict):
            best = (title, names, body)
            break
        verdict = None
    else:
        raise StoryQualityError(f"faithful translation failed all plans: {meta['attempts']}")

    title, names, body = best

    dens = tashkeel_density(_story_body_main(body))
    if dens < 0.5:
        log.info("tashkeel density %.2f < 0.5 → language-only tashkeel pass", dens)
        body = tashkeel_pass(body)
    meta.update(
        names=names,
        arabic_words=arabic_word_count(body),
        ratio=round(fidelity_ratio(body, src_words), 2),
        tashkeel_density=round(tashkeel_density(_story_body_main(body)), 2),
        verify=verdict,
    )
    log.info(
        "arabize ok title_ar=%s names=%s src_words=%s ar_words=%s ratio=%.2f ending_preserved=%s",
        title[:80], names[:6], src_words, meta["arabic_words"], meta["ratio"],
        (verdict or {}).get("ending_preserved", "unverified"),
    )
    return title, body, meta


# ---------------------------------------------------------------------------
# TTS (Aoede only). Long text is split into ordered parts — never shortened.
# ---------------------------------------------------------------------------

def _spoken_body(narration: str) -> str:
    spoken_body = _story_body_main(narration)
    spoken_body = spoken_body.replace("العنوان:", "").strip()
    return spoken_body


def _tts_key_candidates() -> list[tuple[str, str]]:
    """TTS prefers main GEMINI_API_KEY (Aoede); talking-mini only as backup."""
    ordered = [
        ("gemini-main", GEMINI_API_KEY),
        ("gemini-talking-mini", TALKING_MINI_API_KEY),
    ]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, key in ordered:
        key = (key or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((label, key))
    return out


def _pcm_from_response(resp) -> tuple[bytes, int]:
    parts = resp.candidates[0].content.parts
    raw = b""
    rate = 24000
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if not inline or not inline.data:
            continue
        data = inline.data
        if isinstance(data, str):
            import base64
            data = base64.b64decode(data)
        raw += data
        mime = getattr(inline, "mime_type", "") or ""
        if "rate=" in mime:
            try:
                rate = int(mime.split("rate=")[-1].split(";")[0])
            except ValueError:
                pass
    if not raw:
        raise RuntimeError("empty tts audio")
    return raw, rate


def synthesize_pcm(narration: str, voice: str | None = None) -> tuple[bytes, int]:
    """Synthesize one PCM clip with Aoede + fixed Arabic kids-narrator instruction.

    Prefer GEMINI_API_KEY (main) for Aoede; fall back to talking-mini only if main fails.
    """
    voice_name = voice or TTS_VOICE
    spoken_body = _spoken_body(narration)
    # Keep tashkeel; fixed instruction every call for stable pitch/pace.
    spoken = f"{TTS_INSTRUCTION}\n\n{spoken_body}"
    cands = _tts_key_candidates()
    if not cands:
        raise RuntimeError("GEMINI_API_KEY missing")

    last_err: Exception | None = None
    for key_label, api_key in cands:
        try:
            def _gen(k=api_key):
                return _client_for_key(k).models.generate_content(
                    model=TTS_MODEL,
                    contents=spoken,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                            )
                        ),
                    ),
                )

            resp = _call_with_retry(_gen, tries=6, base=3.0)
            raw, rate = _pcm_from_response(resp)
            if key_label != "gemini-main":
                log.info("tts ok via fallback key=%s voice=%s", key_label, voice_name)
            return raw, rate
        except Exception as exc:
            last_err = exc
            log.warning("tts soft-fail key=%s: %s", key_label, str(exc)[:180])
            continue
    raise last_err or RuntimeError("tts failed")




def _split_tts_chunks(body: str, limit: int = 3000) -> list[str]:
    """Split on paragraph, then sentence, boundaries (never mid-word)."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", body or ""):
        para = para.strip()
        if not para:
            continue
        if len(para) <= limit:
            units.append(para)
            continue
        sents = re.split(r"(?<=[.!؟?…»:])\s+", para)
        buf = ""
        for sent in sents:
            cand = f"{buf} {sent}".strip()
            if buf and len(cand) > limit:
                units.append(buf)
                buf = sent
            else:
                buf = cand
        while len(buf) > limit:  # pathological: no punctuation at all
            cut = buf.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            units.append(buf[:cut].strip())
            buf = buf[cut:].strip()
        if buf:
            units.append(buf)
    chunks: list[str] = []
    buf = ""
    for u in units:
        cand = f"{buf}\n\n{u}" if buf else u
        if buf and len(cand) > limit:
            chunks.append(buf)
            buf = u
        else:
            buf = cand
    if buf:
        chunks.append(buf)
    return chunks or [body]


TTS_CHUNK_CHARS = 3000          # per Gemini TTS call (keeps calls well under limits)
VOICE_MAX_SECONDS = 12 * 60     # group consecutive chunks into voice messages up to this


def synthesize_parts(narration: str, voice: str | None = None) -> list[tuple[bytes, int]]:
    """Synthesize the FULL story with Aoede in ordered chunks.

    Returns a list of voice messages (pcm, rate) to send in order. Each voice
    groups consecutive chunks up to VOICE_MAX_SECONDS. Text is never shortened.
    """
    voice_name = voice or TTS_VOICE
    body = _spoken_body(narration)
    chunks = _split_tts_chunks(body, limit=TTS_CHUNK_CHARS)
    voices: list[tuple[bytes, int]] = []
    cur = b""
    rate = 24000
    for i, ch in enumerate(chunks, 1):
        words = len(ch.split())
        log.info("tts chunk %s/%s chars=%s words=%s voice=%s", i, len(chunks), len(ch), words, voice_name)
        pcm, rate = synthesize_pcm(ch, voice_name)
        secs = len(pcm) / (2 * rate)
        # Truncated audio guard: Arabic narration is ~1.5–2.5 words/s.
        if words >= 20 and secs < words / 4.0:
            log.warning("tts chunk %s looks truncated (%.0fs for %s words); retrying once", i, secs, words)
            pcm2, rate2 = synthesize_pcm(ch, voice_name)
            if len(pcm2) > len(pcm):
                pcm, rate = pcm2, rate2
                secs = len(pcm) / (2 * rate)
        cur_secs = len(cur) / (2 * rate)
        if cur and cur_secs + secs > VOICE_MAX_SECONDS:
            voices.append((cur, rate))
            cur = b""
        cur += pcm
    if cur:
        voices.append((cur, rate))
    log.info("tts done chunks=%s voices=%s total_sec=%.0f", len(chunks), len(voices),
             sum(len(p) / (2 * r) for p, r in voices))
    return voices


def synthesize_long(narration: str, voice: str | None = None) -> tuple[bytes, int]:
    """Back-compat: whole story as one PCM clip (all parts concatenated in order)."""
    voices = synthesize_parts(narration, voice)
    return b"".join(p for p, _ in voices), voices[0][1]
