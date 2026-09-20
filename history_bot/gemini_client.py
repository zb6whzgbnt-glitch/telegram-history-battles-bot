from __future__ import annotations
import logging
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


def _arabic_words(text: str) -> list[str]:
    return [w for w in text.split() if any("\u0600" <= ch <= "\u06FF" for ch in w) or w.isalpha()]


def enforce_audio_budget(title: str, body: str, age_band: str) -> tuple[str, str]:
    """Hard-cap Arabic body for ~3–6 minutes audio (~100 wpm expressive)."""
    max_words = 320 if age_band == "under7" else 600
    # Keep source line if present
    source = ""
    main = body
    if "المصدر:" in body:
        main, _, rest = body.rpartition("المصدر:")
        source = "المصدر:" + rest
        main = main.rstrip()
    words = main.split()
    if len(words) <= max_words:
        return title, body
    cut = " ".join(words[:max_words]).rstrip("،,;؛:")
    # Prefer ending on a sentence boundary
    for end in ("۔", ".", "!", "؟", "…"):
        idx = cut.rfind(end)
        if idx >= int(max_words * 0.55):
            cut = cut[: idx + 1]
            break
    out = cut.strip()
    if source:
        out = out + "\n\n" + source.strip()
    return title, out


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
    models: list[str] = []
    for m in (TEXT_MODEL, "gemini-flash-latest", "gemini-3.6-flash"):
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


def _local_arabize(story: dict, age_band: str, genre: str) -> tuple[str, str]:
    """Fallback fully-Arabized short story (with tashkeel) when Gemini text is down."""
    names = {
        "animals": ("سَارَةُ الْقِطَّةُ", "يُوسُفُ الْأَسَدُ", "نُورَةُ الْعُصْفُورَةُ"),
        "legend": ("الْأَمِيرُ خَالِدٌ", "الْجِنِّيَّةُ لَيْلَى", "قَرْيَةُ النَّخِيلِ"),
        "human": ("سَارَةُ", "يُوسُفُ", "أُمُّ نُورَةَ"),
        "history": ("الصَّبِيُّ عُمَرُ", "بَغْدَادُ", "الْعَالِمُ الْحَسَنُ"),
    }
    a, b, c = names.get(genre, names["human"])
    # Title = hero/heroine name only (Arabic)
    title = a
    if age_band == "under7":
        body = (
            f"فِي قَرْيَةٍ عَرَبِيَّةٍ جَمِيلَةٍ، عَاشَتْ {a} مَعَ {b}. "
            f"ذَاتَ يَوْمٍ ضَاعَتْ طَرِيقُهُمَا، فَسَاعَدَتْهُمَا {c} بِقَلْبٍ طَيِّبٍ. "
            f"تَعَلَّمَ الْجَمِيعُ أَنَّ اللُّطْفَ وَالصَّبْرَ يُضِيئَانِ الدَّرْبَ. "
            f"وَعَادُوا إِلَى الْبَيْتِ فَرِحِينَ آمِنِينَ.\n\n"
            f"المصدر: {story.get('source', 'محلي')}"
        )
    else:
        body = (
            f"فِي زَمَانٍ قَدِيمٍ، انْطَلَقَتْ {a} مَعَ {b} فِي رِحْلَةٍ طَوِيلَةٍ عَبْرَ الصَّحْرَاءِ وَالْوَاحَاتِ. "
            f"وَاجَهَا عَطَشاً وَرِيحاً شَدِيدَةً، لَكِنَّهُمَا تَعَاوَنَا وَلَمْ يَسْتَسْلِمَا. "
            f"فِي الطَّرِيقِ الْتَقَيَا أَطْفَالاً ضَائِعِينَ، فَأَرْشَدَاهُمْ بِالْخَرِيطَةِ وَالْمَاءِ. "
            f"وَلَمَّا وَصَلَا إِلَى {c}، احْتَفَلَ النَّاسُ بِشَجَاعَتِهِمَا وَكَرَمِهِمَا. "
            f"هَكَذَا تَعَلَّمَ الْجَمِيعُ أَنَّ الشَّجَاعَةَ الْحَقِيقِيَّةَ تَكُونُ فِي خِدْمَةِ الْآخَرِينَ.\n\n"
            f"المصدر: {story.get('source', 'محلي')}"
        )
    return title, body


def finalize_kids_story(story: dict, age_band: str = "over8", genre: str = "animals", used_heroes: list[str] | None = None, extra_instruction: str = "") -> tuple[str, str]:
    """Gemini final step: dramatic Arabic kids retelling WITH full tashkeel.

    Title MUST be the main hero/heroine Arabic name only.
    used_heroes: blocklist of recently used hero names (do not reuse as titles).
    Prefers an Arabic draft from prep when present; else source text.
    Finalize keys: gemini-story → gemini-main → gemini-talking-mini.
    """
    age_line = (
        "الفئة: دون السابعة — لغة بسيطة، جمل قصيرة، مغامرة خفيفة مع غموض لطيف يحفّز الانتباه، بدون عنف أو رعب مرعب."
        if age_band == "under7"
        else "الفئة: 8 سنوات فأكثر — مغامرة مشوّقة مع تشويق ورعب خفيف مناسب للأطفال (ظلال، أصوات، أسرار، مفاجآت) بدون دماء أو عنف قاسٍ أو محتوى مرعب للكبار."
    )
    genre_map = {
        "animals": "النوع: حكاية مغامرة على ألسنة الحيوانات مع تشويق.",
        "legend": "النوع: أسطورة / حكاية عجيبة فيها سرّ أو لغز.",
        "human": "النوع: قصة بشرية عن أولاد أو بنات في مغامرة.",
        "history": "النوع: قصة تاريخية مبسطة للأطفال بطابع مغامرة.",
    }
    genre_line = genre_map.get(genre, "النوع: قصة أطفال مغامرة متنوعة.")
    length_line = (
        "اجعل النص حوالي 180–320 كلمة عربية بحيث يكون الإلقاء الصوتي بين 3 و5 دقائق تقريباً."
        if age_band == "under7"
        else "اجعل النص حوالي 320–580 كلمة عربية بحيث يكون الإلقاء الصوتي بين 3 و6 دقائق (فضّل 4–6)."
    )
    draft = (story.get("arabic_draft") or "").strip()
    draft_title = (story.get("arabic_draft_title") or "").strip()
    if draft:
        log.info("gemini finalize from arabic_draft (pre models: extract=%s rewrite=%s translate=%s)",
                 story.get("extract_model"), story.get("rewrite_model"), story.get("translate_model"))
        source_block = (
            f"مسودة عربية جاهزة (بدون تشكيل كامل بالضرورة) — أعد صياغتها دراميًا مع تشكيل كامل:\n"
            f"عنوان المسودة: {draft_title or story.get('title_en', '')}\n\n"
            f"{draft}\n"
        )
        transform_line = (
            "أنت كاتب قصص أطفال عربي محترف. حوّل المسودة العربية التالية إلى رواية نهائية "
            "فصحى مشكّلة بالكامل (تَشْكِيل على كل كلمة)، بأسلوب مغامرة وتشويق خفيف.\n"
        )
    else:
        log.info("gemini finalize from source text (no arabic_draft; classic arabize)")
        source_block = (
            f"العنوان الأصلي: {story['title_en']}\n"
            f"المصدر: {story['source']}\n\n"
            f"النص:\n{story['text_en']}\n"
        )
        transform_line = (
            "أنت كاتب قصص أطفال عربي محترف متخصص في المغامرات والتشويق والتفاعل اللطيف.\n"
            "حوّل النص الأصلي التالي (أي لغة) إلى قصة عربية فصحى بسيطة عالية الجودة تناسب الأطفال، بمغامرة حسية وحوار نشط وتفاعل خفيف مع المستمع، مع تعريب كامل:\n"
        )
    prompt = (
        transform_line
        + f"- {age_line}\n"
        + f"- {genre_line}\n"
        + f"- {length_line}\n"
        + "- اكتب بأسلوب مشوّق تفاعلي: بداية جذّابة فوراً، تصاعد مغامرة، محطات تشويق قصيرة (cliffhangers)، "
        "ذروة واضحة، ثم نهاية مطمئنة دافئة.\n"
        + "- زِد المغامرة والتشويق الخفيف المناسب للأطفال: أسرار، طرق ملتوية، أصوات بعيدة، ظلال، أبواب تُطرق، "
        "خرائط ناقصة، مفاجآت لطيفة — بلا دماء ولا عنف قاسٍ ولا رعب مرعب.\n"
        + "- تفاصيل حسية حيّة (ماذا يُرى/يُسمع/يُشم/يُلمس) وحوار نشط قصير بين الشخصيات يحرّك الحدث.\n"
        + "- تفاعلية خفيفة مع الطفل المستمع دون تحويل القصة إلى اختبار أو ألعاب أسئلة: "
        "سؤال بلاغي واحد أو اثنان مثل «ماذا سيحدث؟» أو «هل تتخيّل ما وراء الباب؟»، "
        "ودعوات قصيرة للتخيّل، ثم عُد فوراً إلى السرد.\n"
        + "- لا تكسر الإيقاع بسرد طويل من التعليمات؛ اجعل التفاعل جزءاً من الحكاية نفسها.\n"
        + "- غيّر الأسماء والشخصيات والأماكن والطابع إلى بيئة عربية أصيلة "
        "(مثل: سارة، يوسف، نورة، خالد، بغداد، دمشق، قرية النخيل...) ولا تُبقِ أسماء أجنبية.\n"
        + "- اجعل الأجواء عربية خالصة مع عبرة واضحة في الخاتمة.\n"
        + "- اكتب فصحى بسيطة مضبوطة لغوياً ونحوياً، مع تشكيل كامل (تَشْكِيل) على كل الكلمات "
        "في العنوان ونص القصة (الحركات: فتحة، ضمة، كسرة، سكون، تنوين، شدة).\n"
        + "- لا تستخدم إيموجي ولا عناوين فرعية كثيرة.\n"
        + "- مهم جداً: العنوان = اسم البطل أو البطلة فقط بالعربية (مثل: سارة، يوسف، نورة، خالد) "
        "أو صيغة قصيرة باسم البطل مثل «يوسف والصياد» إذا لزم — فضّل اسماً شخصياً واضحاً معرّباً.\n"
        + "- لا تجعل العنوان وصفاً عاماً (مثل حكاية القرية / رحلة الصحراء) — يجب أن يكون اسم البطل/البطلة.\n"
        + "- في السطر الأول فقط اكتب: العنوان: ثم اسم البطل/البطلة مع التشكيل الكامل.\n"
        + "- ثم سطر فارغ ثم نص القصة مشكّلاً بالكامل.\n"
        + "- في آخر سطر: المصدر: ثم اسم المصدر كما هو.\n"
    )
    heroes = [h.strip() for h in (used_heroes or []) if h and str(h).strip()]
    if heroes:
        # Pass display names; strip tashkeel for compactness in blocklist
        shown = []
        for h in heroes[:400]:
            clean = strip_tashkeel(h).strip()
            if clean and clean not in shown:
                shown.append(clean)
        if shown:
            prompt += (
                "- لا تستخدم هذه الأسماء للعناوين/الأبطال (مستخدمة سابقاً): "
                + "، ".join(shown)
                + ".\n"
                + "- اختر اسماً عربياً جديداً وغير مكرر للبطل/البطلة.\n"
            )
    if extra_instruction:
        prompt += f"- {extra_instruction.strip()}\n"
    prompt += (
        "\n"
        + f"المصدر الأصلي للموقع: {story.get('source', '')}\n\n"
        + source_block
    )
    key_cands = _finalize_key_candidates()
    model_cands = _text_model_candidates()
    if not key_cands:
        log.error("Gemini finalize: no API keys configured; using local arabized fallback")
        t, b = _local_arabize(story, age_band, genre)
        return enforce_audio_budget(t, b, age_band)

    last_err: Exception | None = None
    for key_label, api_key in key_cands:
        # Permanent key/project denial: skip remaining models for this key quickly.
        key_denied = False
        for model_name in model_cands:
            try:
                def _gen(k=api_key, m=model_name):
                    return _client_for_key(k).models.generate_content(model=m, contents=prompt)

                resp = _call_with_retry(_gen, tries=4, base=2.0)
                out = (resp.text or "").strip()
                if not out:
                    raise RuntimeError("empty finalized story")
                title = draft_title or story.get("title_en", "")
                body = out
                if out.startswith("العنوان:") or "العنوان:" in out.split("\n", 1)[0]:
                    first, _, rest = out.partition("\n")
                    title = first.split(":", 1)[-1].strip()
                    body = rest.strip()
                if "المصدر:" not in body:
                    body = body + f"\n\nالمصدر: {story['source']}"
                log.info(
                    "finalize ok key=%s model=%s title_chars=%s body_words~%s",
                    key_label,
                    model_name,
                    len(title),
                    len(body.split()),
                )
                return enforce_audio_budget(title, body, age_band)
            except Exception as exc:
                last_err = exc
                msg = str(exc)
                log.warning(
                    "finalize soft-fail key=%s model=%s: %s",
                    key_label,
                    model_name,
                    msg[:200],
                )
                # 403 / project denied / invalid key → skip other models on this key
                if any(
                    x in msg.upper()
                    for x in ("403", "PERMISSION_DENIED", "API_KEY_INVALID", "CONSUMER_INVALID")
                ):
                    key_denied = True
                    break
                # 404 / NOT_FOUND on model → try next model
                continue
        if key_denied:
            continue

    log.error(
        "Gemini finalize: all keys/models failed (last=%s); using local arabized fallback",
        str(last_err)[:180] if last_err else "none",
    )
    t, b = _local_arabize(story, age_band, genre)
    return enforce_audio_budget(t, b, age_band)


def _disambiguate_hero_title(title: str, body: str, blocked_keys: set[str]) -> str:
    """Append a distinguishing plot/place word so the hero title key is unique."""
    from .store import normalize_hero_name

    base = (title or "").strip() or "بطل"
    # Prefer Arabic tokens that look like places/roles (length >= 3 letters)
    candidates: list[str] = []
    for w in re.findall(r"[\u0600-\u06FF]{3,}", body or ""):
        clean = strip_tashkeel(w)
        stop = {
            "المصدر", "كان", "كانت", "قالت", "قال", "هذه", "هذا", "هؤلاء",
            "التي", "الذي", "الذين", "على", "إلى", "الي", "من", "في", "عن",
            "مع", "ثم", "بعد", "قبل", "كل", "بعض", "غير", "بين", "حتى",
            "لم", "لن", "ما", "لا", "إن", "ان", "أن", "او", "أو", "إذا",
            "ذلك", "تلك", "هناك", "هنا", "جدا", "ايضا", "أيضا", "وقد", "فقد",
            "وهو", "وهي", "فهم", "فهي", "انهم", "انها", "قصة", "الحكاية",
        }
        if clean in stop or len(clean) < 3:
            continue
        if clean and clean not in candidates:
            candidates.append(clean)
        if len(candidates) >= 40:
            break
    # Prefer place-ish tokens later in shortlist; try several suffixes
    for suffix in candidates:
        cand = f"{strip_tashkeel(base).split()[0] if base else 'بطل'} و{suffix}"
        # Keep some tashkeel-free short form; if base had tashkeel, keep base + suffix
        cand_display = f"{base} و{suffix}"
        if normalize_hero_name(cand_display) not in blocked_keys:
            return cand_display
    # Last resort: numeric-ish distinguisher from body length
    n = (len(body or "") % 97) + 2
    return f"{base} والقصة{n}"


def arabize_story(
    story: dict,
    age_band: str = "over8",
    genre: str = "animals",
    used_heroes: list[str] | None = None,
) -> tuple[str, str]:
    """Prep + Gemini finalize; title is hero name; avoid used hero names."""
    from .store import normalize_hero_name

    try:
        from .ai_pipeline import prepare_story_for_gemini

        story = prepare_story_for_gemini(story, age_band=age_band, genre=genre)
    except Exception:
        log.exception("ai_pipeline prepare failed; continuing with raw story to Gemini")

    blocked = {normalize_hero_name(h) for h in (used_heroes or []) if h}
    title, body = finalize_kids_story(
        story, age_band=age_band, genre=genre, used_heroes=used_heroes
    )
    key = normalize_hero_name(title)
    if key and key not in blocked:
        return title, body

    log.warning("hero title collision or empty (%s); retrying finalize once", title[:40])
    title2, body2 = finalize_kids_story(
        story,
        age_band=age_band,
        genre=genre,
        used_heroes=used_heroes,
        extra_instruction=(
            "يجب اختيار اسم بطل/بطلة عربي جديد تماماً وغير موجود في قائمة الأسماء المستخدمة. "
            "لا تكرر الاسم السابق. العنوان = الاسم فقط."
        ),
    )
    key2 = normalize_hero_name(title2)
    if key2 and key2 not in blocked:
        return title2, body2

    # Still colliding: append distinguishing word from plot
    fixed = _disambiguate_hero_title(title2 or title, body2 or body, blocked)
    log.warning("hero title still colliding; disambiguated to %s", fixed[:60])
    return fixed, (body2 or body)


def _spoken_body(narration: str) -> str:
    spoken_body = narration
    if "المصدر:" in spoken_body:
        spoken_body = spoken_body.split("المصدر:")[0].strip()
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


def _split_chunks(body: str, limit: int = 1400) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for para in body.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        cand = (buf + "\n\n" + para).strip() if buf else para
        if len(cand) <= limit:
            buf = cand
        else:
            if buf:
                chunks.append(buf)
            if len(para) <= limit:
                buf = para
            else:
                for i in range(0, len(para), limit):
                    chunks.append(para[i : i + limit])
                buf = ""
    if buf:
        chunks.append(buf)
    if not chunks:
        chunks = [body[:limit] or body]
    return chunks


def synthesize_long(narration: str, voice: str | None = None) -> tuple[bytes, int]:
    """Prefer one TTS call for the whole story; chunk only if needed.

    Always uses Aoede + the same TTS_INSTRUCTION — never switch voice mid-story.
    """
    voice_name = voice or TTS_VOICE
    body = _spoken_body(narration)

    # Prefer a single call for stable pitch across the clip.
    try:
        log.info("tts single-call chars=%s voice=%s", len(body), voice_name)
        return synthesize_pcm(body, voice_name)
    except Exception as exc:
        log.warning("tts single-call failed (%s); falling back to chunks with same voice", str(exc)[:160])

    chunks = _split_chunks(body, limit=1400)
    pcm_all = b""
    rate = 24000
    for i, ch in enumerate(chunks):
        log.info("tts chunk %s/%s chars=%s voice=%s", i + 1, len(chunks), len(ch), voice_name)
        pcm, rate = synthesize_pcm(ch, voice_name)
        pcm_all += pcm
    return pcm_all, rate
