from __future__ import annotations
import logging
import re
import random
import time
from google import genai
from google.genai import types

from .config import FEMALE_VOICE, GEMINI_API_KEY, MALE_VOICE, TEXT_MODEL, TTS_MODEL

log = logging.getLogger("history_bot.gemini")
_client = None


_TASHKEEL_RE = re.compile(r"[\u064B-\u065F\u0670]")


def strip_tashkeel(text: str) -> str:
    """Remove Arabic diacritics; keep letters and punctuation only."""
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
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY missing")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _call_with_retry(fn, tries: int = 5, base: float = 2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            msg = str(e)
            retryable = any(x in msg for x in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "high demand", "timed out", "Timeout"))
            if not retryable or i == tries - 1:
                raise
            sleep_s = base * (2 ** i) + random.uniform(0, 1.5)
            log.warning("gemini retry %s/%s after %.1fs: %s", i + 1, tries, sleep_s, msg[:160])
            time.sleep(sleep_s)
    raise last  # type: ignore[misc]


def _local_arabize(story: dict, age_band: str, genre: str) -> tuple[str, str]:
    """Fallback fully-Arabized short story when Gemini text is down."""
    names = {
        "animals": ("سَارَةُ الْقِطَّةُ", "يُوسُفُ الْأَسَدُ", "نُورَةُ الْعُصْفُورَةُ"),
        "legend": ("الْأَمِيرُ خَالِدٌ", "الْجِنِّيَّةُ لَيْلَى", "قَرْيَةُ النَّخِيلِ"),
        "human": ("سَارَةُ", "يُوسُفُ", "أُمُّ نُورَةَ"),
        "history": ("الصَّبِيُّ عُمَرُ", "بَغْدَادُ", "الْعَالِمُ الْحَسَنُ"),
    }
    a, b, c = names.get(genre, names["human"])
    if age_band == "under7":
        title = f"حِكَايَةُ {a}"
        body = (
            f"فِي قَرْيَةٍ عَرَبِيَّةٍ جَمِيلَةٍ، عَاشَتْ {a} مَعَ {b}. "
            f"ذَاتَ يَوْمٍ ضَاعَتْ طَرِيقُهُمَا، فَسَاعَدَتْهُمَا {c} بِقَلْبٍ طَيِّبٍ. "
            f"تَعَلَّمَ الْجَمِيعُ أَنَّ اللُّطْفَ وَالصَّبْرَ يُضِيئَانِ الدَّرْبَ. "
            f"وَعَادُوا إِلَى الْبَيْتِ فَرِحِينَ آمِنِينَ.\n\n"
            f"المصدر: {story.get('source', 'محلي')}"
        )
    else:
        title = f"رِحْلَةُ {a} إِلَى {c}"
        body = (
            f"فِي زَمَانٍ قَدِيمٍ، انْطَلَقَتْ {a} مَعَ {b} فِي رِحْلَةٍ طَوِيلَةٍ عَبْرَ الصَّحْرَاءِ وَالْوَاحَاتِ. "
            f"وَاجَهَا عَطَشاً وَرِيحاً شَدِيدَةً، لَكِنَّهُمَا تَعَاوَنَا وَلَمْ يَسْتَسْلِمَا. "
            f"فِي الطَّرِيقِ الْتَقَيَا أَطْفَالاً ضَائِعِينَ، فَأَرْشَدَاهُمْ بِالْخَرِيطَةِ وَالْمَاءِ. "
            f"وَلَمَّا وَصَلَا إِلَى {c}، احْتَفَلَ النَّاسُ بِشَجَاعَتِهِمَا وَكَرَمِهِمَا. "
            f"هَكَذَا تَعَلَّمَ الْجَمِيعُ أَنَّ الشَّجَاعَةَ الْحَقِيقِيَّةَ تَكُونُ فِي خِدْمَةِ الْآخَرِينَ.\n\n"
            f"المصدر: {story.get('source', 'محلي')}"
        )
    return title, body


def arabize_story(story: dict, age_band: str = "over8", genre: str = "animals") -> tuple[str, str]:
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
    prompt = (
        "أنت كاتب قصص أطفال عربي محترف متخصص في المغامرات والتشويق.\n"
        "حوّل النص الإنجليزي التالي إلى قصة عربية فصحى بسيطة عالية الجودة تناسب الأطفال، مع تعريب كامل:\n"
        f"- {age_line}\n"
        f"- {genre_line}\n"
        f"- {length_line}\n"
        "- اكتب بأسلوب مشوّق: بداية جذابة، تصاعد توتر، لحظة ذروة، ثم نهاية مطمئنة.\n"
        "- أدخل عنصر مغامرة ولغز/غموض خفيف يحفّز الانتباه (باب يُطرق ليلاً، ظل في الممر، صوت غريب، خريطة ناقصة...) مع بقاء القصة آمنة للأطفال.\n"
        "- غيّر الأسماء والشخصيات والأماكن والطابع إلى بيئة عربية أصيلة "
        "(مثل: سارة، يوسف، نورة، خالد، بغداد، دمشق، قرية النخيل...) ولا تُبقِ أسماء أجنبية.\n"
        "- اجعل الأجواء عربية خالصة مع عبرة واضحة.\n"
        "- اكتب فصحى بسيطة مضبوطة لغوياً ونحوياً، بدون أي علامات تشكيل.\n"
        "- لا تستخدم إيموجي ولا عناوين فرعية كثيرة.\n"
        "- في السطر الأول فقط اكتب: العنوان: ثم عنوان عربي قصير بدون تشكيل.\n"
        "- ثم سطر فارغ ثم نص القصة.\n"
        "- في آخر سطر: المصدر: ثم اسم المصدر كما هو.\n\n"
        f"العنوان الأصلي: {story['title_en']}\n"
        f"المصدر: {story['source']}\n\n"
        f"النص:\n{story['text_en']}\n"
    )
    try:
        def _gen():
            return client().models.generate_content(model=TEXT_MODEL, contents=prompt)

        resp = _call_with_retry(_gen)
        text = (resp.text or "").strip()
        if not text:
            raise RuntimeError("empty arabized story")
        title = story["title_en"]
        body = text
        if text.startswith("العنوان:") or "العنوان:" in text.split("\n", 1)[0]:
            first, _, rest = text.partition("\n")
            title = first.split(":", 1)[-1].strip()
            body = rest.strip()
        if "المصدر:" not in body:
            body = body + f"\n\nالمصدر: {story['source']}"
        title, body = strip_tashkeel(title), strip_tashkeel(body)
        return enforce_audio_budget(title, body, age_band)
    except Exception:
        log.exception("Gemini arabize failed; using local arabized fallback")
        t, b = _local_arabize(story, age_band, genre)
        t, b = strip_tashkeel(t), strip_tashkeel(b)
        return enforce_audio_budget(t, b, age_band)


def synthesize_pcm(narration: str, gender: str) -> tuple[bytes, int]:
    voice = FEMALE_VOICE if gender == "female" else MALE_VOICE
    spoken_body = narration
    if "المصدر:" in spoken_body:
        spoken_body = spoken_body.split("المصدر:")[0].strip()
    spoken_body = spoken_body.replace("العنوان:", "").strip()
    spoken_body = strip_tashkeel(spoken_body)
    if gender == "female":
        persona = (
            "You are a professional Arabic female audiobook narrator for children. "
            "Warm, clear, polished, and highly engaging — like a premium bedtime/story podcast. "
            "Natural voice only: no music, no sound effects, no stage noise. "
            "Performance craft: build suspense before reveals, soft whisper in tender/mysterious beats, "
            "bright lively energy in adventure, hush and whisper in mild spooky beats, gentle smile in kind moments, "
            "measured pace with cinematic pauses, careful emphasis and slight elongation on emotional words, "
            "never flat, never rushed, never shouty."
        )
    else:
        persona = (
            "You are a professional Arabic male audiobook narrator for children. "
            "Warm, clear, polished, and highly engaging — like a premium bedtime/story podcast. "
            "Natural voice only: no music, no sound effects, no stage noise. "
            "Performance craft: build suspense before reveals, soft lower tone in tender/mysterious beats, "
            "bright lively energy in adventure, hush and whisper in mild spooky beats, gentle smile in kind moments, "
            "measured pace with cinematic pauses, careful emphasis and slight elongation on emotional words, "
            "never flat, never rushed, never shouty."
        )
    spoken = (
        persona
        + "\nNarrate this grammatically correct Arabic children's story (plain text, no diacritics) with professional storytelling presence:\n\n"
        + spoken_body
    )

    def _gen():
        return client().models.generate_content(
            model=TTS_MODEL,
            contents=spoken,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                    )
                ),
            ),
        )

    resp = _call_with_retry(_gen, tries=6, base=3.0)
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


def synthesize_long(narration: str, gender: str) -> tuple[bytes, int]:
    body = narration
    if "المصدر:" in body:
        body = body.split("المصدر:")[0].strip()
    chunks: list[str] = []
    buf = ""
    for para in body.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        cand = (buf + "\n\n" + para).strip() if buf else para
        if len(cand) <= 1400:
            buf = cand
        else:
            if buf:
                chunks.append(buf)
            if len(para) <= 1400:
                buf = para
            else:
                for i in range(0, len(para), 1400):
                    chunks.append(para[i : i + 1400])
                buf = ""
    if buf:
        chunks.append(buf)
    if not chunks:
        chunks = [body[:1400]]
    pcm_all = b""
    rate = 24000
    for i, ch in enumerate(chunks):
        log.info("tts chunk %s/%s chars=%s gender=%s", i + 1, len(chunks), len(ch), gender)
        pcm, rate = synthesize_pcm(ch, gender)
        pcm_all += pcm
    return pcm_all, rate
