"""Fidelity rules: complete source extraction, no truncation, genre from source."""
from pathlib import Path

from bs4 import BeautifulSoup

from history_bot import gemini_client as G
from history_bot import stories as S
from history_bot.pipeline import build_card

FIX = Path(__file__).parent / "fixtures" / "surlalune_gingerbread.html"


def _gingerbread() -> dict:
    soup = BeautifulSoup(FIX.read_text(encoding="utf-8"), "lxml")
    body = S._surlalune_tale_text(soup)
    return {
        "id": "x",
        "title_en": "Gingerbread Man",
        "source": "SurLaLune Fairy Tales",
        "text_en": body,
        "words": S._word_count(body),
    }


def test_surlalune_keeps_all_refrains_and_ending():
    st = _gingerbread()
    body = st["text_en"]
    # Every cumulative refrain (threshers, mowers, cow, pig, fox) is kept
    assert body.count("And I can run away from you, I can!") == 5
    assert "A cow and a pig," in body
    # Original ending: the fox eats him
    assert body.rstrip().endswith("never spoke again.")
    assert "I'm all gone!" in body
    # Footnote digits and site junk removed
    assert "ago:1" not in body and "house3" not in body
    assert "Amazon" not in body and "Note:" not in body
    assert st["words"] > 450


def test_tale_type_is_cumulative_not_historical():
    st = _gingerbread()
    assert S.classify_tale_type(st) == "cumulative"
    assert S.tale_type_label(st) == "حكاية تراكمية"
    assert "تاريخية" not in S.tale_type_label(st)


def test_fable_and_fairy_types():
    fable = {"title_en": "The Fox and the Crow", "source": "Project Gutenberg (Aesop's Fables)", "text_en": "A fox saw a crow."}
    assert S.classify_tale_type(fable) == "fable"
    fairy = {"title_en": "The Princess", "source": "x", "text_en": "A fairy cast a magic spell on the princess. The witch laughed."}
    assert S.classify_tale_type(fairy) == "fairy"


def test_card_shows_source_genre_and_original_title():
    st = _gingerbread()
    card = build_card(st, "فَتَى الْكَعْكِ", "نَصٌّ\n\nالمصدر: x", "under7")
    assert "النوع: حكاية تراكمية" in card
    assert "SurLaLune Fairy Tales" in card and "Gingerbread Man" in card
    assert "تاريخية" not in card


def test_fidelity_ratio_gate():
    src_words = 100
    short = "كلمة " * 40
    full = "كلمة " * 80
    assert any(r.startswith("too_short_vs_source") for r in G.validate_story_quality("عنوان", short, src_words))
    assert G.validate_story_quality("عنوان", full, src_words) == []


def test_no_truncation_helpers_left():
    assert not hasattr(G, "enforce_audio_budget")
    assert not hasattr(G, "polish_kids_story")
    assert not hasattr(G, "_local_arabize")


def test_source_split_is_sequential_and_lossless():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 120 for i in range(10))
    parts = G._split_source_parts(text, 300)
    assert len(parts) > 1
    assert " ".join(" ".join(parts).split()) == " ".join(text.split())


def test_tts_split_lossless_no_midword():
    body = "\n\n".join(("جُمْلَةٌ طَوِيلَةٌ جِدًّا. " * 60).strip() for _ in range(4))
    chunks = G._split_tts_chunks(body, limit=800)
    assert all(len(c) <= 800 for c in chunks)
    assert " ".join(" ".join(chunks).split()) == " ".join(body.split())


def test_parse_header_title_and_names():
    out = "العنوان: فَتَى الْكَعْكِ\nالأسماء: سَلِيمٌ، هِنْدُ\n\nكَانَ يَا مَا كَانَ"
    t, names, body = G._parse_header(out)
    assert t == "فَتَى الْكَعْكِ"
    assert names == ["سَلِيمٌ", "هِنْدُ"]
    assert body == "كَانَ يَا مَا كَانَ"


def test_original_title_prompt_not_hero_name():
    # Title instruction must ask for a translation of the ORIGINAL title
    import inspect
    src = inspect.getsource(G._translate_part)
    assert "ترجمة عربية أمينة للعنوان الأصلي" in src
    assert "اسم البطل" in src  # explicitly told NOT to use the hero name as title
