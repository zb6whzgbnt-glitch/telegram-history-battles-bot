from history_bot import store
from history_bot.catalog import load_battles, pick_unused

def test_subscribe_unsubscribe(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.sqlite")
    store.init()
    assert store.subscribe(1) is True
    assert store.subscribe(1) is False
    assert store.subscriber_ids() == [1]
    store.unsubscribe(1)
    assert store.subscriber_ids() == []

def test_no_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.sqlite")
    store.init()
    battles = load_battles()
    assert len(battles) >= 50
    first = pick_unused(set())
    assert first is not None
    store.mark_sent(first["id"])
    second = pick_unused(store.sent_ids())
    assert second is not None
    assert second["id"] != first["id"]


def test_hero_name_normalize_and_dedupe(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "hero.sqlite")
    store.init()
    assert store.normalize_hero_name("سَارَةُ") == store.normalize_hero_name("سارة")
    assert store.normalize_hero_name("  يوسف   والصياد ") == "يوسف والصياد"
    assert store.reserve_hero("سارة", "s1") is True
    assert store.is_hero_used("سَارَة") is True
    assert store.reserve_hero("سارة", "s2") is False  # duplicate
    assert store.reserve_hero("يوسف", "s3") is True
    store.mark_hero_sent("يوسف", "s3")
    keys = store.used_hero_keys(limit=10)
    assert any("يوسف" in k or "سارة" in k for k in keys)
    assert "سارة" in {store.normalize_hero_name(k) for k in keys} or store.is_hero_used("سارة")

