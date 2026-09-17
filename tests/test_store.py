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
