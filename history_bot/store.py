from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import DB_PATH

GENRES = ["animals", "legend", "human", "history"]

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

@contextmanager
def db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init() -> None:
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS subscribers (
            user_id INTEGER PRIMARY KEY,
            joined_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sent_battles (
            battle_id TEXT PRIMARY KEY,
            sent_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sent_stories (
            story_id TEXT PRIMARY KEY,
            sent_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS send_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            battle_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            ok INTEGER NOT NULL,
            error TEXT,
            sent_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sent_story_titles (
            title_key TEXT PRIMARY KEY,
            title_en TEXT NOT NULL,
            story_id TEXT,
            reserved_at TEXT NOT NULL,
            sent_at TEXT
        );
        """)
        rows = conn.execute("SELECT battle_id, sent_at FROM sent_battles").fetchall()
        for r in rows:
            conn.execute(
                "INSERT OR IGNORE INTO sent_stories(story_id, sent_at) VALUES (?, ?)",
                (r["battle_id"], r["sent_at"]),
            )

def _get_meta(conn, key: str, default: str) -> str:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default

def _set_meta(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )

def subscribe(user_id: int) -> bool:
    now = datetime.now(timezone.utc).isoformat()
    with db() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO subscribers(user_id, joined_at) VALUES (?, ?)",
            (user_id, now),
        )
        return cur.rowcount > 0

def unsubscribe(user_id: int) -> None:
    with db() as conn:
        conn.execute("DELETE FROM subscribers WHERE user_id = ?", (user_id,))

def subscriber_ids() -> list[int]:
    with db() as conn:
        rows = conn.execute("SELECT user_id FROM subscribers ORDER BY user_id").fetchall()
    return [int(r["user_id"]) for r in rows]

def sent_ids() -> set[str]:
    with db() as conn:
        rows = conn.execute("SELECT story_id FROM sent_stories").fetchall()
    return {r["story_id"] for r in rows}

def sent_title_keys() -> set[str]:
    with db() as conn:
        rows = conn.execute("SELECT title_key FROM sent_story_titles").fetchall()
    return {r["title_key"] for r in rows}

def reserve_story(story_id: str, title_en: str) -> None:
    """Lock original English title BEFORE translation so it is never reprocessed."""
    from .stories import normalize_title, title_id
    key = normalize_title(title_en)
    if not key:
        key = story_id
    tid = title_id(title_en)
    now = datetime.now(timezone.utc).isoformat()
    with db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sent_story_titles(title_key, title_en, story_id, reserved_at) VALUES (?, ?, ?, ?)",
            (key, title_en, story_id, now),
        )
        conn.execute(
            "INSERT OR IGNORE INTO sent_stories(story_id, sent_at) VALUES (?, ?)",
            (story_id, now),
        )
        if tid and tid != story_id:
            conn.execute(
                "INSERT OR IGNORE INTO sent_stories(story_id, sent_at) VALUES (?, ?)",
                (tid, now),
            )

def mark_sent(story_id: str, title_en: str | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sent_stories(story_id, sent_at) VALUES (?, ?)",
            (story_id, now),
        )
        conn.execute(
            "INSERT OR IGNORE INTO sent_battles(battle_id, sent_at) VALUES (?, ?)",
            (story_id, now),
        )
        if title_en:
            from .stories import normalize_title, title_id
            key = normalize_title(title_en) or story_id
            tid = title_id(title_en)
            conn.execute(
                """
                INSERT INTO sent_story_titles(title_key, title_en, story_id, reserved_at, sent_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(title_key) DO UPDATE SET
                    story_id=excluded.story_id,
                    sent_at=excluded.sent_at
                """,
                (key, title_en, story_id, now, now),
            )
            if tid and tid != story_id:
                conn.execute(
                    "INSERT OR IGNORE INTO sent_stories(story_id, sent_at) VALUES (?, ?)",
                    (tid, now),
                )

def log_send(story_id: str, user_id: int, ok: bool, error: str | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with db() as conn:
        conn.execute(
            "INSERT INTO send_log(battle_id, user_id, ok, error, sent_at) VALUES (?, ?, ?, ?, ?)",
            (story_id, user_id, 1 if ok else 0, error, now),
        )

def drop_subscriber(user_id: int) -> None:
    unsubscribe(user_id)

def next_voice_gender() -> str:
    with db() as conn:
        gender = _get_meta(conn, "next_voice", "female")
        nxt = "male" if gender == "female" else "female"
        _set_meta(conn, "next_voice", nxt)
    return gender

def peek_episode_profile() -> dict:
    """Age cadence: every 4 stories -> 3 for 8+, 1 for under 7.
    Genre rotates across animals/legend/human/history.
    """
    with db() as conn:
        slot = int(_get_meta(conn, "story_slot", "0"))
        genre_i = int(_get_meta(conn, "genre_index", "0"))
    age_band = "under7" if (slot % 4) == 3 else "over8"
    genre = GENRES[genre_i % len(GENRES)]
    return {"slot": slot, "age_band": age_band, "genre": genre}

def advance_episode_profile() -> None:
    with db() as conn:
        slot = int(_get_meta(conn, "story_slot", "0")) + 1
        genre_i = int(_get_meta(conn, "genre_index", "0")) + 1
        _set_meta(conn, "story_slot", str(slot))
        _set_meta(conn, "genre_index", str(genre_i))
