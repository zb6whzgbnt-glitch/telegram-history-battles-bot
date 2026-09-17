from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
TZ_NAME = os.environ.get("TZ", "Asia/Riyadh")
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "bot.sqlite"
TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
FEMALE_VOICE = os.environ.get("GEMINI_FEMALE_VOICE", "Sulafat")
MALE_VOICE = os.environ.get("GEMINI_MALE_VOICE", "Charon")
SEND_HOURS = [(0, 0), (3, 0), (6, 0), (9, 0), (12, 0), (15, 0), (18, 0), (21, 0)]
USER_AGENT = "KidsStoriesBot/1.0 (+personal telegram bedtime bot)"
