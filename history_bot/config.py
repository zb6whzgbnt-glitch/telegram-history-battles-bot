from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY", "").strip()
    or os.environ.get("GEMINI_STORY_API_KEY", "").strip()
)
# Optional dedicated keys (fall back to GEMINI_API_KEY).
GEMINI_STORY_API_KEY = os.environ.get("GEMINI_STORY_API_KEY", "").strip() or GEMINI_API_KEY
GEMINI_TASHKEEL_API_KEY = os.environ.get("GEMINI_TASHKEEL_API_KEY", "").strip() or GEMINI_API_KEY

# Pre-Gemini models (extract / rewrite / translate). Soft-fail if missing/unpaid.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip()
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "").strip()
COHERE_MODEL = os.environ.get("COHERE_MODEL", "command-a-03-2025").strip()

# Optional Cloudflare Workers AI (Llama). Requires BOTH token + account id.
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()

TZ_NAME = os.environ.get("TZ", "Asia/Riyadh")
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "bot.sqlite"
TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-3.6-flash")
TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
# Fixed narrator: Gemini TTS Aoede only (female). Calm, stable kids-story narrator;
# clear articulation; moderate fixed pace. Never Charon/Sulafat alternating.
TTS_VOICE = "Aoede"
FEMALE_VOICE = "Aoede"
MALE_VOICE = "Aoede"
SEND_HOURS = [(0, 0), (3, 0), (6, 0), (9, 0), (12, 0), (15, 0), (18, 0), (21, 0)]
USER_AGENT = "KidsStoriesBot/1.0 (+personal telegram bedtime bot)"
