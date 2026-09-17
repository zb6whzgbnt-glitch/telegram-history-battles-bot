from __future__ import annotations
import json
from pathlib import Path

from .config import CATALOG_PATH

def load_battles() -> list[dict]:
    path = Path(CATALOG_PATH)
    data = json.loads(path.read_text(encoding="utf-8"))
    battles = data["battles"] if isinstance(data, dict) else data
    return list(battles)

def pick_unused(sent: set[str]) -> dict | None:
    for battle in load_battles():
        if battle["id"] not in sent:
            return battle
    return None
