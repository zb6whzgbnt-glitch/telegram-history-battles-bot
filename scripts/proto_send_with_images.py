"""Local/manual prototype: one short send with STORY_IMAGES=1.

Usage:
  STORY_IMAGES=1 STORY_IMAGE_MAX=12 \\
  PROTO_CHAT_ID=743051047 PYTHONPATH=. .venv/bin/python scripts/proto_send_with_images.py

Does not change the scheduled pipeline defaults. Caps image count for cost/time.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("proto_images")


def main() -> None:
    os.environ.setdefault("STORY_IMAGES", "1")
    os.environ.setdefault("STORY_IMAGE_MAX", "12")
    os.environ.setdefault("STORY_IMAGE_WORDS_PER_SCENE", "20")
    # Prefer under-7 / shorter episodes for the prototype when store profile allows.
    os.environ.setdefault("STORY_IMAGE_OUT", "/workspace/story-images-proto/run")

    from history_bot import store
    from history_bot.pipeline import send_one_to

    store.init()
    chat = (os.environ.get("PROTO_CHAT_ID") or "").strip()
    if chat.isdigit():
        ids = [int(chat)]
    else:
        ids = store.subscriber_ids()[:1]
    if not ids:
        log.error("no chat id; set PROTO_CHAT_ID")
        sys.exit(2)
    log.info("proto send with_images chat=%s STORY_IMAGE_MAX=%s", ids, os.environ.get("STORY_IMAGE_MAX"))
    t0 = time.time()
    title = asyncio.run(send_one_to(ids))
    log.info("done title=%s duration_s=%.1f", title, time.time() - t0)
    print(title or "nothing sent")


if __name__ == "__main__":
    main()
