from __future__ import annotations
import asyncio
from . import store
from .pipeline import send_one_to

def main() -> None:
    store.init()
    ids = store.subscriber_ids()
    if not ids:
        raise SystemExit("no subscribers; send /start first")
    title = asyncio.run(send_one_to(ids))
    print(title or "nothing sent")

if __name__ == "__main__":
    main()
