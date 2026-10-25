"""Prototype: Gemini image scenes (~every 10s of narration) → numbered ZIP.

Gated by STORY_IMAGES=1 / --with-images. Uses Gemini image models only
(generate_content + IMAGE modality). Never overlays text on images.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import zipfile
from io import BytesIO
from pathlib import Path

from google.genai import types

from .config import (
    GEMINI_API_KEY,
    GEMINI_STORY_API_KEY,
    TALKING_MINI_API_KEY,
)
from .gemini_client import (
    _call_with_retry,
    _client_for_key,
    _spoken_body,
    strip_tashkeel,
)

log = logging.getLogger("history_bot.story_images")

# ~2 Arabic words/sec for Aoede-style kids narration → ~20 words ≈ 10s
WORDS_PER_SCENE = int(os.environ.get("STORY_IMAGE_WORDS_PER_SCENE", "20"))
MAX_SCENES = int(os.environ.get("STORY_IMAGE_MAX", "15"))
IMAGE_MODEL = os.environ.get("STORY_IMAGE_MODEL", "gemini-3.1-flash-lite-image").strip()
ASPECT = os.environ.get("STORY_IMAGE_ASPECT", "16:9").strip() or "16:9"
STYLE_LOCK = (
    "Children's storybook illustration, soft watercolor and gentle gouache, "
    "warm pastel palette, friendly rounded shapes, cozy lighting, "
    "consistent character designs across scenes, widescreen cinematic 16:9 framing. "
    "Absolutely no text, no letters, no numbers, no Arabic calligraphy, "
    "no watermarks, no logos, no captions, no speech bubbles."
)


def images_enabled() -> bool:
    v = (os.environ.get("STORY_IMAGES") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _image_key_candidates() -> list[tuple[str, str]]:
    ordered = [
        ("gemini-main", GEMINI_API_KEY),
        ("gemini-story", GEMINI_STORY_API_KEY),
        ("gemini-talking-mini", TALKING_MINI_API_KEY),
    ]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, key in ordered:
        key = (key or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((label, key))
    return out


def chunk_narration_for_scenes(
    narration: str,
    *,
    words_per_scene: int = WORDS_PER_SCENE,
    max_scenes: int = MAX_SCENES,
) -> list[str]:
    """Split spoken Arabic body into ~10s chunks by word count (sentence-aware)."""
    body = strip_tashkeel(_spoken_body(narration)).strip()
    if not body:
        return []
    # Prefer sentence boundaries
    sents = [s.strip() for s in re.split(r"(?<=[.!؟?…»])\s+", body) if s.strip()]
    if not sents:
        sents = [body]
    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for sent in sents:
        w = len(sent.split())
        if buf and buf_words + w > words_per_scene and len(chunks) + 1 < max_scenes:
            chunks.append(" ".join(buf))
            buf = [sent]
            buf_words = w
        else:
            buf.append(sent)
            buf_words += w
    if buf:
        if len(chunks) >= max_scenes:
            # fold remainder into last scene
            if chunks:
                chunks[-1] = (chunks[-1] + " " + " ".join(buf)).strip()
            else:
                chunks.append(" ".join(buf))
        else:
            chunks.append(" ".join(buf))
    # Hard cap
    if len(chunks) > max_scenes:
        head, tail = chunks[: max_scenes - 1], chunks[max_scenes - 1 :]
        chunks = head + [" ".join(tail)]
    return chunks


def _plan_scene_prompts(title_ar: str, chunks: list[str], style_lock: str = STYLE_LOCK) -> list[str]:
    """Ask Gemini text to turn each chunk into an English visual prompt (no text in image)."""
    cands = _image_key_candidates()
    if not cands:
        raise RuntimeError("GEMINI_API_KEY missing for scene planning")

    numbered = "\n".join(f"{i+1}. {c[:400]}" for i, c in enumerate(chunks))
    ask = (
        "You plan storyboard frames for a kids picture-book YouTube video.\n"
        f"Story title (Arabic, for context only — do NOT put any text in prompts): {title_ar[:120]}\n"
        f"There are {len(chunks)} ordered narration chunks (~10 seconds each).\n"
        "For EACH chunk write ONE English image prompt (1-2 sentences) describing the visual scene "
        "that matches the narration. Keep the same main characters' appearance across all prompts "
        "(age, clothes, colors, species). Kids-safe, gentle, no violence, no scary faces.\n"
        "CRITICAL: every prompt must forbid text/letters/calligraphy/watermarks on the image.\n"
        f"Shared style to include in every prompt: {style_lock}\n\n"
        f"Chunks:\n{numbered}\n\n"
        'Return ONLY JSON: {"prompts":["...","..."]} with exactly '
        f"{len(chunks)} strings, same order."
    )

    last_err: Exception | None = None
    for label, key in cands:
        for model in ("gemini-3.6-flash", "gemini-flash-latest", "gemini-2.5-flash"):
            try:
                def _gen(k=key, m=model):
                    return _client_for_key(k).models.generate_content(
                        model=m,
                        contents=ask,
                        config=types.GenerateContentConfig(
                            temperature=0.4,
                            response_mime_type="application/json",
                        ),
                    )

                resp = _call_with_retry(_gen, tries=4, base=2.0)
                text = (getattr(resp, "text", None) or "").strip()
                if not text and resp.candidates:
                    parts = resp.candidates[0].content.parts or []
                    text = "".join(getattr(p, "text", "") or "" for p in parts).strip()
                data = json.loads(text)
                prompts = data.get("prompts") or data.get("scenes") or []
                if isinstance(prompts, list) and len(prompts) == len(chunks):
                    out = []
                    for p in prompts:
                        s = str(p).strip()
                        if "no text" not in s.lower():
                            s = s.rstrip(".") + ". No text, letters, or calligraphy."
                        out.append(f"{style_lock} Scene: {s}")
                    log.info("scene prompts planned via %s/%s n=%s", label, model, len(out))
                    return out
            except Exception as exc:
                last_err = exc
                log.warning("scene plan soft-fail %s/%s: %s", label, model, str(exc)[:160])
                continue

    # Fallback: literal chunk paraphrase without LLM
    log.warning("scene plan LLM failed (%s); using heuristic prompts", str(last_err)[:120] if last_err else "none")
    return [
        f"{style_lock} Illustrate this kids-story moment (no text on image): {c[:280]}"
        for c in chunks
    ]


def _extract_image_bytes(resp) -> bytes | None:
    import base64

    for cand in resp.candidates or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in content.parts or []:
            inline = getattr(part, "inline_data", None)
            if not inline or not inline.data:
                continue
            data = inline.data
            if isinstance(data, str):
                data = base64.b64decode(data)
            return data
    return None


def _generate_one_image(prompt: str, *, model: str = IMAGE_MODEL) -> bytes:
    cands = _image_key_candidates()
    if not cands:
        raise RuntimeError("GEMINI_API_KEY missing for image gen")
    last_err: Exception | None = None
    models = [model]
    for alt in (
        "gemini-3.1-flash-lite-image",
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-image-preview",
        "gemini-2.5-flash-image",
    ):
        if alt not in models:
            models.append(alt)

    for label, key in cands:
        for m in models:
            try:
                def _gen(k=key, mdl=m, p=prompt):
                    cfg = types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                        image_config=types.ImageConfig(aspect_ratio=ASPECT),
                    )
                    return _client_for_key(k).models.generate_content(
                        model=mdl,
                        contents=p,
                        config=cfg,
                    )

                resp = _call_with_retry(_gen, tries=5, base=4.0)
                data = _extract_image_bytes(resp)
                if data:
                    log.info("image ok key=%s model=%s bytes=%s", label, m, len(data))
                    return data
                raise RuntimeError("empty image response")
            except Exception as exc:
                last_err = exc
                msg = str(exc)
                log.warning("image soft-fail key=%s model=%s: %s", label, m, msg[:180])
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep(8)
                continue
    raise last_err or RuntimeError("image generation failed")


def build_scenes_zip(
    title_ar: str,
    narration: str,
    *,
    out_dir: Path | None = None,
    words_per_scene: int | None = None,
    max_scenes: int | None = None,
) -> tuple[bytes, list[Path], dict]:
    """Generate scene PNGs + return (zip_bytes, sample_paths, meta)."""
    t0 = time.time()
    wps = words_per_scene if words_per_scene is not None else WORDS_PER_SCENE
    mx = max_scenes if max_scenes is not None else MAX_SCENES
    chunks = chunk_narration_for_scenes(narration, words_per_scene=wps, max_scenes=mx)
    if not chunks:
        raise RuntimeError("no narration chunks for images")

    prompts = _plan_scene_prompts(title_ar, chunks)
    out_dir = Path(out_dir or os.environ.get("STORY_IMAGE_OUT", "/tmp/story-images"))
    out_dir.mkdir(parents=True, exist_ok=True)
    # clear previous numbered files in this dir for this run
    paths: list[Path] = []
    failures = 0
    for i, prompt in enumerate(prompts, 1):
        try:
            t1 = time.time()
            data = _generate_one_image(prompt)
            path = out_dir / f"{i:02d}.png"
            path.write_bytes(data)
            paths.append(path)
            log.info("scene %s/%s saved %s (%.1fs)", i, len(prompts), path.name, time.time() - t1)
            # gentle pacing for free-tier RPM
            time.sleep(1.5)
        except Exception as exc:
            failures += 1
            log.error("scene %s/%s FAILED: %s", i, len(prompts), str(exc)[:200])
            if failures >= 3 and not paths:
                raise

    if not paths:
        raise RuntimeError("all scene image generations failed")

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=p.name)
        # tiny manifest for debugging (not shown on images)
        manifest = {
            "title": title_ar[:200],
            "scenes": len(paths),
            "model": IMAGE_MODEL,
            "aspect": ASPECT,
            "words_per_scene": wps,
            "chunk_previews": [c[:160] for c in chunks[: len(paths)]],
        }
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    zip_bytes = buf.getvalue()
    meta = {
        "scenes": len(paths),
        "failures": failures,
        "seconds": round(time.time() - t0, 1),
        "zip_bytes": len(zip_bytes),
        "model": IMAGE_MODEL,
        "out_dir": str(out_dir),
        "paths": [str(p) for p in paths],
    }
    log.info(
        "story images zip ready scenes=%s failures=%s zip_kb=%.0f duration_s=%s",
        meta["scenes"],
        failures,
        len(zip_bytes) / 1024,
        meta["seconds"],
    )
    return zip_bytes, paths, meta
