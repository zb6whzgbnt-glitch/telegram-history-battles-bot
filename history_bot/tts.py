from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path

def pcm_to_ogg(pcm: bytes, sample_rate: int = 24000) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "a.pcm"
        dst = Path(td) / "a.ogg"
        src.write_bytes(pcm)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
            "-i", str(src),
            "-c:a", "libopus", "-b:a", "48k",
            str(dst),
        ]
        subprocess.run(cmd, check=True)
        return dst.read_bytes()
