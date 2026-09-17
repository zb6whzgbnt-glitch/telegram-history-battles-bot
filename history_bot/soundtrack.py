from __future__ import annotations

"""
Build a documentary war bed (drums + tension drone + sword-like hits)
and mix it under Gemini narration. FX are synthesized with ffmpeg only.
"""

import logging
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("history_bot.soundtrack")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _duration_seconds(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return max(float(out), 1.0)


def build_war_bed(duration: float, out_wav: Path) -> None:
    dur = max(duration + 1.5, 8.0)
    filt = (
        f"aevalsrc=exprs='0.04*sin(2*PI*55*t)+0.025*sin(2*PI*82*t)+0.015*sin(2*PI*110*t)':"
        f"s=44100:d={dur}[drone];"
        f"aevalsrc=exprs='0.55*sin(2*PI*70*t)*exp(-6*mod(t\\,0.85))':"
        f"s=44100:d={dur}[drums];"
        f"aevalsrc=exprs='0.22*sin(2*PI*(900+400*sin(2*PI*3*t))*t)*exp(-18*mod(t+0.4\\,2.4))':"
        f"s=44100:d={dur}[swords];"
        f"[drone][drums][swords]amix=inputs=3:normalize=0:duration=longest,"
        f"highpass=f=40,lowpass=f=6000,volume=0.9,afade=t=in:st=0:d=1.2,"
        f"afade=t=out:st={max(dur-1.8,0)}:d=1.8"
    )
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-filter_complex", filt,
        "-ac", "1",
        str(out_wav),
    ])


def mix_documentary(voice_ogg: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        voice_in = td_path / "voice.ogg"
        voice_wav = td_path / "voice.wav"
        bed_wav = td_path / "bed.wav"
        mixed_wav = td_path / "mixed.wav"
        out_ogg = td_path / "out.ogg"
        voice_in.write_bytes(voice_ogg)

        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(voice_in),
            "-ac", "1", "-ar", "44100",
            str(voice_wav),
        ])
        dur = _duration_seconds(voice_wav)
        build_war_bed(dur, bed_wav)

        filt = (
            "[0:a]volume=1.0,afade=t=in:st=0:d=0.15[v];"
            "[1:a]volume=0.18[b];"
            "[v][b]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
            "loudnorm=I=-16:TP=-1.5:LRA=11"
        )
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(voice_wav),
            "-i", str(bed_wav),
            "-filter_complex", filt,
            "-ar", "48000", "-ac", "1",
            str(mixed_wav),
        ])
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(mixed_wav),
            "-c:a", "libopus", "-b:a", "48k",
            str(out_ogg),
        ])
        return out_ogg.read_bytes()
