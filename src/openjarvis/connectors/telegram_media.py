"""YouTube lookup for streaming music/video into Telegram calls.

The actual streaming is done by py-tgcalls (``MediaStream`` resolves YouTube
links through ``yt-dlp`` and decodes with ``ffmpeg``); this module only turns
"a song name" or a link into one concrete video with title and duration, so
the agent can tell the user what is playing.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from openjarvis.connectors.telegram_user import TelegramUserError

_SEARCH_TIMEOUT_S = 45


@dataclass(frozen=True)
class MediaInfo:
    url: str
    title: str
    duration_s: int | None
    channel: str | None
    is_live: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "duration_s": self.duration_s,
            "channel": self.channel,
            "live": self.is_live,
        }


def _yt_dlp() -> str:
    path = shutil.which("yt-dlp")
    if path is None:
        raise TelegramUserError(
            "yt-dlp is not installed or not on PATH (install: pipx install yt-dlp)."
        )
    if shutil.which("ffmpeg") is None:
        raise TelegramUserError("ffmpeg is required for streaming but is not on PATH.")
    return path


_LINK_PREFIXES = ("http://", "https://", "www.", "youtu.be/", "youtube.com/")


def _is_link(query: str) -> bool:
    return query.startswith(_LINK_PREFIXES)


def resolve_youtube(query: str) -> MediaInfo:
    """Return the first YouTube match for *query* (a search or a link)."""
    query = query.strip()
    if not query:
        raise TelegramUserError("Say what to play (a song name or a YouTube link).")
    target = query if _is_link(query) else f"ytsearch1:{query}"
    cmd = [
        _yt_dlp(),
        "--dump-json",
        "--no-playlist",
        "--skip-download",
        "--no-warnings",
        target,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_SEARCH_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise TelegramUserError("YouTube lookup timed out.") from exc
    line = next((ln for ln in proc.stdout.splitlines() if ln.strip()), "")
    if proc.returncode != 0 or not line:
        lines = (proc.stderr or "").strip().splitlines()
        detail = lines[-1] if lines else "no results"
        raise TelegramUserError(f"Nothing found on YouTube for {query!r}: {detail}")
    data = json.loads(line)
    url = data.get("webpage_url") or data.get("original_url")
    if not url:
        raise TelegramUserError(f"Nothing playable found for {query!r}.")
    duration = data.get("duration")
    return MediaInfo(
        url=url,
        title=str(data.get("title") or url),
        duration_s=int(duration) if isinstance(duration, (int, float)) else None,
        channel=data.get("channel") or data.get("uploader"),
        is_live=bool(data.get("is_live")),
    )


# ---------------------------------------------------------------------------
# Local media folders ([tools.telegram_user] media_dirs)
# ---------------------------------------------------------------------------

MEDIA_EXTENSIONS = frozenset(
    {
        ".mp3",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".wav",
        ".flac",
        ".mp4",
        ".mkv",
        ".webm",
        ".mov",
        ".avi",
    }
)
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".webm", ".mov", ".avi"})
# Words people say around a title that never appear in file names.
_FILLER = frozenset(
    {
        "play",
        "the",
        "a",
        "an",
        "song",
        "track",
        "music",
        "video",
        "episode",
        "ep",
        "part",
        "from",
        "local",
        "folder",
        "file",
        "please",
        "with",
        "of",
        "my",
    }
)
_MIN_LOCAL_SCORE = 0.6


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    # "03" and "3" should match each other.
    return [w.lstrip("0") or "0" if w.isdigit() else w for w in words]


def iter_local_media(dirs: Iterable[str]) -> list[Path]:
    """All playable files under *dirs* (recursively), sorted by name."""
    files: list[Path] = []
    for raw in dirs:
        root = Path(os.path.expanduser(raw))
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS:
                files.append(path)
    return sorted(files, key=lambda p: p.name.lower())


def _score(query: str, path: Path) -> float:
    wanted = [t for t in _tokens(query) if t not in _FILLER]
    if not wanted:
        return 0.0
    name = _tokens(path.stem)
    hits = sum(1 for t in wanted if t in name)
    ratio = difflib.SequenceMatcher(None, " ".join(wanted), " ".join(name)).ratio()
    return hits / len(wanted) * 0.8 + ratio * 0.2


def probe_duration(path: Path) -> Optional[int]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        out = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout.strip()
        return int(float(out)) if out else None
    except (subprocess.TimeoutExpired, ValueError):
        return None


def find_local_media(query: str, dirs: Iterable[str]) -> Optional[MediaInfo]:
    """Best local match for *query*, or ``None`` if nothing is close enough."""
    dirs = list(dirs)
    candidates = iter_local_media(dirs)
    if not candidates:
        return None
    scored = sorted(((_score(query, p), p) for p in candidates), key=lambda x: -x[0])
    best_score, best = scored[0]
    if best_score < _MIN_LOCAL_SCORE:
        return None
    return local_media_info(best, dirs)


def local_media_info(path: Path, dirs: Iterable[str]) -> MediaInfo:
    """Describe *path*, refusing anything outside the configured folders."""
    resolved = path.expanduser().resolve()
    roots = [Path(os.path.expanduser(d)).resolve() for d in dirs]
    if not any(resolved.is_relative_to(root) for root in roots):
        raise TelegramUserError(f"{path} is not inside a configured media folder.")
    if not resolved.is_file():
        raise TelegramUserError(f"File not found: {path}")
    return MediaInfo(
        url=str(resolved),
        title=resolved.stem,
        duration_s=probe_duration(resolved),
        channel="local",
        is_live=False,
    )


def match_local_media(query: str, dirs: Iterable[str]) -> Optional[Path]:
    """Exact-ish match for destructive actions (stricter than playback)."""
    candidates = iter_local_media(dirs)
    scored = sorted(((_score(query, p), p) for p in candidates), key=lambda x: -x[0])
    if not scored or scored[0][0] < 0.85:
        return None
    if len(scored) > 1 and scored[1][0] >= scored[0][0] - 0.02:
        return None  # ambiguous, e.g. two episodes with near-identical names
    return scored[0][1]


def safe_media_stem(name: str) -> str:
    """File-system-safe base name (no paths, no control characters)."""
    stem = Path(name).name
    stem = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", stem).strip(" .") or "media"
    return stem[:150]


def describe_local_media(dirs: Iterable[str], query: str = "") -> list[dict[str, Any]]:
    files = iter_local_media(dirs)
    if query:
        files = [p for p in files if _score(query, p) >= _MIN_LOCAL_SCORE / 2]
    return [
        {
            "title": p.stem,
            "type": "video" if p.suffix.lower() in _VIDEO_EXTENSIONS else "audio",
            "size_mb": round(p.stat().st_size / 1_048_576, 1),
            "path": str(p),
        }
        for p in files
    ]


# ---------------------------------------------------------------------------
# Transport commands ("forward 30 seconds", "go to 1:23", "pause", ...)
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "forty-five": 45,
    "fifty": 50,
    "sixty": 60,
    "ninety": 90,
}
_UNIT_SECONDS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
}
_AMOUNT = re.compile(
    r"(?P<half>half (?:a|an) minute)|"
    r"(?P<num>\d+(?:\.\d+)?|"
    + "|".join(sorted(map(re.escape, _NUMBER_WORDS), key=len, reverse=True))
    + r")\s*(?P<unit>seconds?|secs?|s|minutes?|mins?|m)?\b",
    re.IGNORECASE,
)
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")
_FORWARD = re.compile(
    r"\b(forward|fast[- ]?forward|skip (?:ahead|forward)|ahead|jump ahead)\b", re.I
)
_REWIND = re.compile(r"\b(rewind|back(?:wards?)?|go back|skip back|replay)\b", re.I)
_SEEK_TO = re.compile(r"\b(go to|jump to|seek to|skip to|start (?:at|from))\b", re.I)
_PAUSE = re.compile(r"\b(pause|hold on|hold)\b", re.I)
_RESUME = re.compile(r"\b(resume|continue|unpause|play)\b", re.I)
_STOP = re.compile(r"\b(stop|enough|skip|turn (?:it )?off)\b", re.I)
DEFAULT_STEP_S = 10.0


def parse_timestamp(text: str) -> Optional[float]:
    """ "1:23" / "1:02:03" / "90" / "90s" / "2 minutes" → seconds."""
    text = str(text).strip()
    clock = _CLOCK.search(text)
    if clock:
        h, m, sec = (
            (int(clock.group(1)), int(clock.group(2)), int(clock.group(3)))
            if clock.group(3)
            else (0, int(clock.group(1)), int(clock.group(2)))
        )
        return float(h * 3600 + m * 60 + sec)
    amount = _AMOUNT.search(text)
    if amount is None:
        return None
    if amount.group("half"):
        return 30.0
    raw = amount.group("num").lower()
    value = float(raw) if raw[0].isdigit() else float(_NUMBER_WORDS[raw])
    unit = (amount.group("unit") or "s").lower()
    return value * _UNIT_SECONDS.get(unit, 1)


def parse_media_command(text: str) -> Optional[tuple[str, Optional[float]]]:
    """Map a spoken/typed command to ``(action, seconds)``.

    Actions: ``seek_by`` (signed seconds), ``seek_to`` (absolute seconds),
    ``pause``, ``resume``, ``stop``. Returns ``None`` for anything else.
    """
    forward, rewind = _FORWARD.search(text), _REWIND.search(text)
    if forward or rewind:
        amount = parse_timestamp(text)
        step = amount if amount is not None else DEFAULT_STEP_S
        return ("seek_by", step if forward and not rewind else -step)
    if _SEEK_TO.search(text):
        target = parse_timestamp(text)
        if target is not None:
            return ("seek_to", target)
    if _PAUSE.search(text):
        return ("pause", None)
    if _STOP.search(text):
        return ("stop", None)
    if _RESUME.search(text):
        return ("resume", None)
    return None


def format_position(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    total = int(max(0, seconds))
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


__all__ = [
    "MEDIA_EXTENSIONS",
    "match_local_media",
    "safe_media_stem",
    "describe_local_media",
    "find_local_media",
    "iter_local_media",
    "local_media_info",
    "MediaInfo",
    "format_position",
    "parse_media_command",
    "parse_timestamp",
    "resolve_youtube",
]
