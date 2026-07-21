from __future__ import annotations

import re
import shutil
import urllib.request
from pathlib import Path

import click
import yt_dlp

YOUTUBE_PATTERNS = [
    re.compile(r"^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+"),
    re.compile(r"^https?://(?:www\.)?youtube\.com/v/[\w-]+"),
    re.compile(r"^https?://youtu\.be/[\w-]+"),
    re.compile(r"^https?://(?:www\.)?youtube\.com/shorts/[\w-]+"),
]


def is_youtube_url(url: str) -> bool:
    """Check if the URL is a YouTube URL."""
    return any(pattern.match(url) for pattern in YOUTUBE_PATTERNS)


def sanitize_filename(filename: str) -> str:
    """Sanitize a string to be used as a filename."""
    filename = "".join(c for c in filename if c.isalnum() or c in (" ", "-", "_"))
    return filename.strip()


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:mm:ss format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def is_url(string: str) -> bool:
    """Check if a string is a URL."""
    return string.startswith(("http://", "https://"))


def download_youtube_audio(url: str, temp_dir: Path) -> tuple[Path, str]:
    """Download audio from a YouTube video into ``temp_dir``.

    Returns the path to the extracted audio file and the video title.
    """
    output_path = temp_dir / "audio"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_path),
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url)
        return output_path.with_suffix(".wav"), info.get("title", "video")
    except Exception as e:
        raise click.ClickException(f"Failed to download YouTube audio: {e}") from e


def download_file(url: str, local_path: Path) -> Path:
    """Download a file from a URL to a local path."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response, local_path.open("wb") as f:
            shutil.copyfileobj(response, f)
        return local_path
    except Exception as e:
        raise click.ClickException(f"Failed to download file: {e}") from e
