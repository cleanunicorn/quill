import io

import click
import pytest

from app.cli.utils import (
    download_file,
    download_youtube_audio,
    is_url,
    is_youtube_url,
    sanitize_filename,
    seconds_to_timestamp,
)


class FakeYoutubeDL:
    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def extract_info(self, url):
        return {"title": "My Video"}


def test_is_youtube_url_matches_common_formats():
    assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert is_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
    assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
    assert is_youtube_url("https://www.youtube.com/shorts/dQw4w9WgXcQ")
    assert is_youtube_url("http://www.youtube.com/v/dQw4w9WgXcQ")


def test_is_youtube_url_rejects_other_urls():
    assert not is_youtube_url("https://example.com/audio.mp3")
    assert not is_youtube_url("https://vimeo.com/12345")
    assert not is_youtube_url("not a url")


def test_is_url():
    assert is_url("https://example.com/audio.mp3")
    assert is_url("http://example.com")
    assert not is_url("audio.mp3")
    assert not is_url("/home/user/audio.mp3")


def test_sanitize_filename():
    assert sanitize_filename("My Video Title") == "My Video Title"
    assert sanitize_filename("What?! A/B\\C: Test") == "What ABC Test"
    assert sanitize_filename("  spaced  ") == "spaced"
    assert sanitize_filename("dash-and_underscore") == "dash-and_underscore"
    assert sanitize_filename("") == ""
    assert sanitize_filename("!!!") == ""
    assert sanitize_filename("😀🎉") == ""


def test_download_youtube_audio_returns_wav_path_and_title(tmp_path, monkeypatch):
    monkeypatch.setattr("app.cli.utils.yt_dlp.YoutubeDL", FakeYoutubeDL)
    audio_path, title = download_youtube_audio("https://youtu.be/x", tmp_path)
    assert audio_path == tmp_path / "audio.wav"
    assert title == "My Video"


def test_download_youtube_audio_missing_title_falls_back(tmp_path, monkeypatch):
    class NoTitleYoutubeDL(FakeYoutubeDL):
        def extract_info(self, url):
            return {}

    monkeypatch.setattr("app.cli.utils.yt_dlp.YoutubeDL", NoTitleYoutubeDL)
    _, title = download_youtube_audio("https://youtu.be/x", tmp_path)
    assert title == "video"


def test_download_youtube_audio_wraps_errors(tmp_path, monkeypatch):
    class FailingYoutubeDL(FakeYoutubeDL):
        def extract_info(self, url):
            raise RuntimeError("network down")

    monkeypatch.setattr("app.cli.utils.yt_dlp.YoutubeDL", FailingYoutubeDL)
    with pytest.raises(click.ClickException, match="Failed to download YouTube audio"):
        download_youtube_audio("https://youtu.be/x", tmp_path)


def test_download_file_streams_response_to_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda url, timeout=None: io.BytesIO(b"audio-bytes")
    )
    target = tmp_path / "audio"
    result = download_file("https://example.com/a.mp3", target)
    assert result == target
    assert target.read_bytes() == b"audio-bytes"


def test_download_file_wraps_errors(tmp_path, monkeypatch):
    def failing_urlopen(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    with pytest.raises(click.ClickException, match="Failed to download file"):
        download_file("https://example.com/a.mp3", tmp_path / "audio")


def test_seconds_to_timestamp():
    assert seconds_to_timestamp(0) == "00:00:00"
    assert seconds_to_timestamp(59.9) == "00:00:59"
    assert seconds_to_timestamp(61) == "00:01:01"
    assert seconds_to_timestamp(3661) == "01:01:01"
    assert seconds_to_timestamp(7325.5) == "02:02:05"
