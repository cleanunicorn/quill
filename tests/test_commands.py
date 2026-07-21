from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

import app.cli.commands as commands
from app.cli.commands import resolve_input, transcribe


class FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio, beam_size, language):
        segments = iter(
            [
                SimpleNamespace(start=0.0, end=2.5, text=" hello"),
                SimpleNamespace(start=2.5, end=5.0, text=" world"),
            ]
        )
        info = SimpleNamespace(duration=5.0, language="en", language_probability=0.9)
        return segments, info


def test_resolve_input_local_file_defaults_output(tmp_path):
    audio_path, output_path = resolve_input("podcast.mp3", None, tmp_path)
    assert audio_path == Path("podcast.mp3")
    assert output_path == Path("podcast.txt")


def test_resolve_input_local_file_strips_directory_from_default(tmp_path):
    audio_path, output_path = resolve_input("/some/dir/audio.mp3", None, tmp_path)
    assert audio_path == Path("/some/dir/audio.mp3")
    assert output_path == Path("audio.txt")


def test_resolve_input_local_file_keeps_explicit_output(tmp_path):
    _, output_path = resolve_input("audio.mp3", "custom.txt", tmp_path)
    assert output_path == Path("custom.txt")


def test_resolve_input_youtube_uses_sanitized_title(tmp_path, monkeypatch):
    def fake_download(url, target):
        return target.with_suffix(".wav"), "My: Video/Title"

    monkeypatch.setattr(commands, "download_youtube_audio", fake_download)
    audio_path, output_path = resolve_input("https://youtu.be/dQw4w9WgXcQ", None, tmp_path)
    assert audio_path == tmp_path / "audio.wav"
    assert output_path == Path("My VideoTitle.txt")


def test_resolve_input_youtube_empty_title_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(
        commands, "download_youtube_audio", lambda url, target: (target.with_suffix(".wav"), "!!!")
    )
    _, output_path = resolve_input("https://youtu.be/dQw4w9WgXcQ", None, tmp_path)
    assert output_path == Path("transcript.txt")


def test_resolve_input_youtube_keeps_explicit_output(tmp_path, monkeypatch):
    monkeypatch.setattr(
        commands,
        "download_youtube_audio",
        lambda url, target: (target.with_suffix(".wav"), "Title"),
    )
    _, output_path = resolve_input("https://youtu.be/dQw4w9WgXcQ", "given.txt", tmp_path)
    assert output_path == Path("given.txt")


def test_resolve_input_direct_url_uses_url_stem(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "download_file", lambda url, target: target)
    audio_path, output_path = resolve_input("https://example.com/audio.mp3?token=x", None, tmp_path)
    assert audio_path == tmp_path / "audio"
    assert output_path == Path("audio.txt")


def test_resolve_input_direct_url_without_path_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "download_file", lambda url, target: target)
    _, output_path = resolve_input("https://example.com/", None, tmp_path)
    assert output_path == Path("transcript.txt")


def test_transcribe_writes_plain_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "WhisperModel", FakeWhisperModel)
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("audio.mp3").touch()
        result = runner.invoke(transcribe, ["audio.mp3"])
        assert result.exit_code == 0
        assert Path("audio.txt").read_text(encoding="utf-8") == " hello  world "
        assert not Path("audio.txt.part").exists()


def test_transcribe_writes_timestamped_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "WhisperModel", FakeWhisperModel)
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("audio.mp3").touch()
        result = runner.invoke(transcribe, ["audio.mp3", "out.txt", "--timestamps"])
        assert result.exit_code == 0
        assert Path("out.txt").read_text(encoding="utf-8") == (
            "[00:00:00 -> 00:00:02] hello\n[00:00:02 -> 00:00:05] world\n"
        )
