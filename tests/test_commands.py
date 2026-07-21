from pathlib import Path
from types import SimpleNamespace

import click
import pytest
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
    source = tmp_path / "podcast.mp3"
    source.touch()
    audio_path, output_path = resolve_input(str(source), None, tmp_path)
    assert audio_path == source
    assert output_path == Path("podcast.txt")


def test_resolve_input_local_file_strips_directory_from_default(tmp_path):
    source = tmp_path / "some" / "dir" / "audio.mp3"
    source.parent.mkdir(parents=True)
    source.touch()
    audio_path, output_path = resolve_input(str(source), None, tmp_path)
    assert audio_path == source
    assert output_path == Path("audio.txt")


def test_resolve_input_local_file_keeps_explicit_output(tmp_path):
    source = tmp_path / "audio.mp3"
    source.touch()
    _, output_path = resolve_input(str(source), "custom.txt", tmp_path)
    assert output_path == Path("custom.txt")


def test_resolve_input_missing_local_file_fails_fast(tmp_path):
    with pytest.raises(click.ClickException, match="Input file not found"):
        resolve_input(str(tmp_path / "nope.mp3"), None, tmp_path)


def test_resolve_input_rejects_directory_output(tmp_path):
    source = tmp_path / "audio.mp3"
    source.touch()
    target_dir = tmp_path / "existing_dir"
    target_dir.mkdir()
    with pytest.raises(click.ClickException, match="is a directory"):
        resolve_input(str(source), str(target_dir), tmp_path)


def test_resolve_input_rejects_missing_output_directory(tmp_path):
    source = tmp_path / "audio.mp3"
    source.touch()
    with pytest.raises(click.ClickException, match="does not exist"):
        resolve_input(str(source), str(tmp_path / "no_such_dir" / "out.txt"), tmp_path)


def test_resolve_input_youtube_uses_sanitized_title(tmp_path, monkeypatch):
    def fake_download(url, temp_dir):
        return temp_dir / "audio.wav", "My: Video/Title"

    monkeypatch.setattr(commands, "download_youtube_audio", fake_download)
    audio_path, output_path = resolve_input("https://youtu.be/dQw4w9WgXcQ", None, tmp_path)
    assert audio_path == tmp_path / "audio.wav"
    assert output_path == Path("My VideoTitle.txt")


def test_resolve_input_youtube_empty_title_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(
        commands, "download_youtube_audio", lambda url, temp_dir: (temp_dir / "audio.wav", "!!!")
    )
    _, output_path = resolve_input("https://youtu.be/dQw4w9WgXcQ", None, tmp_path)
    assert output_path == Path("transcript.txt")


def test_resolve_input_youtube_keeps_explicit_output(tmp_path, monkeypatch):
    monkeypatch.setattr(
        commands,
        "download_youtube_audio",
        lambda url, temp_dir: (temp_dir / "audio.wav", "Title"),
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


def test_version_flag_reports_package_version():
    result = CliRunner().invoke(transcribe, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


class FailingWhisperModel(FakeWhisperModel):
    def transcribe(self, audio, beam_size, language):
        def segment_gen():
            yield SimpleNamespace(start=0.0, end=2.5, text=" hello")
            raise RuntimeError("decode failed")

        info = SimpleNamespace(duration=5.0, language="en", language_probability=0.9)
        return segment_gen(), info


def test_transcribe_failure_keeps_previous_output_and_cleans_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "WhisperModel", FailingWhisperModel)
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("audio.mp3").touch()
        Path("out.txt").write_text("old content", encoding="utf-8")
        result = runner.invoke(transcribe, ["audio.mp3", "out.txt"])
        assert result.exit_code == 1
        assert "Error: decode failed" in result.output
        assert Path("out.txt").read_text(encoding="utf-8") == "old content"
        assert not list(Path().glob("*.part"))


class InterruptedWhisperModel(FakeWhisperModel):
    def transcribe(self, audio, beam_size, language):
        def segment_gen():
            yield SimpleNamespace(start=0.0, end=2.5, text=" hello")
            raise KeyboardInterrupt

        info = SimpleNamespace(duration=5.0, language="en", language_probability=0.9)
        return segment_gen(), info


def test_transcribe_ctrl_c_exits_130_and_keeps_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "WhisperModel", InterruptedWhisperModel)
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("audio.mp3").touch()
        result = runner.invoke(transcribe, ["audio.mp3", "out.txt"])
        assert result.exit_code == 130
        assert "cancelled" in result.output
        assert not Path("out.txt").exists()
        partials = list(Path().glob("out.txt.*.part"))
        assert len(partials) == 1
        assert partials[0].read_text(encoding="utf-8") == " hello "


def test_transcribe_writes_plain_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "WhisperModel", FakeWhisperModel)
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("audio.mp3").touch()
        result = runner.invoke(transcribe, ["audio.mp3"])
        assert result.exit_code == 0
        assert Path("audio.txt").read_text(encoding="utf-8") == " hello  world "
        assert not list(Path().glob("*.part"))


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
