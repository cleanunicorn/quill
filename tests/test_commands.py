from pathlib import Path

from app.cli.commands import resolve_input


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
