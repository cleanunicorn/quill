from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO
from urllib.parse import urlparse

import click
from faster_whisper import WhisperModel

from app.cli.utils import (
    download_file,
    download_youtube_audio,
    is_url,
    is_youtube_url,
    sanitize_filename,
    seconds_to_timestamp,
)


def resolve_input(input_source: str, output_file: str | None, temp_dir: Path) -> tuple[Path, Path]:
    """Resolve the input source to a local audio path and an output file path.

    Downloads remote sources into ``temp_dir`` and derives a default output
    filename when one was not provided.
    """
    if is_url(input_source):
        download_target = temp_dir / "audio"
        if is_youtube_url(input_source):
            click.echo("Detected YouTube URL. Downloading audio...")
            audio_path, video_title = download_youtube_audio(input_source, download_target)
            default_stem = sanitize_filename(video_title) or "transcript"
        else:
            click.echo("Downloading audio file...")
            audio_path = download_file(input_source, download_target)
            default_stem = sanitize_filename(Path(urlparse(input_source).path).stem) or "transcript"
    else:
        audio_path = Path(input_source)
        default_stem = audio_path.stem

    return audio_path, Path(output_file if output_file is not None else f"{default_stem}.txt")


def write_transcript(segments: Iterable, f: TextIO, timestamps: bool) -> None:
    """Echo each transcribed segment and write it to the open transcript file."""
    for segment in segments:
        if timestamps:
            start_time = seconds_to_timestamp(segment.start)
            end_time = seconds_to_timestamp(segment.end)
            line = f"[{start_time} -> {end_time}] {segment.text.strip()}"
            click.echo(line)
            f.write(line + "\n")
        else:
            click.echo(segment.text)
            f.write(segment.text + " ")


@click.command()
@click.version_option(package_name="quill")
@click.argument("input_source")
@click.argument("output_file", required=False)
@click.option(
    "--model",
    "-m",
    default="medium",
    help="Model size to use (tiny, base, small, medium, large)",
    show_default=True,
)
@click.option(
    "--device",
    "-d",
    default="auto",
    type=click.Choice(["auto", "cpu", "cuda"]),
    help="Device to use for inference",
    show_default=True,
)
@click.option(
    "--language",
    "-l",
    default=None,
    help="Language code for transcription (e.g., en, fr, de). Default: auto-detect",
)
@click.option(
    "--timestamps",
    "-t",
    is_flag=True,
    help="Include timestamps in the transcription output",
)
def transcribe(
    input_source: str,
    output_file: str | None,
    model: str,
    device: str,
    language: str | None,
    timestamps: bool,
) -> None:
    """
    Transcribe audio from a file or URL to text.

    INPUT_SOURCE: Path to local audio file or URL to audio file (including YouTube URLs)

    OUTPUT_FILE: Path where the transcription will be saved
    """
    try:
        with tempfile.TemporaryDirectory(prefix="quill-") as temp_dir:
            audio_path, output_path = resolve_input(input_source, output_file, Path(temp_dir))

            click.echo("Loading model...")
            whisper_model = WhisperModel(model, device=device, compute_type="auto")

            click.echo("Transcribing audio...")
            segments, info = whisper_model.transcribe(
                str(audio_path), beam_size=5, language=language
            )

            click.echo(f"Duration: {info.duration:.2f} seconds")
            if language is None:
                click.echo(
                    f"Detected language '{info.language}' "
                    f"with probability {info.language_probability}"
                )

            with output_path.open("w", encoding="utf-8") as f:
                click.echo("\nTranscription:")
                write_transcript(segments, f, timestamps)

        click.echo(f"\nTranscription saved to: {output_path}")

    except KeyboardInterrupt:
        click.echo("\nTranscription cancelled by user.")
        raise click.Abort() from None

    except click.ClickException:
        raise

    except Exception as e:
        raise click.ClickException(str(e)) from e
