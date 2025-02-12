import click
from faster_whisper import WhisperModel
import os
from app.cli.utils import is_youtube_url, download_youtube_audio, download_file, is_url
from pathlib import Path


def sanitize_filename(filename):
    """Sanitize a string to be used as a filename."""
    # Replace spaces with underscores and remove/replace invalid characters
    filename = "".join(c for c in filename if c.isalnum() or c in (" ", "-", "_"))
    return filename.strip()


def seconds_to_timestamp(seconds):
    """Convert seconds to HH:mm:ss format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@click.command()
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
    input_source,
    output_file,
    model,
    device,
    language,
    timestamps,
):
    """
    Transcribe audio from a file or URL to text.

    INPUT_SOURCE: Path to local audio file or URL to audio file (including YouTube URLs)

    OUTPUT_FILE: Path where the transcription will be saved
    """
    temp_file = None
    try:
        # Handle URL input
        if is_url(input_source):
            if is_youtube_url(input_source):
                click.echo("Detected YouTube URL. Downloading audio...")
                temp_file = "temp_audio_file"
                input_source, video_title = download_youtube_audio(
                    input_source, temp_file
                )
                if output_file is None:
                    # Use video title for output file, sanitize it for filesystem
                    safe_title = sanitize_filename(video_title)
                    output_file = f"{safe_title}.txt"
            else:
                temp_file = "temp_audio_file"
                input_source = download_file(input_source, temp_file)
                if output_file is None:
                    # Extract filename from URL
                    output_file = Path(input_source).stem + ".txt"

        # If output_file is still None and it's a local file
        if output_file is None:
            # Use the input filename but change extension to .txt
            output_file = Path(input_source).stem + ".txt"

        # Initialize the model
        click.echo("Loading model...")
        whisper_model = WhisperModel(model, device=device, compute_type="auto")

        # Perform transcription
        click.echo("Transcribing audio...")
        segments, info = whisper_model.transcribe(
            input_source, beam_size=5, language=language
        )

        # Print duration
        click.echo(f"Duration: {info.duration:.2f} seconds")

        # Print detection info
        if language is None:
            click.echo(
                f"Detected language '{info.language}' with probability {info.language_probability}"
            )

        # Open the output file for writing
        with open(output_file, "w", encoding="utf-8") as f:
            click.echo("\nTranscription:")
            current_group = ""
            current_start = None

            for segment in segments:
                if timestamps:
                    if current_start is None:
                        current_start = segment.start
                    current_group += f"{segment.text} "
                    
                    # Output and store the current segment with HH:mm:ss format
                    start_time = seconds_to_timestamp(current_start)
                    end_time = seconds_to_timestamp(segment.end)
                    segment_text = f"[{start_time} -> {end_time}] {current_group.strip()}"
                    click.echo(segment_text)
                    f.write(segment_text + "\n")
                    
                    current_group = ""
                    current_start = None
                else:
                    # Output and store the current segment
                    click.echo(segment.text)
                    f.write(segment.text + " ")

        click.echo(f"\nTranscription saved to: {output_file}")

    except KeyboardInterrupt:
        click.echo("\nTranscription cancelled by user. Cleaning up...")
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
        raise click.Abort()

    except Exception as e:
        raise click.ClickException(str(e))

    finally:
        # Cleanup downloaded files
        if temp_file:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if os.path.exists(f"{temp_file}.wav"):
                os.remove(f"{temp_file}.wav")
