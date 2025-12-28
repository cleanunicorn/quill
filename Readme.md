# Quill

![Quill Logo](static/logo.jpeg)

A command-line tool for transcribing audio files, YouTube videos, and podcasts using Faster Whisper.

![Demo](static/demo.gif)

## Prerequisites

- ffmpeg
- Python 3.10+

## Installation

At system level you need [cuda 12](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu) and [cuddn](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html#ubuntu-and-debian-network-installation).

```bash
sudo apt install libcublas12
sudo apt install cudnn9-cuda-12
```

```bash
# Install dependencies and create virtual environment
uv sync

# Optional: Install globally as a CLI tool
uv tool install . --python 3.12
```

## Usage

```bash
uv run quill INPUT_SOURCE OUTPUT_FILE [--model MODEL] [--device DEVICE] [--language LANGUAGE]

# Examples:
uv run quill audio.mp3 transcript.txt
uv run quill https://youtube.com/watch?v=... transcript.txt
uv run quill https://example.com/audio.mp3 transcript.txt --model large
uv run quill podcast.mp3 output.txt --device cuda --language en
```

Options:

- `--model`: Model size to use (tiny, base, small, medium, large). Default: medium
- `--device`: Device to use for inference (cpu, cuda). Default: cpu
- `--language`: Language code for transcription (e.g., en, fr, de). Default: auto-detect
- `--timestamps, -t`: Include timestamps in the transcription output

## Supported Input Sources

- Local audio files (mp3, wav, m4a, etc.)
- YouTube URLs
- Direct URLs to audio files

## Performance Notes

- Using CUDA-enabled GPU significantly improves transcription speed
- Larger models provide better accuracy but require more memory and processing time
- The 'medium' model provides a good balance between speed and accuracy for most use cases

## License

Apache 2.0
