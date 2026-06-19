# Quill

![Quill Logo](static/logo.jpeg)

A command-line tool for transcribing audio files, YouTube videos, and podcasts
using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

![Demo](static/demo.gif)

Quill runs on **macOS, Linux, and Windows**. By default it transcribes on the
CPU, so it works everywhere out of the box. On Linux and Windows machines with
an NVIDIA GPU, it can optionally use CUDA for a large speed-up.

## Quickstart

```bash
# 1. Install uv (the only tool you need — see per-OS instructions below)
# 2. Install ffmpeg (see per-OS instructions below)

# 3. Clone and install Quill
git clone https://github.com/cleanunicorn/quill.git
cd quill
uv sync

# 4. Transcribe something
uv run quill audio.mp3 transcript.txt
```

That's it. The first run downloads the transcription model automatically.

## Prerequisites

Quill needs two things on your system:

1. **[uv](https://docs.astral.sh/uv/)** — a fast Python package manager that
   also manages the Python version for you. You do **not** need to install
   Python separately; uv handles it.
2. **[ffmpeg](https://ffmpeg.org/)** — used to decode and convert audio.

Follow the section for your operating system below.

### macOS

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ffmpeg (via Homebrew — https://brew.sh)
brew install ffmpeg
```

> On macOS, Quill runs on the CPU. Apple Silicon is well-supported and fast for
> the smaller models. The `--device cuda` option is not available on macOS
> (CUDA is NVIDIA-only).

### Linux

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ffmpeg
sudo apt update && sudo apt install -y ffmpeg     # Debian / Ubuntu
# sudo dnf install ffmpeg                          # Fedora
# sudo pacman -S ffmpeg                            # Arch
```

For GPU acceleration, see [GPU / CUDA acceleration](#gpu--cuda-acceleration).

### Windows

Using [PowerShell](https://learn.microsoft.com/powershell/):

```powershell
# Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install ffmpeg (via winget; or use Chocolatey / Scoop)
winget install --id=Gyan.FFmpeg -e
```

After installing, **open a new terminal** so the updated `PATH` takes effect.

For GPU acceleration, see [GPU / CUDA acceleration](#gpu--cuda-acceleration).

## Installation

Once the prerequisites are in place:

```bash
git clone https://github.com/cleanunicorn/quill.git
cd quill

# Install dependencies into a project-local virtual environment.
# uv picks the right Python version automatically.
uv sync
```

Run Quill with `uv run`:

```bash
uv run quill --help
```

### Optional: install Quill globally

To call `quill` from anywhere without `uv run`:

```bash
uv tool install .
```

Make sure uv's tool directory is on your `PATH` (uv prints a hint the first
time; you can also run `uv tool update-shell`). After that, `quill --help`
works from any directory.

## Usage

```bash
uv run quill INPUT_SOURCE [OUTPUT_FILE] [OPTIONS]
```

If `OUTPUT_FILE` is omitted, Quill writes to a `.txt` file derived from the
input name (or the YouTube video title).

```bash
# Local audio file
uv run quill audio.mp3 transcript.txt

# YouTube video (output name defaults to the video title)
uv run quill "https://youtube.com/watch?v=..."

# Direct URL to an audio file, using a larger model
uv run quill https://example.com/audio.mp3 transcript.txt --model large

# Include timestamps, force English, run on GPU (Linux/Windows + NVIDIA)
uv run quill podcast.mp3 output.txt --device cuda --language en --timestamps
```

### Options

| Option              | Default  | Description                                                        |
| ------------------- | -------- | ------------------------------------------------------------------ |
| `-m`, `--model`     | `medium` | Model size: `tiny`, `base`, `small`, `medium`, `large`.            |
| `-d`, `--device`    | `auto`   | Inference device: `auto`, `cpu`, `cuda`. `auto` picks the best one.|
| `-l`, `--language`  | auto     | Language code (e.g. `en`, `fr`, `de`). Defaults to auto-detect.    |
| `-t`, `--timestamps`| off      | Include `[HH:mm:ss -> HH:mm:ss]` timestamps in the output.         |

### Supported input sources

- Local audio files (`mp3`, `wav`, `m4a`, etc.)
- YouTube URLs (video, shorts, `youtu.be`)
- Direct URLs to audio files

## GPU / CUDA acceleration

GPU acceleration is **optional** and available on **Linux and Windows** with an
NVIDIA GPU. The required NVIDIA Python libraries (cuDNN) are installed
automatically on those platforms; on macOS they are skipped, so the install
stays lean and CPU-only.

To use the GPU, you need a working NVIDIA driver plus the CUDA 12 runtime on
your system. On Debian/Ubuntu:

```bash
sudo apt install libcublas12 cudnn9-cuda-12
```

See NVIDIA's guides for details:
[CUDA 12](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) ·
[cuDNN](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/).

Then run with `--device cuda`:

```bash
uv run quill podcast.mp3 output.txt --device cuda
```

## Performance notes

- A CUDA-enabled NVIDIA GPU significantly improves transcription speed.
- Larger models are more accurate but use more memory and time.
- `medium` is a good balance for most use cases; on a CPU, try `tiny`, `base`,
  or `small` first if speed matters more than accuracy.

## Troubleshooting

**`ffmpeg` not found** — Quill (and YouTube downloads) need ffmpeg on your
`PATH`. Reinstall it for your OS (see [Prerequisites](#prerequisites)) and open
a new terminal.

**`uv: command not found`** — Open a new terminal after installing uv, or add
its install directory to your `PATH` (the installer prints the location).

**Install fails mentioning `nvidia-cudnn`** — You should not see this anymore;
the CUDA dependency is gated to Linux and Windows. If you do, make sure you are
on the latest version and run `uv sync` again.

**CUDA / cuDNN errors at runtime on Linux/Windows** — Ensure the system CUDA 12
runtime and cuDNN are installed (see
[GPU / CUDA acceleration](#gpu--cuda-acceleration)), or fall back to
`--device cpu`.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set
up a development environment.

## License

[Apache 2.0](LICENSE)
