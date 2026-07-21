# Contributing to Quill

Thanks for your interest in improving Quill! This guide covers setting up a
development environment on macOS, Linux, and Windows.

## Prerequisites

You need two tools installed (see the
[README prerequisites](README.md#prerequisites) for per-OS commands):

- [uv](https://docs.astral.sh/uv/) — manages Python and dependencies.
- [ffmpeg](https://ffmpeg.org/) — used for audio decoding.

## Setup

```bash
git clone https://github.com/cleanunicorn/quill.git
cd quill

# Install all dependencies into a project-local .venv.
# uv resolves and pins the correct Python version automatically.
uv sync
```

Run the CLI from the source tree with:

```bash
uv run quill --help
```

## Project layout

```
app/
  cli/
    commands.py   # Click command + transcription pipeline
    utils.py      # URL detection, YouTube + file downloads, pure helpers
tests/            # pytest suite (unit + CLI tests, no network access)
pyproject.toml    # Project metadata, dependencies, ruff/pytest config
```

The console entry point is defined in `pyproject.toml` under
`[project.scripts]` as `quill = "app.cli.commands:transcribe"`.

## Cross-platform notes

Quill must install and run on macOS, Linux, and Windows.

- **Default installs are CPU-only and platform-independent.** Keep it that way.
- The NVIDIA cuDNN dependency lives in the optional `cuda` extra and is gated
  with a platform marker (`sys_platform == 'linux' or sys_platform == 'win32'`)
  because the wheels only exist for Linux and Windows. **Do not add
  unconditional GPU/CUDA dependencies** — that breaks installation on macOS.
- If you add a dependency that ships platform-specific wheels, gate it with the
  appropriate marker and confirm `uv sync` still resolves on all three OSes.

## Before opening a pull request

```bash
# Make sure the lockfile is up to date with pyproject.toml
uv lock

# Confirm a clean install and that the CLI loads
uv sync
uv run quill --help

# Lint, check formatting, and run the tests
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The [CI workflow](.github/workflows/ci.yml) runs the lint, format, and test
checks for every pull request. Commit messages follow
[Conventional Commits](https://www.conventionalcommits.org/) (`type(scope): subject`).

## Reporting issues

When filing a bug, please include your operating system, the output of
`uv --version`, the full command you ran, and the complete error message.
