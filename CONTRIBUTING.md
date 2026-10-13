# Contributing to Quill

Thanks for your interest in improving Quill! This guide covers setting up a
development environment on macOS, Linux, and Windows.

## Prerequisites

You need two tools installed (see the
[README prerequisites](README.md#prerequisites) for per-OS commands):

- [uv](https://docs.astral.sh/uv/) — manages Python and dependencies.
- [ffmpeg](https://ffmpeg.org/) — used for audio decoding.
- GNU `make` — runs the dev tasks (preinstalled on macOS and most Linux
  distros; on Windows, use the `uv run` commands from the Makefile directly).

## Setup

```bash
git clone https://github.com/cleanunicorn/quill.git
cd quill

# Install all dependencies into a project-local .venv.
# uv resolves and pins the correct Python version automatically.
make sync
```

Run the CLI from the source tree with:

```bash
make run ARGS="--help"
```

`make help` lists every task. [AGENTS.md](AGENTS.md) is the full working guide
(branching, commit and PR-title rules, release automation, and the project's
hazards) for humans and AI agents alike.

## Project layout

```
app/
  cli/
    commands.py   # Click command + transcription pipeline
    utils.py      # URL detection, YouTube + file downloads, pure helpers
tests/            # pytest suite (unit + CLI tests, no network access)
pyproject.toml    # Project metadata, dependencies, ruff/pytest/release config
Makefile          # Dev tasks: sync, run, lint, fmt, test, check
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
# If you changed dependencies in pyproject.toml, refresh the lockfile
uv lock

# Format, then run everything CI runs: lockfile check, lint, format check, tests
make fmt
make check
```

The [CI workflow](.github/workflows/ci.yml) runs `make check` on Ubuntu and
macOS and the tests on Windows for every pull request. Commit messages **and PR
titles** follow [Conventional Commits](https://www.conventionalcommits.org/)
(`type(scope): subject`); a check enforces the PR title, and the title decides
the release version when the PR merges (see
[AGENTS.md](AGENTS.md#release-automation)).

## Reporting issues

When filing a bug, please include your operating system, the output of
`uv --version`, the full command you ran, and the complete error message.
