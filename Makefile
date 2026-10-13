.PHONY: help sync install run test lint fmt fmt-check lock-check check clean

# Extras for `make install`. Set QUILL_EXTRAS=cuda on Linux/Windows with an
# NVIDIA GPU; the extra is platform-gated, so it is a no-op on macOS.
QUILL_EXTRAS ?=
# Arguments for `make run`, e.g. make run ARGS="audio.mp3 out.txt --model tiny"
ARGS ?= --help

help:
	@echo "make sync        install dev dependencies into .venv"
	@echo "make install     install the quill CLI globally with uv tool (QUILL_EXTRAS=cuda for GPU)"
	@echo "make run         run quill from source (ARGS=\"...\")"
	@echo "make lint        ruff lint"
	@echo "make fmt         ruff format (writes)"
	@echo "make fmt-check   ruff format --check"
	@echo "make test        pytest"
	@echo "make lock-check  fail if uv.lock is out of date with pyproject.toml"
	@echo "make check       everything CI runs: lock-check, lint, fmt-check, test"

# ── Setup ──────────────────────────────────────────────────────────────────────

sync:
	uv sync --dev

install:
	@if [ -n "$(QUILL_EXTRAS)" ]; then spec='.[$(QUILL_EXTRAS)]'; else spec='.'; fi; \
	echo "[quill] installing $$spec with uv tool ..."; \
	uv tool install --force --reinstall-package quill "$$spec"
	@command -v ffmpeg >/dev/null 2>&1 || \
		echo "[quill] warning: ffmpeg not found on PATH — see README.md#prerequisites"

run:
	uv run quill $(ARGS)

# ── Checks (mirror CI) ─────────────────────────────────────────────────────────

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

fmt-check:
	uv run ruff format --check .

test:
	uv run pytest

lock-check:
	uv lock --check

check: lock-check lint fmt-check test

clean:
	rm -rf .pytest_cache .ruff_cache dist build
