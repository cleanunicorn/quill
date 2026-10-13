# Quill — Agent guide

Guidance for AI agents (and humans) contributing to **Quill**.

Quill is a small Click CLI that transcribes local audio files, YouTube videos,
and direct audio URLs to text with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). It is used from a
terminal on macOS, Linux, and Windows, CPU-only by default with optional CUDA.
A change is done when `make check` passes and the PR is open against `master`
with a Conventional Commits title.

This project follows GitHub flow: `master` is always releasable, all work
happens on short-lived branches, and every change lands through a pull request.
Merging to `master` can cut a release — the PR title decides whether and how
big (see [Release automation](#release-automation)).

Use [README.md](README.md) for setup and user-facing behavior, and
[CONTRIBUTING.md](CONTRIBUTING.md) for the human onboarding path. This file says
how work lands here and what done looks like.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — manages Python (pinned to 3.12 by
  `.python-version`; the package supports ≥ 3.10) and dependencies. Never use
  `pip` or a hand-made venv.
- [ffmpeg](https://ffmpeg.org/) on `PATH` — only for real transcriptions; the
  test suite does not need it.
- GNU `make` — the task runner.
- The `gh` CLI for PRs.
- No credentials. The first real run downloads a Whisper model from Hugging
  Face into the user cache.

## Commands

- **Install dev deps:** `make sync`
- **Install the CLI globally:** `make install` (`QUILL_EXTRAS=cuda` for GPU)
- **Run from source:** `make run ARGS="audio.mp3 out.txt --model tiny"`
- **Lint:** `make lint`
- **Format:** `make fmt` (writes) / `make fmt-check`
- **Test (all):** `make test`
- **Test (focused):** `uv run pytest tests/test_commands.py -k timestamps`
- **Everything CI runs:** `make check` (lockfile, lint, format, tests)

Lint, format, and test touch nothing shared: the tests stub `WhisperModel`,
`yt_dlp`, and downloads, and write only to `tmp_path`. Run them as often as
needed and fix what your change broke without asking. `make run` with a real
input downloads a model (hundreds of MB to ~3 GB) and, for URLs, hits the
network — use `--model tiny` and a local file when you need a real smoke run.

## Golden rules

1. **Never commit directly to `master`.** Always branch, always PR — merging is
   what cuts a release.
2. **Never force-push a shared branch.**
3. **Keep `master` green.** `make check` mirrors CI.
4. **Use the Makefile.** It is the single source of the dev flow. If the flow
   needs to change, change the Makefile, and keep the Windows job in
   `.github/workflows/ci.yml` (which cannot run `make`) in step with it.
5. **Never disable, skip, or delete a test to make a build pass.** If a test is
   wrong, say so and propose the fix.
6. **The PR title is load-bearing.** It drives the release bump, so it must be a
   valid Conventional Commits string (see [PR titles](#pr-titles)).
7. **Default installs stay CPU-only and cross-platform.** GPU/CUDA dependencies
   live only in the `cuda` extra behind
   `sys_platform == 'linux' or sys_platform == 'win32'`; an unconditional one
   breaks `uv sync` on macOS, where the wheels don't exist. Any new dependency
   with platform-specific wheels gets a marker, and `uv.lock` is regenerated
   with `uv lock` in the same PR.
8. **Never lose a transcript.** Output is written to a sibling
   `<name>.<pid>.part` file and renamed into place only on success; a failure
   must never truncate or clobber an existing output file, and Ctrl-C keeps the
   partial and exits 130. Tests in `tests/test_commands.py` pin this — keep
   them passing rather than working around them.
9. **Tests never touch the network or a real model.** Stub `WhisperModel`,
   `yt_dlp.YoutubeDL`, and `download_file` with `monkeypatch`, as the existing
   tests do.

## What done looks like

A change is done when its code, tests, and docs are in one PR, `make check`
passes locally, the PR is open against `master`, and CI is green. User-visible
behavior changes (options, defaults, output format, messages) update the
README's Usage/Options sections in the same PR. If the task includes running
the result, do it — a `--model tiny` run on a short local file is cheap — rather
than returning for review. A read-only request (explain, review, diagnose) is
done when the report is delivered.

## Decision boundaries

- **Safe without asking:** `make check` and its parts, `uv lock` after a
  `pyproject.toml` change, a `--model tiny` smoke run on a local file, removing
  a merged task worktree with no uncommitted work.
- **Needs confirmation unless already authorized, and why:** adding a runtime
  dependency (every user installs it, on three OSes); changing a CLI option's
  name or default (users script against them); raising `requires-python`
  (drops supported interpreters); editing `release.yml` or the
  `[tool.semantic_release]` config (a mistake publishes a wrong version). A
  request that already covers it is the confirmation. In an unattended run,
  leave that part unchanged and list it in the PR with the reason.
- **Never:** see [Golden rules](#golden-rules).

## Communication

- Explain the reasoning behind decisions and approaches.
- When claiming something works or is fixed, prove it — a passing test, a
  command and its output, or a clear explanation of why.
- When uncertain, say so rather than presenting a guess as fact.
- End each response with a confidence indicator: 🟢 High | 🟡 Medium | 🔴 Low

## The GitHub flow, step by step

### 1. Start from an up-to-date `master`

```bash
git checkout master
git pull origin master
```

### 2. Create a branch — in a worktree

**Never edit a checkout of `master` directly.** Create the worktree before the
first edit, do the whole change there, and open the PR from it:

```bash
git worktree add .claude/worktrees/<short-topic> -b <type>/<short-topic>
```

`.claude/worktrees/` is gitignored. Branch names are short, lowercase,
hyphenated, and prefixed with the Conventional Commits type you expect the PR
to use: `feat/`, `fix/`, `refactor/`, `perf/`, `docs/`, `chore/`. Examples:
`fix/youtube-shorts-title`, `feat/srt-output`.

### 3. Make focused changes

- One logical change per PR — and a whole feature *is* one logical change. Ship
  its code, tests, and docs together; don't split it across dependent PRs.
  Don't bundle an unrelated refactor into a fix either.
- Match the surrounding style: `from __future__ import annotations`, `pathlib`
  over `os.path` (ruff's `PTH` rules enforce it), early returns, user-facing
  errors raised as `click.ClickException`, pure helpers in `app/cli/utils.py`
  and the command pipeline in `app/cli/commands.py`.
- **Fix it everywhere.** When you fix a problem, search for the same *shape*
  elsewhere and fix every instance in the same PR, with the count up front.

### 4. Commit

Commits follow [Conventional Commits](https://www.conventionalcommits.org):

```
type(optional-scope): short imperative description
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
`build`, `ci`, `chore`. Add `!` before the colon for a breaking change. Scopes
in use: `cli`, `readme`.

```
fix(cli): add a socket timeout so stalled downloads fail instead of hanging
feat(cli): add --format srt
feat(cli)!: write timestamps by default
```

Imperative mood, subject under ~72 characters, the *why* in the body when it
isn't obvious.

### 5. Run the checks locally

```bash
make fmt     # if you touched Python
make check   # lockfile, lint, format check, tests — same as CI
```

CI also runs the tests on macOS and Windows. If you touched path handling,
filenames, or dependencies, reason about all three OSes; you can't run them
locally, so say in the PR what you checked and what CI will.

### 6. Push and open a PR

```bash
git push -u origin <type>/<short-topic>
gh pr create --base master --fill
```

## PR titles

The PR title follows the same Conventional Commits format:

```
type(optional scope)!: description
```

`.github/workflows/pr-title.yml` blocks the merge on an invalid title: the type
must be one of `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`,
`refactor`, `style`, `test`; the subject must start with a letter and not end
with a period. Editing the title re-validates it.

## PR description

- **What** changed and **why**.
- **Numbers, not adjectives.** `make test: 29 pass`, `-40 lines`, the measured
  timing for a perf claim. "Not measured" beats a vague adjective.
- **The gap**, for a bug fix: what should have caught it, why it didn't, and
  what now does.
- **The sweep**, for any fix: the search you ran for other instances, and its
  count — found, fixed, left (with why).
- `Closes #123` when it resolves an issue; a `BREAKING CHANGE: …` footer for a
  breaking change.

## After opening the PR

- Make sure **CI** (`make check` on Ubuntu and macOS, tests on Windows) and
  **PR Title** are green.
- Address review feedback with more commits on the same branch.
- Mention `@claude` in a PR or issue comment to have Claude Code act on it
  (`.github/workflows/claude.yml`, needs the `CLAUDE_CODE_OAUTH_TOKEN` secret).
- Prefer **squash-merge**, and check the prefilled commit message is the PR
  title alone — that commit is what the release automation parses.

## Release automation

`.github/workflows/release.yml` runs
[python-semantic-release](https://python-semantic-release.readthedocs.io/) v9
on every push to `master`. When a release is due it bumps `project.version` in
`pyproject.toml`, inserts the release notes under the marker in `CHANGELOG.md`,
commits `chore(release): vX.Y.Z` with `[skip ci]`, tags `v{version}`, and
publishes a GitHub Release. There is no PyPI publish — Quill is installed from
this repository with `uv sync` or `uv tool install .`.

| PR title prefix | Release effect |
| --- | --- |
| any `type!:`, or a `BREAKING CHANGE:` footer | **major** |
| `feat:` | **minor** |
| `fix:`, `perf:` | **patch** |
| `docs:`, `refactor:`, `style:`, `test:`, `build:`, `ci:`, `chore:` | **none** |

> ⚠️ Choose the prefix deliberately — it decides whether (and how big) a
> release ships when the PR merges.

## Security

- Never commit secrets, tokens, or user audio/transcripts.
- Treat every input source as untrusted. URLs go through `is_url` /
  `is_youtube_url`, derived output names through `sanitize_filename`, and
  explicit output paths through `validate_output_path` — keep new input paths
  on those helpers rather than building filenames from raw titles or URLs.
- Direct-URL downloads (`download_file`) keep a 30 s socket timeout; without it
  a stalled server hung the CLI forever. Any new network call needs one too.

## Hazards

### Never derive an output filename from a remote title or URL without sanitizing it

**Run every remote-derived name through `sanitize_filename` and fall back to
`transcript` when it comes out empty.**

YouTube titles and URL paths can contain `/`, `..`, or nothing alphanumeric at
all. Used raw, they write outside the working directory or produce an empty
name that `Path` turns into the directory itself. Both fallbacks are pinned in
`tests/test_commands.py` (`*_empty_title_falls_back`,
`*_without_path_falls_back`).

### Never write the transcript straight to the output path

**Write to the `.part` sibling and `replace()` it into place.** Opening the
output with `"w"` truncates it before the first segment arrives, so a model
load failure or Ctrl-C an hour in destroys a transcript the user already had.
This is pinned by `test_transcribe_failure_keeps_previous_output_and_cleans_partial`
and `test_transcribe_ctrl_c_exits_130_and_keeps_partial`.

### Validate an explicit output path before downloading

**Call `validate_output_path` before any download when the user passed
`OUTPUT_FILE`.** Otherwise a typo'd directory fails only after a long YouTube
download. Derived names can only be validated after the download, which is
expected.

- `faster_whisper` is imported at module load in `commands.py`; tests replace
  `commands.WhisperModel`, not `faster_whisper.WhisperModel`. Patch the name
  where it is used.
- `.part` files carry the PID so concurrent runs to the same output don't share
  a partial — don't simplify that away.

## Where to look

- `app/cli/commands.py` — the Click command, input resolution, and the
  transcribe/write pipeline.
- `app/cli/utils.py` — URL detection, downloads (`yt_dlp`, `urllib`), filename
  and timestamp helpers.
- `tests/` — how to stub the model and downloads.
- `pyproject.toml` — dependencies, the `cuda` extra and its platform marker,
  ruff/pytest config, release config.
- `.github/workflows/` — CI (`ci.yml`), title check, release, `@claude`.
