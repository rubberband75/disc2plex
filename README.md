# disc2plex

Turn a folder of MakeMKV rips from a TV-series DVD into correctly named,
Plex-ready episode files using TMDB metadata.

> Work in progress. Nothing is moved unless you run `apply` (planned for a later milestone);
> `plan` is always a dry run.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- FFmpeg (`ffprobe` must be on `PATH`)

## Setup

```bash
uv sync                  # creates .venv and installs runtime + dev dependencies
cp .env.example .env     # then fill in TMDB_API_TOKEN (the "API Read Access Token")
uv run disc2plex --help
```

## Usage

All commands work. Matching uses title order and runtimes, and can be verified
from what is actually said in each episode: local Whisper transcribes a few
minutes of audio and an LLM (Claude, or a local OpenAI-compatible server such as
Ollama) picks the matching TMDB synopsis. The first transcription downloads the
Whisper model (~480 MB for `small`).

```bash
# Verify uncertain matches with Claude (needs ANTHROPIC_API_KEY in .env)
disc2plex plan ./GOOD_PLACE_S1_D1 --show "The Good Place" --season 1 \
    --output "/srv/plex/TV Shows" --llm claude
# ...or with a local model
DISC2PLEX_LLM_MODEL=llama3.1:8b disc2plex plan ... --llm local
```

Transcripts are only sent to an API when you pass `--llm claude`. Answers are
cached, so running `plan` and then `apply` costs one set of calls.

```bash
disc2plex scan ./sample_rip
disc2plex plan ./sample_rip --show "Show Name" --season 1 --disc-episodes 1-4 --output "/srv/plex/TV Shows"
disc2plex plan ./FIREFLY_D1 --show Firefly --disc 1 --order dvd --output "/srv/plex/TV Shows"
disc2plex apply ./FIREFLY_D1 --show Firefly --disc 1 --order dvd --output "/srv/plex/TV Shows" --apply
disc2plex undo ~/.local/share/disc2plex/manifests/<run>.json
disc2plex tmdb search "Show Name"
disc2plex tmdb show 1437                 # seasons and episode groups (DVD order IDs)
disc2plex tmdb season 1437 1 --order dvd
disc2plex transcribe ./FIREFLY_D1 -f FIREFLY_D1_t01.mkv   # local Whisper, CPU is fine
```

`plan` is always a dry run. It prints where each file would go, how confident the
match is, and the evidence behind it. Files it isn't confident about are routed to
`<output>/_review/<disc>/` rather than guessed. Pinning the disc position with
`--disc-episodes` (or `--disc 1`) makes matches much more confident.

`apply` never overwrites an existing file, never deletes anything, and records
every operation in a manifest (under `~/.local/share/disc2plex/manifests/`) that
`undo` can reverse. Without `--apply` it only prints the plan.

## Development

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy
```
