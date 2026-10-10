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

## Usage (planned)

```bash
disc2plex scan ./sample_rip
disc2plex plan ./sample_rip --show "Show Name" --season 1 --disc-episodes 1-4
disc2plex apply ./sample_rip --show "Show Name" --season 1 --apply
disc2plex undo manifests/<run>.json
disc2plex tmdb search "Show Name"
disc2plex tmdb show 1437                 # seasons and episode groups (DVD order IDs)
disc2plex tmdb season 1437 1 --order dvd
```

## Development

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy
```
