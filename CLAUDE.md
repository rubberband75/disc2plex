# CLAUDE.md

Project guide for Claude Code. Read this fully before making changes. If something here conflicts with what the user says in chat, the user wins, but flag the conflict.

---

## 1. Project Overview

**Name (working):** `disc2plex`

**Purpose:** A command-line tool that takes the raw output of a MakeMKV rip of a TV-series DVD, figures out which episode each video file is, and renames and organizes the files to match TMDB metadata in the layout Plex expects.

**Why it exists:** Manually naming and sorting ripped TV episodes is tedious. Existing tools (FileBot, tinyMediaManager) match mostly by file order and runtime. They struggle with shuffled titles, "Play All" tracks, extras, double-length episodes, and DVD orders that differ from aired order. This tool uses runtime heuristics first, then content verification (speech transcription + LLM or embedding matching) for the hard cases.

**Owner context:** Personal hobby project for a developer digitizing a DVD collection for a personal Plex server. Not a commercial product. Prioritize correctness, safety of the user's files, and being easy to debug over polish.

---

## 2. Goals and Non-Goals

### Goals
- Input: a folder of `.mkv` files from one MakeMKV rip (one disc), plus a show name (and optionally year, season, disc number, TMDB ID).
- Output: files renamed and moved into a Plex-compliant structure, with a human-readable report of what was done and how confident each match was.
- Safe by default: **dry-run is the default mode**; nothing is touched until the user passes `--apply`.
- Handle messy discs: junk titles, Play-All tracks, extras, multi-episode files, out-of-order titles.
- Be resumable and undoable: write a manifest of every rename so it can be reversed.
- Work offline for the AI parts if the user wants (local Whisper + local LLM), with an option to use the Claude API.

### Non-Goals (for now)
- Movies (TV series only in v1; structure code so movies can be added later).
- Re-encoding or transcoding video. Only rename/move files; never modify file contents.
- Ripping discs (MakeMKV does that; this tool starts after the rip).
- A GUI. CLI only in v1.
- Automatically editing Plex's database or triggering library scans (optional later).

---

## 3. Plex Naming Rules (the target format)

Plex's TV layout:

```
/library/TV Shows/
  Show Name (2005) {tmdb-12345}/
    Season 01/
      Show Name (2005) - s01e01 - Episode Title.mkv
      Show Name (2005) - s01e02 - Episode Title.mkv
    Season 02/
      ...
    Season 00/                      # Specials
      Show Name (2005) - s00e01 - Special Title.mkv
```

Rules:
- Show folder: `Show Name (Year)`, optionally with `{tmdb-ID}` to force the correct match. Make the `{tmdb-ID}` tag **on by default** since it prevents misidentification.
- Season folders: `Season 01`, `Season 02` (zero-padded to 2 digits). Specials go in `Season 00`.
- Episode files: `Show Name (Year) - sSSeEE.ext`. Appending ` - Episode Title` is optional and configurable (default: include it).
- Multi-episode files (one file contains two episodes): `Show Name (Year) - s01e01-e02.mkv`.
- Multiple versions of the same episode: use ` - Version Name` suffix handling only if the user asks. Out of scope for v1.
- Sanitize filenames: strip or replace characters illegal on Windows/macOS/Linux (`<>:"/\|?*`), collapse whitespace, trim trailing dots/spaces. Keep it safe for the SMB/NFS share the Plex server may use.
- Extras (behind the scenes, deleted scenes, featurettes): configurable destination. Options: (a) `Season 00` as specials, (b) a show-level extras folder using Plex's extras folder names, (c) a `_extras_review` folder for manual handling. **Default to (c)** until the user chooses.

Verify Plex conventions against current Plex docs if unsure; they change occasionally.

---

## 4. Tech Stack and Conventions

- **Language:** Python 3.11+ (use type hints everywhere).
- **Package management:** `uv` or `pip` + `venv`. Keep dependencies in `pyproject.toml`.
- **CLI framework:** `typer` (or `argparse` if we want zero deps).
- **HTTP:** `httpx`. **Data models:** `pydantic` (v2) or `dataclasses`.
- **Media probing:** `ffprobe` (from FFmpeg) invoked via `subprocess` with `-print_format json`. FFmpeg must be installed on the system; check for it at startup and give a clear error if missing.
- **Speech-to-text:** `faster-whisper` (local). Make model size configurable (`tiny`/`base`/`small`/`medium`). GPU optional.
- **LLM backend (pluggable):**
  - `anthropic` SDK for Claude API (reads `ANTHROPIC_API_KEY` from environment).
  - OpenAI-compatible local endpoint (Ollama, llama.cpp server, LM Studio) at a configurable base URL.
  - Define a small `LLMBackend` interface so the matcher doesn't care which is used.
- **TMDB:** v3 API. Auth with the read-access bearer token from env var `TMDB_API_TOKEN`. Never hard-code or commit keys.
- **Testing:** `pytest`. **Lint/format:** `ruff` (lint + format). **Types:** `mypy` or `pyright` in basic mode.
- **Style:** small pure functions, explicit data models, no global state, logging via `logging` (not print) except for CLI report output. Docstrings on public functions. Prefer clarity over cleverness.

---

## 5. Proposed Project Structure

```
disc2plex/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .env.example              # TMDB_API_TOKEN=, ANTHROPIC_API_KEY=
├── .gitignore                # .env, .cache/, *.mkv, manifests/, __pycache__
├── src/disc2plex/
│   ├── __init__.py
│   ├── cli.py                # Typer entry point
│   ├── config.py             # Settings, env loading, defaults
│   ├── models.py             # Pydantic models (see section 6)
│   ├── scan.py               # ffprobe wrapper -> MediaFile objects
│   ├── classify.py           # Junk/extras/play-all detection, duration clustering
│   ├── tmdb.py               # TMDB client + on-disk cache
│   ├── match/
│   │   ├── __init__.py
│   │   ├── heuristic.py      # Order + runtime matching (step 1 of matching)
│   │   ├── content.py        # Whisper transcript -> episode matching
│   │   └── combine.py        # Merge signals into final confidence + assignment
│   ├── transcribe.py         # faster-whisper wrapper, audio extraction via ffmpeg
│   ├── llm/
│   │   ├── base.py           # LLMBackend protocol
│   │   ├── claude.py
│   │   └── openai_compat.py
│   ├── plan.py               # Build the rename/move plan (pure, no side effects)
│   ├── apply.py              # Execute plan, write manifest, support undo
│   ├── naming.py             # Plex filename/folder generation + sanitization
│   └── report.py             # Human-readable + JSON report output
├── tests/
│   ├── fixtures/             # Tiny fake ffprobe JSON, TMDB JSON samples (no real video)
│   └── test_*.py
└── manifests/                # Runtime output (gitignored): one JSON per apply run
```

Keep side effects (file moves, network, subprocess) isolated in `scan.py`, `tmdb.py`, `transcribe.py`, `llm/`, and `apply.py`. Everything else should be pure and unit-testable.

---

## 6. Core Data Models

Define in `models.py`:

- `MediaFile`: path, size_bytes, duration_seconds, audio_tracks (language, codec, channels), subtitle_tracks, chapter_count, makemkv_title_index (parsed from filename like `title_t03.mkv` / `*_t03.mkv` when possible).
- `FileClass` (enum): `EPISODE`, `PLAY_ALL`, `EXTRA`, `JUNK`, `UNKNOWN`.
- `EpisodeInfo` (from TMDB): season, episode, title, overview, runtime_minutes, air_date.
- `ShowInfo`: tmdb_id, name, year, seasons, episode lists, available episode groups.
- `MatchCandidate`: file, episode(s), score, signals (dict: which heuristics contributed and by how much), reasons (list of human-readable strings).
- `Assignment`: file, class, chosen episodes (1 or more), confidence (0.0-1.0), status (`CONFIDENT`, `NEEDS_REVIEW`, `UNMATCHED`), evidence.
- `PlanItem`: source path, destination path, action (`move`/`hardlink`/`copy`/`skip`), reason.
- `Manifest`: run id, timestamp, args, list of executed `PlanItem`s (enough to undo).

---

## 7. Pipeline (how it works)

Implement in stages, each producing inspectable output. Every stage should be runnable and testable on its own.

### Stage 1: Scan
- Run `ffprobe` on each `.mkv` in the input folder (non-recursive by default; `--recursive` flag).
- Extract duration, audio/subtitle tracks, chapter count.
- Cache results by (path, size, mtime) so reruns are fast.

### Stage 2: Classify
- Group files by duration. TV episodes on a disc typically cluster tightly (e.g., 41-45 min).
- Detect **Play All** tracks: a file whose duration is approximately the sum of a subset of other files (tolerance ~2-3%) or much longer than the cluster median with chapter marks lining up at episode boundaries.
- Detect **extras/junk**: very short files (< ~5 min unless the show has short episodes, so compare against TMDB runtimes), menu loops, duplicates (same size/duration), tracks with no audio.
- Be conservative: when unsure, class = `UNKNOWN` and let later stages or the user decide.
- Handle **double-length episodes** (roughly 2x median runtime), which may be two-part episodes or two-episode files. Confirm against TMDB data.

### Stage 3: Fetch TMDB data
- Resolve the show: use `--tmdb-id` if provided, else search by name (+ `--year`), and if ambiguous show the top candidates and require the user to choose (interactive prompt, or fail with a clear message in non-interactive mode).
- Fetch show details and season episode lists. Fetch **episode groups** (`/tv/{id}/episode_groups`) so DVD order can be used when it differs from aired order. Expose `--order aired|dvd|<group-id>`.
- Cache all TMDB responses on disk (`.cache/tmdb/`) with a TTL (e.g., 7 days).

### Stage 4: Heuristic matching
- Inputs: classified episode files, the season's episode list, optional `--disc-episodes 1-4` hint.
- Primary signal: MakeMKV title order (`_t00`, `_t01`, ...) mapped sequentially onto episodes, validated by runtime similarity to TMDB runtimes.
- Output a score per (file, episode) pair and reasons. Many discs resolve fully here.
- If the user didn't give a disc/episode range, infer the most plausible contiguous range by runtime fit.

### Stage 5: Content verification (the AI step)
Only run for files that are low-confidence, ambiguous, or when `--verify all` is passed.
1. Extract a slice of audio (e.g., 3-5 minutes from the middle, plus optionally the opening) with `ffmpeg`.
2. Transcribe with `faster-whisper` (pick the audio track matching the user's language preference).
3. Compare the transcript to candidate episodes' TMDB overviews:
   - Option A (cheap): embedding similarity between transcript chunks and episode overviews.
   - Option B (better): ask the LLM: "Here are N candidate episode synopses and a transcript excerpt. Which episode is this? Return JSON: `{episode, confidence, reasoning}`." Always request structured JSON and validate it.
4. Treat LLM output as one signal, not truth. Combine with heuristic score.
- Optionally use subtitle tracks (if text-based or OCR'd) as another signal; DVD subtitles are usually bitmap, so Whisper is the default.
- Respect rate limits and cost: batch candidates, cap API calls per run, log token usage.

### Stage 6: Combine and decide
- Merge signals into a final confidence per file. Thresholds are configurable (defaults: >= 0.85 confident, 0.5-0.85 needs review, < 0.5 unmatched).
- Detect conflicts: two files assigned to the same episode, or gaps in the episode sequence. Flag them, never silently pick one.
- Anything not confident goes to a review bucket with its evidence shown.

### Stage 7: Plan
- Build the list of `PlanItem`s as a pure function of assignments + config. No filesystem changes.
- Print the plan as a table: source -> destination, confidence, notes.
- Refuse to plan if two files would map to the same destination path.

### Stage 8: Apply (opt-in)
- Only runs with `--apply`. Re-validate that sources still exist and destinations don't.
- **Never overwrite** an existing file. On collision, skip and report.
- Default action is `move` within the same filesystem. Offer `--hardlink` and `--copy`.
- Write a `manifest` JSON to `manifests/` before and during the run so `disc2plex undo <manifest>` can reverse it.
- Low-confidence and unmatched files go to `<output>/_review/<disc-name>/` unchanged (or are left in place; make this a flag).

---

## 8. CLI Design

```
disc2plex scan   <rip_dir>                       # show probed files + classification
disc2plex plan   <rip_dir> --show "Name" [opts]  # full pipeline, dry-run, print plan
disc2plex apply  <rip_dir> --show "Name" [opts]  # same as plan, then execute
disc2plex undo   <manifest.json>                 # reverse a previous apply
disc2plex tmdb   search "Show Name"              # helper to find TMDB IDs
```

Common options:
- `--show TEXT`, `--year INT`, `--tmdb-id INT`
- `--season INT` (required unless inferable), `--disc INT`, `--disc-episodes 1-4`
- `--order aired|dvd|<episode-group-id>`
- `--output PATH` (Plex library root; also settable via config)
- `--verify none|low|all` (default `low`), `--llm claude|local|none`, `--whisper-model small`
- `--lang en` (audio language preference)
- `--action move|hardlink|copy`, `--include-titles/--no-titles`, `--extras review|season0|skip`
- `--apply` (only on `apply`; `plan` is always dry-run), `--yes` (skip prompts), `--json` (machine-readable output), `-v/-vv`

Config file: `~/.config/disc2plex/config.toml` (library root, default LLM backend, thresholds). CLI flags override config; config overrides defaults.

---

## 9. Safety Rules (important)

These apply to you (Claude Code) while developing AND to the tool's runtime behavior.

**Runtime behavior of the tool:**
1. Dry-run by default. Destructive actions require explicit `--apply`.
2. Never delete user files. Never overwrite existing files.
3. Never modify the contents of video files.
4. Always write an undo manifest.
5. Validate every path stays inside the intended output root (guard against path traversal from weird titles).
6. Log decisions with reasons so a wrong match can be diagnosed afterward.

**While developing:**
- **Do not run `--apply` against real media.** Use the test fixtures or a throwaway temp directory with tiny dummy files.
- Don't commit video files, API keys, or `.env`.
- Don't run large downloads or install system packages without asking.
- Ask before adding a new dependency or making a large architectural change.
- Never send real user transcripts or media to an external API in tests. Mock the LLM and TMDB in unit tests.

---

## 10. Testing Strategy

- **Unit tests** (no network, no real media): classification logic, filename sanitization, naming, plan generation, collision detection, confidence combination, manifest/undo.
- **Fixtures:** saved `ffprobe` JSON from sample discs (hand-written or anonymized), saved TMDB responses. Build a helper to generate a fake rip folder of zero-byte `.mkv` files plus matching probe data.
- **Edge-case tests to include:**
  - Play-All track present
  - Titles out of order
  - Double-length episode / two-part episode
  - Missing episode (gap in sequence)
  - Two files with near-identical duration
  - Extras and very short clips
  - DVD order differing from aired order
  - Show names with illegal filename characters (e.g., `Star Trek: The Next Generation`)
  - Specials in Season 00
  - Re-running `apply` twice (idempotency, no overwrites)
- **Content-matching tests:** mock Whisper and LLM outputs; separately keep a small manual eval script (not part of CI) that runs real transcription on a short sample clip.
- Run: `pytest -q`, `ruff check .`, `ruff format --check .`, `mypy src` (or `pyright`).

---

## 11. Common Commands

Uses `uv`. Dev tools live in the `dev` dependency group, which `uv sync` installs by default.

```bash
# setup
uv sync                  # creates .venv, installs runtime + dev deps
cp .env.example .env     # then fill in keys

# dev loop
uv run pytest -q
uv run ruff check . && uv run ruff format .
uv run mypy              # strict mode, configured for src/ in pyproject.toml

# try it (safe)
uv run disc2plex scan ./sample_rip
uv run disc2plex plan ./sample_rip --show "Show Name" --season 1 --disc-episodes 1-4
```

---

## 12. Milestones (build in this order)

Work one milestone at a time. After each, run tests, summarize what changed, and wait for the user before moving on.

1. **Scaffold:** `pyproject.toml`, package layout, `.gitignore`, `.env.example`, README stub, CLI skeleton with `--help`, pytest + ruff configured.
2. **Naming module:** Plex path generation + sanitization with thorough tests.
3. **Scan:** ffprobe wrapper, `MediaFile` models, `disc2plex scan` command, fixture-based tests.
4. **Classify:** duration clustering, Play-All and extras detection, tests for the edge cases above.
5. **TMDB client:** search, show/season fetch, episode groups, disk cache; mocked tests.
6. **Heuristic matching + plan output:** end-to-end dry-run `plan` command working with no AI.
7. **Apply + manifest + undo:** safe execution, collision handling, idempotency tests.
8. **Whisper transcription:** audio slice extraction + faster-whisper wrapper.
9. **Content matching:** LLM backend interface, Claude + local implementations, structured JSON output with validation, combine signals into final confidence.
10. **Polish:** config file, better reports (`--json`), review-folder flow, README with examples, optional Plex library-scan trigger.

**Milestone 6 is the first genuinely useful version.** Prioritize getting there.

---

## 13. LLM Prompt Notes (for Stage 5)

- Keep prompts short and structured. Provide: show name, season, candidate episodes (number, title, overview), and the transcript excerpt.
- Require JSON output: `{"episode": int | null, "confidence": 0-1, "reasoning": "short"}`. Allow `null` when no candidate fits.
- Tell the model to answer from the evidence only and to say "unsure" rather than guess.
- Validate and clamp the response in code; retry once on malformed JSON, then fall back to "no signal."
- Transcripts from Whisper can contain errors and names get mangled; give the model the character/place names from the TMDB overview as context.
- Log prompts and responses (redacting keys) at debug level to help tune.

---

## 14. Open Questions (ask the user before assuming)

When starting, ask the user about these and record answers in this file:

1. What OS does the dev machine and the Plex server use? (path handling, hardlink support)
2. Where is the Plex library root, and is it local or a network share?
3. Typical rip layout: one folder per disc? Are MakeMKV default filenames kept (`title_t00.mkv`)?
4. Preferred language for audio/subtitles?
5. Include episode titles in filenames? Include `{tmdb-ID}` in folder names?
6. Where should extras go by default?
7. Which LLM backend first: Claude API or a local model? What GPU/RAM does the local machine have (for Whisper and local LLM sizing)?
8. A few example shows to test against, especially any with weird DVD ordering.

---

## 15. Working Agreements for Claude Code

- Start by reading this file, then propose a short plan for the current milestone before writing code.
- Make small, reviewable changes. Run tests and linters after each change and fix failures before moving on.
- Prefer simple, readable code. Don't over-engineer; no plugin systems or abstractions beyond what a milestone needs.
- Explain non-obvious design decisions in brief comments or docstrings.
- If requirements are unclear or a decision would be hard to reverse, ask instead of guessing.
- Keep this file up to date: when commands, structure, or decisions change, update the relevant section.
- Never run commands that touch real media directories unless the user explicitly points you at a specific path and asks.

---

## 16. Decisions Log

(Append decisions here as they're made, with dates.)

- 2026-10-10: Project started. Tool is a deterministic pipeline with LLM calls at specific steps, not a free-roaming agent.
- 2026-10-10: Dev machine OS is Linux Mint. Package manager is `uv` (hatchling build backend, `dev` dependency group). CLI uses `typer`. mypy runs in strict mode. Dependencies are added in the milestone that first needs them (`httpx`/`pydantic` not yet added).
- 2026-10-10: Milestone 1 (scaffold) done. Unimplemented commands exit 1 with a "not implemented (milestone N)" message. The `ffprobe` check runs only in commands that need it (scan/plan/apply), so `--help`, `undo`, and `tmdb` still work without FFmpeg.
- 2026-10-10: Defaults confirmed: `{tmdb-ID}` in show folders ON, episode titles in filenames ON.
- 2026-10-10: Milestone 2 (naming) done. Sanitization rules: `": "` -> `" - "`, bare `:` `/` `\` `|` -> `-`, `"` -> `'`, `<>?*` removed, control chars removed, NFC-normalized, Windows reserved stems get `_` suffix (`CON.mkv` -> `CON_.mkv`), 255-byte UTF-8 limit per component (only the episode title is truncated). Multi-episode titles are joined with `" & "` (duplicates collapsed); ranges must be contiguous. `ensure_within` is a lexical check; `apply` must re-check with resolved paths.
- 2026-10-10: One git commit per milestone (user preference).
- 2026-10-10: Milestone 3 (scan) done. Added `pydantic` and `rich` (rich was already a typer dep; now declared since report.py imports it). Models are frozen pydantic models. `MediaFile` stores `chapter_starts` (seconds) rather than just a count, for Play-All detection; `chapter_count` is a property. Probe cache lives at `$XDG_CACHE_HOME/disc2plex/probe.json` (default `~/.cache/...`), keyed by resolved path + size + mtime_ns, stores raw ffprobe JSON so parser changes don't need cache invalidation; `scan --no-cache` bypasses it. Probe failures are collected in `ScanResult.failures` and shown to the user, never raised. `scan_directory` takes an injectable `probe` function; tests use the `fake_rip` fixture in `tests/conftest.py` (zero-byte files + fixture/synthetic probe JSON). One test generates a 3-second MKV with real ffmpeg in `tmp_path` (skipped if ffmpeg is missing).
- 2026-10-10: Milestone 4 (classify) done, tested on synthetic data only. Real-rip fixtures still needed (user will copy a rip folder over later). Rules in `classify.py` (thresholds in `ClassifyConfig`):
  - Junk: duration 0, < 60 s, no audio, or a duplicate (duration ±1 s, size ±0.5%, same chapter layout; the first by title order is kept).
  - Play-All: duration within 2.5% of the sum of ≥2 shorter titles (subset-sum), confirmed by total size ±3% OR chapter marks at the boundaries (any ordering for ≤6 components). Chapters present but misaligned → not Play-All. No chapters and no size match → "suspected" → UNKNOWN.
  - Reference length: the largest cluster (±15%) of titles ≥10 min, or the cluster nearest the expected runtime when one is given (the observed median beats rounded TMDB runtimes); falls back to the expected runtime.
  - Bands by ratio to the reference: 0.8–1.2 episode, 1.6–2.4 episode with `episode_count_hint=2`, < 0.5 extra, everything else UNKNOWN. Without any reference: < 5 min extra, else UNKNOWN.
  - `scan` shows Class/Why columns; `scan --episode-length MIN` supplies the expected runtime until TMDB is wired in.


