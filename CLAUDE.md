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
disc2plex plan   <rip_dir> --show "Name" --output <library> [opts]  # full pipeline, dry-run
disc2plex apply  <rip_dir> --show "Name" --output <library> [--apply] [--yes]  # plan, then execute
disc2plex undo   <manifest.json> [--dry-run] [--yes]  # reverse a previous apply
disc2plex tmdb   search "Show Name"              # helper to find TMDB IDs
disc2plex tmdb   show <tmdb-id>                  # seasons + episode-group IDs for --order
disc2plex tmdb   season <tmdb-id> <n> [--order]  # episode list in the chosen order
disc2plex transcribe <rip_dir> [-f NAME] [--json]  # Whisper transcript of each episode's middle (debug)
# content verification: add  --verify low|all --llm claude|local [--llm-model M]  to plan/apply
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
DISC2PLEX_REAL_WHISPER=1 uv run pytest -q tests/test_transcribe.py -k real   # opt-in real Whisper eval
uv run ruff check . && uv run ruff format .
uv run mypy              # strict mode, configured for src/ in pyproject.toml

# try it (safe)
uv run disc2plex scan ./sample_rip
uv run disc2plex plan ./sample_rip --show "Show Name" --season 1 --disc-episodes 1-4 --output /tmp/fake-library
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
- 2026-10-10: Milestone 5 (TMDB) done, tested with mocks only (no token configured yet, so no live call has been made). Added `httpx`; `.env` is loaded by a tiny parser in `config.py` (no python-dotenv) from `./.env` then `~/.config/disc2plex/.env`, and never overrides real env vars. `TMDB_API_TOKEN` may be a v4 bearer token (preferred) or a 32-hex v3 API key (sent as `api_key`).
  - Cache: `~/.cache/disc2plex/tmdb/` (XDG, same base as the probe cache, instead of the `.cache/tmdb/` in section 7), one file per request keyed by path + params, 7-day TTL, credentials never written. Errors are not cached.
  - 429: honours Retry-After (capped at 10 s), 2 retries.
  - Show resolution: `--tmdb-id` wins; otherwise accept the only search hit or the only exact normalized-name match (year-filtered); otherwise raise `AmbiguousShowError` listing candidates. (The interactive prompt comes with the `plan` command in M6.)
  - **DVD/group order keeps aired numbering:** `--order dvd|<group-id>` decides which episodes and in what sequence (for matching), but each `EpisodeInfo` keeps TMDB's aired season/episode numbers, and files are named with those, since that is what Plex's TMDB agent matches on.
  - Group sub-season selection: a name match ("Season N"/"Series N"/"Volume N"), "Specials" for season 0, else the Nth non-special sub-group by `order`.
  - `ShowInfo` holds season summaries + episode groups; episode lists are fetched per season on demand (not all seasons up front).
  - Extra debug commands: `tmdb show <id>` (seasons + episode-group IDs) and `tmdb season <id> <n> [--order]`.
- 2026-10-10: Milestone 6 (heuristic matching + `plan`) done: the first end-to-end dry run. Tested with synthetic rips, generated MKVs, and fixture-backed TMDB; not yet against a real rip or live TMDB.
  - Pipeline glue lives in `pipeline.py` (`PlanRequest` -> `run_plan` -> `PlanRun`), shared by `plan` and the upcoming `apply`; `cli.py` only parses, prompts, and prints.
  - Matching (`match/heuristic.py`): DP aligns EPISODE files (title order) to a contiguous run of episode positions. Pair score = log Gaussian runtime fit (sigma 15%); a double-length file may take 2 episodes (small single-episode prior); mid-run skips and unmatched files are penalised. `--disc-episodes A-B` (positions in the chosen order) pins the run; `--disc 1` pins the start at position 1; otherwise every start is tried and the score margin over the next-best start sets the order confidence (0.5 to 0.9). Per-file confidence = order confidence x runtime agreement; gaps cap it at 0.8. Out-of-order titles are not reordered (that needs content matching, M9): misfits are left unmatched and go to review.
  - `MatchCandidate` from section 6 is deferred to M9; `Assignment.signals` carries the per-signal numbers for now.
  - Classification now also receives the season's individual TMDB runtimes: a "suspected Play-All" or odd-length title matching a specific TMDB episode runtime (±5%) is treated as an episode (e.g. an 86-min pilot). Play-All size confirmation tightened to size ±1% AND duration ±1%, because same-show episodes have similar bitrates and looser size matches happen by chance.
  - Plan routing: confident episodes -> library; needs-review/unmatched/unknown -> `<output>/_review/<disc>/`; extras -> `<output>/_extras_review/<disc>/` (`--extras skip` leaves them; `season0` falls back to review until M9); Play-All/junk/unprobeable -> left in place. Duplicate destinations are plan errors (exit 1). Existing destinations become `skip`. Multi-episode files whose episodes aren't consecutive in aired numbering (possible with DVD order) go to review.
  - `plan` requires `--output` (until the config file exists in M10) and `--show` or `--tmdb-id`; an ambiguous show triggers a numbered prompt on a TTY, or an error listing candidates with `--yes`/`--json`/non-TTY.
- 2026-10-10: Milestone 7 (apply/manifest/undo) done. Verified with unit tests plus a real `apply --apply` and `undo` round trip on generated MKVs in a scratch directory (checksums identical afterwards). Never run against real media.
  - Review folders stay inside the library root (user said continue; my recommended option). `apply` writes `.plexignore` (`*`, `*/*`, `*/*/*`; Plex only matches slash-less patterns in the same folder) into `_review/` and `_extras_review/` when it puts files there. An existing `.plexignore` is never touched.
  - Manifests live in `$XDG_DATA_HOME/disc2plex/manifests/` (default `~/.local/share/...`), not a cwd-relative `manifests/` (section 7). Each one is written before the first operation and rewritten atomically after every operation; an interrupted run can be undone.
  - No-clobber placement: move = `os.link` + unlink source (atomic EEXIST if the destination is taken). Without hardlink support (EPERM/ENOTSUP...): exists-check + rename. Across filesystems (EXDEV): copy to a `.disc2plex-partial-*` temp beside the destination, verify size, no-clobber place, then unlink the source. If the source can't be removed, the placement is rolled back.
  - Before each operation: the source must still exist (else skipped), the destination (including a dangling symlink) must not exist (else skipped, "not overwritten"), and containment is re-checked with `realpath`, so symlinked folders can't escape. `apply` refuses if the output root doesn't exist (typo or unmounted share) rather than creating it.
  - `apply` without `--apply` is a dry run; with `--apply` it asks for confirmation on a TTY and requires `--yes` otherwise. `undo` has `--dry-run`, the same confirmation rules, reverses newest-first, never overwrites the original location, and removes copies/hardlinks only if the original still exists and the copy's size/inode are unchanged. It then removes our `.plexignore` (if unchanged) and any directories we created that are empty.
- 2026-10-10: First real-data run: The Good Place S1 Blu-ray disc 1 (`inputs/`, gitignored) with live TMDB (id 66573). Findings and fixes:
  - Disc episodes ran ~5-20% longer than TMDB's runtimes, so good matches looked like mismatches. The matcher now calibrates a per-disc `runtime_scale` (median of each single-episode file's duration / its nearest TMDB runtime, clamped 0.8-1.35) and notes it. "Mismatch" now means fit < 0.3 (~17% off after scaling); 0.3-0.6 reads "roughly matches".
  - With `--disc 1` (aired order) t01-t06 -> s01e03-e07 confident; t00 (42:43) -> s01e01-e02 at ~82% (two-episode files drop a set of credits, so they run shorter than 2x). Without a disc hint everything correctly needs review (uniform runtimes).
  - TMDB's "Bluray Release" DVD group merges e01+e02 into its first entry (lists e01, then e03). Not handled specially yet; aired order works better for this show.
  - Trimmed real ffprobe metadata (no paths, no video) is in `tests/fixtures/real/`, trimmed real TMDB responses in `tests/fixtures/tmdb/*66573*`, and `tests/test_real_discs.py` locks in this disc's behavior.
  - Dev hardware (from `.device-info`, gitignored): Ryzen 7 5700X (8C/16T), 64 GB RAM, AMD RX 6700 XT. CTranslate2/faster-whisper has no ROCm support, so Whisper runs on CPU (int8).
- 2026-10-10: Milestone 8 (Whisper transcription) done and verified on the real Blu-ray. Added `faster-whisper` (regular dependency, imported lazily) and `numpy` (declared because `transcribe.py` imports it). `HF_TOKEN` in `.env` is picked up by huggingface_hub for model downloads.
  - Default model `small` (user OK'd; quality is sufficient: names get mangled, plot is clear), CPU int8 (no CUDA on this AMD machine). About 23 s per 4-minute slice on the 5700X, including model load per run.
  - Audio: best track by language (en/eng aliases), not commentary, default, most channels. A 240 s slice centred on the middle (avoids recaps/credits); `--opening` adds a 120 s slice at 1:00. ffmpeg fast-seeks (`-ss` before `-i`) to a 16 kHz mono WAV in a temp dir; source files are only read.
  - The WAV is decoded with stdlib `wave` + numpy and passed to Whisper as samples, bypassing faster-whisper's PyAV decoder (faster-whisper 1.2.1 + PyAV 19 crash with "unexpected keyword argument 'metadata_errors'").
  - Transcripts are cached in `~/.cache/disc2plex/transcripts/`, keyed by resolved path + size + mtime + stream + slice + model.
  - New debugging command `disc2plex transcribe <rip> [-f NAME] [--all] [--whisper-model] [--lang] [--opening] [--json]`; by default it transcribes EPISODE/UNKNOWN titles only.
  - The manual real-Whisper eval from section 10 is an opt-in test: `DISC2PLEX_REAL_WHISPER=1 uv run pytest -q tests/test_transcribe.py -k real` (flite speaks a sentence that Whisper must hear). All other tests fake Whisper but use real ffmpeg on generated clips.
  - Real transcripts of disc 1 are consistent with the `--disc 1` aired-order plan (t02 = "Jason Mendoza", etc.).
- 2026-10-10: Milestone 9 (content matching) done and verified live on the real Blu-ray with Claude. Added `anthropic` (SDK 1.13).
  - LLM backends (`llm/`): `LLMBackend` protocol with `complete_json(system, user, schema)`. `ClaudeBackend`: `claude-opus-5-5`, effort `medium` (explicit; Opus 5.5 can't disable thinking), structured outputs via `output_config.format` (JSON schema), server-side refusal fallback (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`), refusals/max_tokens -> `LLMError`. `OpenAICompatBackend`: `/chat/completions` with `response_format: json_schema`, config via `--llm-model` / `DISC2PLEX_LLM_MODEL`, `DISC2PLEX_LLM_BASE_URL` (default Ollama `http://localhost:11434/v1`), `DISC2PLEX_LLM_API_KEY`. No local server was available, so it's tested only against a mocked HTTP server. `CachedLLM` caches replies in `~/.cache/disc2plex/llm/` so plan-then-apply doesn't pay twice.
  - Prompt (section 13): show, season, every episode of the season as `sSSeEE "Title": overview`, then numbered transcript excerpts. Reply schema: `{"excerpts": [{"excerpt", "episode" (enum of listed codes + "none"), "confidence", "reasoning"}]}`. That's per-excerpt rather than the single `{episode, confidence, reasoning}` in section 13, so a two-episode file (one excerpt per half) gets an answer for each half. Validated with pydantic, confidences clamped, unknown codes -> no match, one retry on malformed JSON, then "no signal".
  - Flow (key design change after the first live run): `gather` (transcribe + ask, for target files) -> re-run the heuristic aligner with the answers as votes (`content_score`: per-excerpt log-likelihood ratio vs a 1/N baseline, weight 0.6 because LLM confidence is not calibrated) -> `apply_evidence` (agreement: `1-(1-h)(1-0.9c)`; partial: weight 0.5; disagreement: reassign only if c >= 0.85, runtime fits, consecutive, and no other file has supporting evidence for that episode, always capped at 0.84 = review; otherwise cap at 0.7 = review). Combining after alignment instead (the first attempt) reassigned files one by one onto a wrong start.
  - `--verify low` (default) checks episodes below 0.85 plus UNKNOWN titles of plausible length; `all` checks every episode; cap of 30 LLM calls per run; an `LLMError` stops further calls with a note. `--llm` defaults to `none` (explicit opt-in before transcripts go to an API); with `--verify` != none and `--llm none` a note explains verification is off. Two-episode files get two 180 s slices at 25% and 75%.
  - Live result, The Good Place S1D1 with no disc hint: start inferred at position 1 (margin 4.4), t00/t01/t02/t03/t06 confident and correct (96-98%), t04 -> review because Claude said "Flying" (e02, 85%) for what is really e06 and t00 has supporting evidence for e02. 6 calls, ~17.7k input / 1.5k output tokens.
  - Known limit: synopsis-only matching can be confidently wrong when synopses overlap (t04). Possible follow-up: a second excerpt (`--opening`) when content disagrees.
- 2026-10-10: First real `apply --apply` (user asked), on scratch copies of t03 + t04 only; `inputs/` untouched. Found and fixed a confidently wrong match: with no disc hint, t04's wrong "e02" answer pulled the inferred start to position 1 and then "agreed" with it (91%). Now, when the start is inferred, agreement only boosts past review if agreeing files outnumber disagreeing ones more than 2:1 (full disc 1: 5 vs 1, still confident; this pair: 1 vs 1, both review, with a note to pass `--disc-episodes`). With `--disc-episodes 5-6` both moved to the right names; apply, undo, and the review-folder path all round-tripped with identical checksums.
