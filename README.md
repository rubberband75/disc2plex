# disc2plex

Turn a folder of MakeMKV rips from a TV-series DVD or Blu-ray into correctly
named, Plex-ready episode files using TMDB metadata.

It works out which title is which episode from title order and runtimes, and
can check the uncertain ones against what is actually said in each episode:
local Whisper transcribes a few minutes of audio, and an LLM (Claude, or a local
model) picks the matching TMDB synopsis. Anything it isn't sure about goes to a
review folder instead of being guessed.

**Safe by default:** nothing moves without `apply --apply`. It never overwrites
or deletes a file, never modifies video, and every run can be undone.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- FFmpeg (`ffprobe` and `ffmpeg` on `PATH`; `sudo apt install ffmpeg`)
- A free [TMDB](https://www.themoviedb.org/settings/api) API Read Access Token
- Optional: an Anthropic API key (`--llm claude`) or a local OpenAI-compatible
  server such as Ollama (`--llm local`) for content verification
- Optional: an NVIDIA GPU for faster transcription. It needs the CUDA 12 cuBLAS
  library (`libcublas.so.12`; on Ubuntu/Mint the `libcublas12` package, part of
  `nvidia-cuda-toolkit`) and about 0.8 GB of free GPU memory

## Setup

```bash
uv sync                      # creates .venv and installs everything
cp .env.example .env         # fill in TMDB_API_TOKEN (and ANTHROPIC_API_KEY if wanted)
uv run disc2plex config init # optional: writes ~/.config/disc2plex/config.toml
uv run disc2plex --help
```

Secrets live in the environment or `.env`: `TMDB_API_TOKEN`, `ANTHROPIC_API_KEY`,
`PLEX_TOKEN`, `HF_TOKEN` (optional, for Whisper model downloads), and
`DISC2PLEX_LLM_*` for a local LLM. `.env` is read from the folder you run the
command in, then from `~/.config/disc2plex/.env`; put it in the second place to
run from anywhere.

The examples below write `disc2plex` for short. Run it as `uv run disc2plex ...`
from the repository, or activate the environment first
(`source .venv/bin/activate`). After moving or renaming the repository folder,
run `uv sync --reinstall` once: the environment's launchers store its old path.

## A typical disc

```bash
# 1. Look at what's on the disc (episodes, Play-All, extras, junk)
disc2plex scan "./The Good Place Season One- Disc One"

# 2. Dry run: where would everything go, and how sure is it?
disc2plex plan "./The Good Place Season One- Disc One" --show "The Good Place" \
    --season 1 --disc 1 --output "/srv/plex/TV Shows"

# 3. Same command with `apply --apply` to do it (asks for confirmation)
disc2plex apply "./The Good Place Season One- Disc One" --show "The Good Place" \
    --season 1 --disc 1 --output "/srv/plex/TV Shows" --apply

# Changed your mind? The apply output prints the exact undo command.
disc2plex undo ~/.local/share/disc2plex/manifests/<run>.json
```

Result:

```
/srv/plex/TV Shows/
  The Good Place (2016) {tmdb-66573}/
    Season 01/
      The Good Place (2016) - s01e01-e02 - Everything Is Fine & Flying.mkv
      The Good Place (2016) - s01e03 - Tahani Al-Jamil.mkv
      ...
  _review/            # uncertain files, hidden from Plex by a .plexignore
  _extras_review/     # featurettes etc., for you to file
```

Telling it where the disc sits in the season (`--disc 1`, or `--disc-episodes 5-8`)
makes matches much more confident. Without it, runtimes alone often can't tell
episodes apart, so expect review items unless you turn on content verification.

## Content verification (for discs where runtimes aren't enough)

```bash
# Claude (needs ANTHROPIC_API_KEY); a few cents per checked file, answers cached
disc2plex plan ./DISC --show "The Good Place" --season 1 --llm claude

# A local model via Ollama, LM Studio, or llama.cpp server (nothing leaves the machine)
DISC2PLEX_LLM_MODEL=llama3.1:8b disc2plex plan ./DISC --show ... --llm local

# llama.cpp: start the server, then point disc2plex at it (the default URL is Ollama's)
llama-server -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf -c 16384 --port 8090 -a qwen
DISC2PLEX_LLM_BASE_URL=http://127.0.0.1:8090/v1 disc2plex plan ./DISC --show ... \
    --llm local --llm-model qwen

# Check every episode, not just the uncertain ones
disc2plex plan ./DISC --show ... --llm claude --verify all
```

If a file's answer contradicts its match, or is unclear, it gets a second listen:
two more excerpts are transcribed and the LLM is asked again with all of them. This
usually fixes a misleading first excerpt; files that still conflict go to review.

Local models are asked not to "think" before answering (a small reasoning model
can think for minutes here without reaching an answer); set `thinking = true`
under `[local_llm]` to allow it. Give the server at least an 8k context: a season's
synopses plus the excerpts run to 3-6k tokens. A larger model is worth it: on one
real disc a 35B model matched Claude's answers on the first listen, while a 9B
model got two of seven episodes wrong at "95%" until the second listen fixed them.

Transcripts are only sent to an API when you choose `--llm claude`. Whisper runs
locally; the first run downloads the model (~480 MB for `small`). To see what
Whisper hears: `disc2plex transcribe ./DISC -f DISC_t03.mkv`.

Whisper runs on an NVIDIA GPU when it can and on the CPU otherwise
(`--whisper-device auto|cpu|cuda`, or `whisper_device` under `[matching]`). With
`auto`, a GPU that can't be used (no CUDA library, memory full) costs one warning
and the run continues on the CPU; `cuda` makes that an error. On a GTX 1080 Ti the
`small` model transcribes at about 32x real time against 9x on an 8-core CPU, and
fits beside a local LLM on the same card. `-v` shows which device was used.

## Reviewing uncertain files

Files that need review go to `<output>/_review/<disc>/` (or stay in the rip folder
with `--review leave`). The plan's evidence section says why. Once you know what a
file is, file it with `--assign` (episode numbers are TMDB's aired numbers):

```bash
disc2plex apply "/srv/plex/TV Shows/_review/The Good Place Season One- Disc One" \
    --show "The Good Place" --season 1 --assign t04=6 --apply
# two episodes in one file:  --assign t00=1-2
```

An assigned file is placed exactly as you said; any other file that claimed the
same episode goes to review rather than overwriting or competing.

## DVD order and odd discs

```bash
disc2plex tmdb search "Firefly"
disc2plex tmdb show 1437                  # seasons and episode-group IDs
disc2plex tmdb season 1437 1 --order dvd  # the episode list in DVD order
disc2plex plan ./FIREFLY_D1 --show Firefly --disc 1 --order dvd
```

`--order dvd` (or an episode-group ID) decides which episodes are on the disc and
in what sequence; files are still named with aired numbers, which is what Plex's
TMDB agent matches. Play-All titles and junk are left in place; extras go to
`_extras_review/` (`--extras skip` leaves them too).

## Config file

`~/.config/disc2plex/config.toml` (or `$DISC2PLEX_CONFIG`). Command-line flags win
over the file; the file wins over built-in defaults. `disc2plex config init`
writes a commented template; `disc2plex config show` prints the effective values.

```toml
[library]
root = "/srv/plex/TV Shows"   # so you can drop --output
review = "move"               # or "leave"

[matching]
llm = "claude"                # verify uncertain matches by default ("local" for your own server)
confident = 0.85              # thresholds for library / review / unmatched
review = 0.5

[local_llm]                   # used with llm = "local"
base_url = "http://127.0.0.1:8090/v1"
model = "qwen"

[plex]
url = "http://localhost:32400"
scan_after_apply = true       # needs PLEX_TOKEN in the environment or .env
```

## Plex library scan (optional)

With `[plex]` configured, `apply --apply --plex-scan` (or `scan_after_apply = true`)
asks Plex to scan just the show folders that received files. The library section
is found from its folder; if Plex sees the library at a different path (for
example a share), set `plex.library_path`, or set `plex.section` directly
(`disc2plex plex sections` lists them). A failed scan request is only a warning.

## JSON output

`scan`, `plan`, `apply`, `undo`, and `transcribe` accept `--json`. `apply --apply
--json` needs `--yes` and prints the plan plus the manifest of what was done.

## Development

```bash
uv run pytest -q
DISC2PLEX_REAL_WHISPER=1 uv run pytest -q tests/test_transcribe.py -k real   # real Whisper
DISC2PLEX_REAL_WHISPER=1 DISC2PLEX_WHISPER_DEVICE=cuda uv run pytest -q -s tests/test_transcribe.py -k real   # on the GPU
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

See `CLAUDE.md` for the design, milestones, and decisions log.
