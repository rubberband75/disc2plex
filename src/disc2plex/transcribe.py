"""Stage 5a: extract audio slices with ffmpeg and transcribe them with faster-whisper.

Side effects (ffmpeg subprocess, model loading, cache files) live here. Source
video is only ever read. Transcripts are cached on disk because transcription
is by far the slowest step of the pipeline.

Hardware note: faster-whisper runs on CTranslate2, which accelerates only on
NVIDIA CUDA. On other machines (including AMD GPUs) it runs on the CPU, where
int8 quantisation keeps the ``small`` model fast enough for a few minutes of audio.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tempfile
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from disc2plex.config import cache_dir
from disc2plex.models import AudioTrack, FileTranscript, MediaFile, SliceTranscript

log = logging.getLogger(__name__)

WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")
FFMPEG_TIMEOUT_SECONDS = 120
_CACHE_VERSION = 1

# ffprobe reports ISO 639-2 codes ("eng"); users type 639-1 ("en"). Whisper wants 639-1.
_LANGUAGE_CODES = {
    "en": ("en", "eng"),
    "es": ("es", "spa"),
    "fr": ("fr", "fre", "fra"),
    "de": ("de", "ger", "deu"),
    "it": ("it", "ita"),
    "pt": ("pt", "por"),
    "ja": ("ja", "jpn"),
    "ko": ("ko", "kor"),
    "zh": ("zh", "chi", "zho"),
    "nl": ("nl", "dut", "nld"),
    "sv": ("sv", "swe"),
    "da": ("da", "dan"),
    "no": ("no", "nor"),
    "fi": ("fi", "fin"),
    "pl": ("pl", "pol"),
    "ru": ("ru", "rus"),
}


class TranscriptionError(Exception):
    """Audio could not be extracted or transcribed."""


def language_aliases(lang: str) -> tuple[str, ...]:
    """All codes that mean ``lang`` (e.g. "en" -> ("en", "eng"))."""
    lang = lang.lower()
    for codes in _LANGUAGE_CODES.values():
        if lang in codes:
            return codes
    return (lang,)


def whisper_language(lang: str) -> str:
    """The ISO 639-1 code Whisper expects for ``lang``."""
    return language_aliases(lang)[0]


# --- pure selection -----------------------------------------------------------


def _is_commentary(track: AudioTrack) -> bool:
    return "comment" in (track.title or "").lower()


def choose_audio_track(media: MediaFile, lang: str) -> AudioTrack | None:
    """Pick the track most likely to carry the episode's dialogue in ``lang``.

    Preference: matching language, not a commentary, the default track, more
    channels (main mixes over stereo extras), then the lowest stream index.
    Falls back to any track if none matches the language.
    """
    if not media.audio_tracks:
        return None
    wanted = language_aliases(lang)

    def rank(t: AudioTrack) -> tuple[bool, bool, bool, int, int]:
        return (
            t.language.lower() not in wanted,
            _is_commentary(t),
            not t.default,
            -t.channels,
            t.index,
        )

    return min(media.audio_tracks, key=rank)


@dataclass(frozen=True)
class SliceSpec:
    start_seconds: float
    duration_seconds: float


def choose_slices(
    duration_seconds: float,
    *,
    slice_seconds: float = 240.0,
    include_opening: bool = False,
    opening_offset_seconds: float = 60.0,
    opening_seconds: float = 120.0,
) -> list[SliceSpec]:
    """Where to listen: a slice centred on the middle (avoids recaps and credits).

    ``include_opening`` adds a short slice just after the cold open starts,
    which often names the episode's premise. Slices are clipped to the file.
    """
    if duration_seconds <= 0:
        return []
    length = min(slice_seconds, duration_seconds)
    slices = [SliceSpec(max(0.0, duration_seconds / 2 - length / 2), length)]
    if include_opening and duration_seconds > opening_offset_seconds + opening_seconds:
        slices.insert(0, SliceSpec(opening_offset_seconds, opening_seconds))
    return slices


# --- ffmpeg -------------------------------------------------------------------


def extraction_command(source: Path, stream_index: int, spec: SliceSpec, output: Path) -> list[str]:
    """ffmpeg command for a mono 16 kHz WAV slice (Whisper's native format).

    ``-ss`` before ``-i`` seeks by container index, which is fast on large files.
    """
    return [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{spec.start_seconds:.3f}",
        "-t",
        f"{spec.duration_seconds:.3f}",
        "-i",
        str(source),
        "-map",
        f"0:{stream_index}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(output),
    ]


def extract_audio(source: Path, stream_index: int, spec: SliceSpec, output: Path) -> None:
    """Write one audio slice of ``source`` to ``output`` (a .wav file)."""
    cmd = extraction_command(source, stream_index, spec, output)
    log.debug("running %s", cmd)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SECONDS, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionError(f"ffmpeg timed out extracting audio from {source.name}") from exc
    if proc.returncode != 0 or not output.exists() or output.stat().st_size == 0:
        message = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "no output"
        raise TranscriptionError(f"ffmpeg failed on {source.name}: {message}")


def read_wav_samples(wav: Path) -> npt.NDArray[np.float32]:
    """Decode our own 16 kHz mono 16-bit WAV into float32 samples in [-1, 1].

    Handing Whisper raw samples avoids its PyAV-based decoder, whose keyword
    arguments have drifted out of sync with PyAV releases (faster-whisper 1.2.1
    vs PyAV 19 fails with "unexpected keyword argument 'metadata_errors'").
    """
    with wave.open(str(wav), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 16000):
            raise TranscriptionError(f"{wav.name} is not 16 kHz mono 16-bit PCM")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


# --- transcribers -------------------------------------------------------------


class Transcriber(Protocol):
    """Anything that turns a WAV file into text. Tests use a fake."""

    @property
    def model_name(self) -> str: ...

    def transcribe(self, wav: Path, language: str | None) -> SliceTranscript: ...


class FasterWhisperTranscriber:
    """faster-whisper on CPU (int8) by default; the model loads on first use.

    Model weights download from Hugging Face on first use into its standard
    cache (``~/.cache/huggingface``); ``HF_TOKEN`` from the environment/.env is
    picked up automatically and avoids anonymous rate limits.
    """

    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,
    ) -> None:
        if model_size not in WHISPER_MODELS:
            raise TranscriptionError(
                f"unknown Whisper model {model_size!r}; choose from {', '.join(WHISPER_MODELS)}"
            )
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return f"faster-whisper/{self._model_size}"

    def _load(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel  # heavy import, deferred

            log.info("loading Whisper model %s on %s", self._model_size, self._device)
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=self._cpu_threads,
            )
        return self._model

    def transcribe(self, wav: Path, language: str | None) -> SliceTranscript:
        model = self._load()
        segments, info = model.transcribe(
            read_wav_samples(wav),
            language=language,
            beam_size=5,
            vad_filter=True,  # skip music/silence, which also curbs hallucinated text
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return SliceTranscript(
            start_seconds=0.0,
            duration_seconds=float(info.duration),
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
        )


# --- cache --------------------------------------------------------------------


def default_transcript_cache_dir() -> Path:
    return cache_dir() / "transcripts"


class TranscriptCache:
    """One JSON file per (file identity, stream, slice, model)."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _file(self, media: MediaFile, stream: int, spec: SliceSpec, model: str) -> Path:
        stat = media.path.stat()
        key = json.dumps(
            [
                _CACHE_VERSION,
                str(media.path.resolve()),
                stat.st_size,
                stat.st_mtime_ns,
                stream,
                round(spec.start_seconds, 3),
                round(spec.duration_seconds, 3),
                model,
            ]
        )
        return self.directory / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def get(
        self, media: MediaFile, stream: int, spec: SliceSpec, model: str
    ) -> SliceTranscript | None:
        try:
            raw = self._file(media, stream, spec, model).read_text(encoding="utf-8")
            return SliceTranscript.model_validate_json(raw)
        except (OSError, ValueError):
            return None

    def put(
        self, media: MediaFile, stream: int, spec: SliceSpec, model: str, value: SliceTranscript
    ) -> None:
        file = self._file(media, stream, spec, model)
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix(".tmp")
        tmp.write_text(value.model_dump_json(), encoding="utf-8")
        tmp.replace(file)


# --- orchestration ------------------------------------------------------------


def transcribe_file(
    media: MediaFile,
    transcriber: Transcriber,
    *,
    lang: str = "en",
    slices: Sequence[SliceSpec] | None = None,
    cache: TranscriptCache | None = None,
) -> FileTranscript:
    """Transcribe the chosen slices of ``media``'s best audio track."""
    track = choose_audio_track(media, lang)
    if track is None:
        raise TranscriptionError(f"{media.path.name} has no audio tracks")
    specs = list(slices) if slices is not None else choose_slices(media.duration_seconds)
    results: list[SliceTranscript] = []
    with tempfile.TemporaryDirectory(prefix="disc2plex-audio-") as tmp:
        for number, spec in enumerate(specs):
            cached = cache.get(media, track.index, spec, transcriber.model_name) if cache else None
            if cached is not None:
                log.debug(
                    "transcript cache hit for %s @ %.0fs", media.path.name, spec.start_seconds
                )
                results.append(cached)
                continue
            wav = Path(tmp) / f"slice{number}.wav"
            extract_audio(media.path, track.index, spec, wav)
            raw = transcriber.transcribe(wav, whisper_language(lang))
            result = raw.model_copy(
                update={
                    "start_seconds": spec.start_seconds,
                    "duration_seconds": spec.duration_seconds,
                }
            )
            log.info(
                "transcribed %s @ %.0fs: %d chars",
                media.path.name,
                spec.start_seconds,
                len(result.text),
            )
            if cache:
                cache.put(media, track.index, spec, transcriber.model_name, result)
            results.append(result)
    return FileTranscript(
        file=media.path,
        audio_stream_index=track.index,
        model=transcriber.model_name,
        slices=tuple(results),
    )
