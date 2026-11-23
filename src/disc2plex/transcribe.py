"""Stage 5a: extract audio slices with ffmpeg and transcribe them with faster-whisper.

Side effects (ffmpeg subprocess, model loading, cache files) live here. Source
video is only ever read. Transcripts are cached on disk because transcription
is by far the slowest step of the pipeline.

Hardware note: faster-whisper runs on CTranslate2, which accelerates only on
NVIDIA CUDA. With device ``auto`` (the default) an NVIDIA card is used when
there is one, and anything that goes wrong there (most often the card's memory
being full of a local LLM) falls back to the CPU, where int8 quantisation keeps
the ``small`` model fast enough for a few minutes of audio. Other GPUs (AMD,
Intel) always mean CPU.
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
WHISPER_DEVICES = ("auto", "cpu", "cuda")
# GPU compute types, best first: half precision where the card supports it
# (not on GTX 10-series), then 8-bit weights, which need the least memory.
_CUDA_COMPUTE_TYPES = ("float16", "int8_float32", "int8", "float32")
# When a passage decodes badly, Whisper retries it by sampling this many
# candidates and keeping the best. On the GPU, sampling the default 5 at once
# is the most memory a run ever needs (measured with `small`: over 1 GB, and
# out of memory with 1.2 GB free, against 0.75 GB with 2), which is exactly
# what runs out when a local LLM shares the card. Two keep the retry and fit.
_CPU_BEST_OF = 5
_CUDA_BEST_OF = 2
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


def cuda_compute_types() -> frozenset[str]:
    """Compute types CTranslate2 supports on this machine's CUDA device (empty: no usable GPU)."""
    try:
        import ctranslate2  # heavy import, deferred

        if ctranslate2.get_cuda_device_count() < 1:
            return frozenset()
        return frozenset(ctranslate2.get_supported_compute_types("cuda"))
    except Exception as exc:  # a broken driver must not break CPU transcription
        log.debug("no usable CUDA device: %s", exc)
        return frozenset()


class FasterWhisperTranscriber:
    """faster-whisper on an NVIDIA GPU when possible, else on the CPU (int8).

    ``device``: ``"auto"`` tries the GPU and falls back to the CPU for the rest
    of the run if the GPU fails (no CUDA libraries, not enough free memory);
    ``"cuda"`` makes such a failure an error; ``"cpu"`` never touches the GPU.
    The model loads on first use.

    Model weights download from Hugging Face on first use into its standard
    cache (``~/.cache/huggingface``); ``HF_TOKEN`` from the environment/.env is
    picked up automatically and avoids anonymous rate limits.
    """

    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "auto",
        compute_type: str | None = None,
        cpu_threads: int = 0,
    ) -> None:
        if model_size not in WHISPER_MODELS:
            raise TranscriptionError(
                f"unknown Whisper model {model_size!r}; choose from {', '.join(WHISPER_MODELS)}"
            )
        if device not in WHISPER_DEVICES:
            raise TranscriptionError(
                f"unknown Whisper device {device!r}; choose from {', '.join(WHISPER_DEVICES)}"
            )
        self._model_size = model_size
        self._requested_device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        self._model: Any = None
        self._device: str | None = None  # decided when the model first loads

    @property
    def model_name(self) -> str:
        # No device here on purpose: a transcript is the same evidence whichever
        # processor produced it, so cached transcripts are shared between devices.
        return f"faster-whisper/{self._model_size}"

    @property
    def device(self) -> str | None:
        """The device in use ("cpu"/"cuda"), or None before the first transcription."""
        return self._device

    def _choose_device(self) -> tuple[str, str]:
        """(device, compute type) for the first load."""
        if self._requested_device != "cpu":
            supported = cuda_compute_types()
            if supported:
                best = next((t for t in _CUDA_COMPUTE_TYPES if t in supported), "default")
                return "cuda", self._compute_type or best
            if self._requested_device == "cuda":
                raise TranscriptionError(
                    "Whisper device is cuda, but no NVIDIA GPU that CTranslate2 can use was found"
                )
        return "cpu", self._compute_type or "int8"

    def _load(self, device: str, compute_type: str) -> Any:
        from faster_whisper import WhisperModel  # heavy import, deferred

        log.info("loading Whisper model %s on %s (%s)", self._model_size, device, compute_type)
        return WhisperModel(
            self._model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=self._cpu_threads,
        )

    def _run(self, wav: Path, language: str | None) -> SliceTranscript:
        segments, info = self._model.transcribe(
            read_wav_samples(wav),
            language=language,
            beam_size=5,
            best_of=_CUDA_BEST_OF if self._device == "cuda" else _CPU_BEST_OF,
            vad_filter=True,  # skip music/silence, which also curbs hallucinated text
            condition_on_previous_text=False,
        )
        # Segments are produced lazily, so GPU errors can surface here too.
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return SliceTranscript(
            start_seconds=0.0,
            duration_seconds=float(info.duration),
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
        )

    def _fall_back_to_cpu(self, exc: RuntimeError) -> None:
        """After a GPU failure: an error if the GPU was demanded, else reload on the CPU."""
        if self._requested_device == "cuda":
            raise TranscriptionError(f"Whisper could not run on the GPU: {exc}") from exc
        log.warning(
            "Whisper could not use the GPU (%s); using the CPU for the rest of this run", exc
        )
        self._model = None  # drops the GPU model, which frees its memory
        self._device = "cpu"
        self._model = self._load("cpu", "int8")

    def transcribe(self, wav: Path, language: str | None) -> SliceTranscript:
        # CTranslate2 reports GPU trouble (out of memory, missing CUDA libraries)
        # as RuntimeError, either while loading or part-way through a slice.
        if self._model is None:
            self._device, compute_type = self._choose_device()
            try:
                self._model = self._load(self._device, compute_type)
            except RuntimeError as exc:
                if self._device != "cuda":
                    raise
                self._fall_back_to_cpu(exc)
        try:
            return self._run(wav, language)
        except RuntimeError as exc:
            if self._device != "cuda":
                raise
            self._fall_back_to_cpu(exc)
            return self._run(wav, language)


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
