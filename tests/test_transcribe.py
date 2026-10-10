"""Tests for audio extraction and transcription.

Whisper is faked everywhere except ``test_real_whisper_on_synthesized_speech``,
which is opt-in (``DISC2PLEX_REAL_WHISPER=1``) because it loads a real model.
ffmpeg is real (on tiny generated clips in tmp_path).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from disc2plex.models import AudioTrack, MediaFile, SliceTranscript
from disc2plex.transcribe import (
    FasterWhisperTranscriber,
    SliceSpec,
    TranscriptCache,
    TranscriptionError,
    choose_audio_track,
    choose_slices,
    extract_audio,
    extraction_command,
    language_aliases,
    read_wav_samples,
    transcribe_file,
    whisper_language,
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def track(index: int, lang: str = "eng", **kwargs: object) -> AudioTrack:
    return AudioTrack(index=index, codec="ac3", language=lang, **kwargs)  # type: ignore[arg-type]


def media_with(
    *tracks: AudioTrack, path: Path = Path("/rip/x.mkv"), seconds: float = 1500
) -> MediaFile:
    return MediaFile(path=path, size_bytes=1, duration_seconds=seconds, audio_tracks=tracks)


# --- languages and track choice -------------------------------------------------


def test_language_aliases() -> None:
    assert language_aliases("en") == ("en", "eng")
    assert language_aliases("ENG") == ("en", "eng")
    assert language_aliases("fre") == ("fr", "fre", "fra")
    assert language_aliases("xx") == ("xx",)
    assert whisper_language("eng") == "en"


def test_choose_track_prefers_language() -> None:
    media = media_with(track(1, "fre", default=True, channels=6), track(2, "eng", channels=2))
    chosen = choose_audio_track(media, "en")
    assert chosen is not None and chosen.index == 2


def test_choose_track_avoids_commentary() -> None:
    # Real Blu-ray: t00 had two DTS 5.1 tracks plus a stereo commentary.
    media = media_with(
        track(1, channels=2, title="Commentary with cast", default=True),
        track(2, channels=6),
    )
    chosen = choose_audio_track(media, "en")
    assert chosen is not None and chosen.index == 2


def test_choose_track_prefers_default_then_channels() -> None:
    media = media_with(
        track(1, channels=2), track(2, channels=6), track(3, channels=6, default=True)
    )
    chosen = choose_audio_track(media, "en")
    assert chosen is not None and chosen.index == 3
    media = media_with(track(1, channels=2), track(2, channels=6))
    chosen = choose_audio_track(media, "en")
    assert chosen is not None and chosen.index == 2


def test_choose_track_falls_back_to_other_language() -> None:
    chosen = choose_audio_track(media_with(track(1, "jpn")), "en")
    assert chosen is not None and chosen.index == 1


def test_choose_track_none_without_audio() -> None:
    assert choose_audio_track(media_with(), "en") is None


# --- slices -------------------------------------------------------------------


def test_middle_slice() -> None:
    assert choose_slices(1500) == [SliceSpec(630.0, 240.0)]


def test_short_file_slice_is_clipped() -> None:
    assert choose_slices(100) == [SliceSpec(0.0, 100.0)]
    assert choose_slices(0) == []


def test_opening_slice() -> None:
    slices = choose_slices(1500, include_opening=True)
    assert slices[0] == SliceSpec(60.0, 120.0)
    assert len(slices) == 2
    assert len(choose_slices(150, include_opening=True)) == 1  # too short for an opening slice


# --- ffmpeg extraction --------------------------------------------------------


def test_extraction_command_seeks_before_input_and_maps_stream() -> None:
    cmd = extraction_command(Path("in.mkv"), 3, SliceSpec(12.5, 30), Path("out.wav"))
    assert cmd.index("-ss") < cmd.index("-i")
    assert cmd[cmd.index("-map") + 1] == "0:3"
    assert cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[cmd.index("-ac") + 1] == "1"


def make_clip(path: Path, seconds: int = 6, *, speech: str | None = None) -> None:
    """A tiny MKV: test video plus a sine tone, or flite-synthesized speech."""
    audio = f"flite=text='{speech}'" if speech else f"sine=frequency=440:duration={seconds}"
    video = f"testsrc=duration={seconds if not speech else 30}:size=64x48:rate=5"
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", video, "-f", "lavfi", "-i", audio]
    cmd += ["-shortest", "-c:v", "mpeg2video", "-c:a", "ac3"]
    cmd += ["-metadata:s:a:0", "language=eng", "-y", str(path)]
    subprocess.run(cmd, check=True)


@needs_ffmpeg
def test_extract_audio_writes_16k_mono_wav(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mkv"
    make_clip(clip)
    wav = tmp_path / "out.wav"
    extract_audio(clip, 1, SliceSpec(1.0, 3.0), wav)
    with wave.open(str(wav)) as handle:
        assert (handle.getnchannels(), handle.getframerate(), handle.getsampwidth()) == (
            1,
            16000,
            2,
        )
        assert handle.getnframes() == pytest.approx(3 * 16000, rel=0.05)
    samples = read_wav_samples(wav)
    assert samples.dtype.name == "float32"
    assert 0.05 < float(abs(samples).max()) <= 1.0


@needs_ffmpeg
def test_extract_audio_failure_is_reported(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.mkv"
    bogus.write_bytes(b"not a video")
    with pytest.raises(TranscriptionError, match="ffmpeg failed"):
        extract_audio(bogus, 1, SliceSpec(0, 3), tmp_path / "out.wav")


def test_read_wav_rejects_wrong_format(tmp_path: Path) -> None:
    wav = tmp_path / "stereo.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00" * 200)
    with pytest.raises(TranscriptionError, match="16 kHz mono"):
        read_wav_samples(wav)


def test_unknown_model_rejected() -> None:
    with pytest.raises(TranscriptionError, match="unknown Whisper model"):
        FasterWhisperTranscriber("gigantic")


# --- transcribe_file with a fake transcriber ----------------------------------


class FakeTranscriber:
    model_name = "fake/whisper"

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def transcribe(self, wav: Path, language: str | None) -> SliceTranscript:
        self.calls.append(wav)
        assert wav.exists() and language == "en"
        return SliceTranscript(
            start_seconds=0, duration_seconds=0, text=f"words #{len(self.calls)}"
        )


@pytest.fixture
def clip_media(tmp_path: Path) -> MediaFile:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    clip = tmp_path / "rip" / "title_t00.mkv"
    clip.parent.mkdir()
    make_clip(clip, seconds=8)
    return MediaFile(
        path=clip,
        size_bytes=clip.stat().st_size,
        duration_seconds=8.0,
        audio_tracks=(track(1, channels=1),),
    )


def test_transcribe_file_uses_slice_positions(clip_media: MediaFile) -> None:
    fake = FakeTranscriber()
    result = transcribe_file(clip_media, fake, slices=[SliceSpec(1, 2), SliceSpec(4, 3)])
    assert [s.text for s in result.slices] == ["words #1", "words #2"]
    assert [s.start_seconds for s in result.slices] == [1, 4]
    assert result.audio_stream_index == 1
    assert result.model == "fake/whisper"
    assert result.text == "words #1\nwords #2"
    assert all(not wav.exists() for wav in fake.calls)  # temp audio cleaned up


def test_transcript_cache(clip_media: MediaFile, tmp_path: Path) -> None:
    fake = FakeTranscriber()
    cache = TranscriptCache(tmp_path / "cache")
    first = transcribe_file(clip_media, fake, cache=cache)
    second = transcribe_file(clip_media, fake, cache=cache)
    assert len(fake.calls) == 1
    assert first.slices == second.slices
    # A different model is a different cache entry.
    other = FakeTranscriber()
    other.model_name = "fake/other"
    transcribe_file(clip_media, other, cache=cache)
    assert len(other.calls) == 1


def test_transcript_cache_invalidated_when_file_changes(
    clip_media: MediaFile, tmp_path: Path
) -> None:
    fake = FakeTranscriber()
    cache = TranscriptCache(tmp_path / "cache")
    transcribe_file(clip_media, fake, cache=cache)
    make_clip(clip_media.path, seconds=7)  # re-rip: new size/mtime
    transcribe_file(clip_media, fake, cache=cache)
    assert len(fake.calls) == 2


def test_transcribe_file_without_audio_raises() -> None:
    with pytest.raises(TranscriptionError, match="no audio"):
        transcribe_file(media_with(), FakeTranscriber())


# --- real Whisper (opt-in manual eval) ----------------------------------------


@needs_ffmpeg
@pytest.mark.skipif(
    os.environ.get("DISC2PLEX_REAL_WHISPER") != "1",
    reason="set DISC2PLEX_REAL_WHISPER=1 to run real Whisper (loads/downloads a model)",
)
def test_real_whisper_on_synthesized_speech(tmp_path: Path) -> None:
    """Manual eval: flite speaks a sentence, real faster-whisper must hear it.

    Run with: DISC2PLEX_REAL_WHISPER=1 uv run pytest -q tests/test_transcribe.py -k real
    """
    clip = tmp_path / "speech.mkv"
    make_clip(clip, speech="Welcome to the good place. Everything is fine.")
    probe = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"]
    seconds = float(
        subprocess.run([*probe, str(clip)], capture_output=True, text=True, check=True).stdout
    )
    media = MediaFile(
        path=clip,
        size_bytes=clip.stat().st_size,
        duration_seconds=seconds,
        audio_tracks=(track(1, channels=1),),
    )
    model = os.environ.get("DISC2PLEX_WHISPER_MODEL", "small")
    result = transcribe_file(media, FasterWhisperTranscriber(model))
    text = result.text.lower()
    assert "good place" in text
    assert "fine" in text


# --- CLI ----------------------------------------------------------------------


@pytest.fixture
def cli_clip(clip_media: MediaFile, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    from disc2plex import cli

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "FasterWhisperTranscriber", lambda _model: FakeTranscriber())
    return clip_media.path.parent


def test_cli_transcribe_named_file(cli_clip: Path) -> None:
    from typer.testing import CliRunner

    from disc2plex import cli

    result = CliRunner().invoke(cli.app, ["transcribe", str(cli_clip), "-f", "title_t00.mkv"])
    assert result.exit_code == 0, result.output
    assert "words #1" in result.output


def test_cli_transcribe_json(cli_clip: Path) -> None:
    import json

    from typer.testing import CliRunner

    from disc2plex import cli

    result = CliRunner().invoke(
        cli.app, ["transcribe", str(cli_clip), "-f", "title_t00.mkv", "--json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data[0]["slices"][0]["text"] == "words #1"


def test_cli_transcribe_skips_junk_by_default(cli_clip: Path) -> None:
    from typer.testing import CliRunner

    from disc2plex import cli

    # The 8-second clip classifies as junk, so nothing is transcribed without --file/--all.
    result = CliRunner().invoke(cli.app, ["transcribe", str(cli_clip), "--json"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "[]"


def test_cli_transcribe_errors(cli_clip: Path) -> None:
    from typer.testing import CliRunner

    from disc2plex import cli

    runner = CliRunner()
    missing = runner.invoke(cli.app, ["transcribe", str(cli_clip), "-f", "nope.mkv"])
    assert missing.exit_code == 1 and "not found" in missing.output
    bad_model = runner.invoke(cli.app, ["transcribe", str(cli_clip), "--whisper-model", "huge"])
    assert bad_model.exit_code == 1 and "--whisper-model" in bad_model.output
