import importlib
from pathlib import Path
import re
from typing import Any, Callable

from .media import get_audio_duration_seconds, get_media_duration_seconds, recovered_audio_path
from .runtime import model_cache_dir, try_install_nvidia_runtime


MAX_WORD_GAP_SECONDS = 2.5
MAX_MERGED_SEGMENT_SECONDS = 7.5
MIN_SHORT_SEGMENT_WORD_PROBABILITY = 0.2
LONG_AUDIO_THRESHOLD_SECONDS = 1200
TRANSCRIPTION_WINDOW_SECONDS = 300
TRANSCRIPTION_WINDOW_OVERLAP_SECONDS = 5
KNOWN_HALLUCINATION_PHRASES = ("ご視聴ありがとうございました",)


def _split_segment_at_word_gaps(segment: Any, words: list[dict[str, object]]) -> list[dict[str, object]]:
    if not words:
        return [{
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
            "words": [],
        }]

    groups: list[list[dict[str, object]]] = [[words[0]]]
    for word in words[1:]:
        previous_word = groups[-1][-1]
        if float(word["start"]) - float(previous_word["end"]) > MAX_WORD_GAP_SECONDS:
            groups.append([])
        groups[-1].append(word)

    if len(groups) == 1:
        text_by_group = [segment.text.strip()]
    else:
        text_by_group = ["".join(str(word["word"]) for word in group).strip() for group in groups]

    return [
        {
            "start": float(group[0]["start"]),
            "end": float(group[-1]["end"]),
            "text": text,
            "words": group,
        }
        for group, text in zip(groups, text_by_group)
        if text
    ]


def _merge_incomplete_nearby_segments(
    segments: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged_segments: list[dict[str, object]] = []
    for segment in segments:
        if not merged_segments:
            merged_segments.append(segment)
            continue

        previous = merged_segments[-1]
        gap = float(segment["start"]) - float(previous["end"])
        merged_duration = float(segment["end"]) - float(previous["start"])
        previous_text = str(previous["text"]).rstrip()
        if (
            gap > MAX_WORD_GAP_SECONDS
            or merged_duration > MAX_MERGED_SEGMENT_SECONDS
            or previous_text.endswith((".", "!", "?", "…"))
        ):
            merged_segments.append(segment)
            continue

        previous["end"] = segment["end"]
        previous["text"] = f"{previous_text} {str(segment['text']).lstrip()}"
        previous_words = previous.get("words", [])
        segment_words = segment.get("words", [])
        if isinstance(previous_words, list) and isinstance(segment_words, list):
            previous["words"] = [*previous_words, *segment_words]

    return merged_segments


def _filter_low_confidence_short_segments(
    segments: list[dict[str, object]],
) -> list[dict[str, object]]:
    filtered_segments = []
    for segment in segments:
        words = segment.get("words", [])
        if not isinstance(words, list) or len(words) > 2 or not words:
            filtered_segments.append(segment)
            continue

        probabilities = [float(word.get("probability", 1.0)) for word in words]
        if sum(probabilities) / len(probabilities) >= MIN_SHORT_SEGMENT_WORD_PROBABILITY:
            filtered_segments.append(segment)
    return filtered_segments


def _rejoin_japanese_boundary_characters(
    segments: list[dict[str, object]],
) -> list[dict[str, object]]:
    rejoined_segments: list[dict[str, object]] = []
    pending_character = ""
    for segment in segments:
        text = str(segment["text"]).strip()
        should_rejoin = (
            len(pending_character) == 1
            and re.match(r"[\u3040-\u30ff\u3400-\u9fff]", text)
        ) or (
            len(pending_character) == 2
            and text.startswith(pending_character)
        )
        if pending_character and should_rejoin:
            text = pending_character + text
            pending_character = ""
        elif pending_character:
            if rejoined_segments:
                rejoined_segments[-1]["text"] = str(rejoined_segments[-1]["text"]) + pending_character
            pending_character = ""

        trailing_character = re.search(r"(?:^|\s)([\u3040-\u30ff\u3400-\u9fff]{1,2})$", text)
        if trailing_character:
            pending_character = trailing_character.group(1)
            text = text[:trailing_character.start()].rstrip()

        if text:
            normalized_segment = segment.copy()
            normalized_segment["text"] = text
            rejoined_segments.append(normalized_segment)

    if pending_character and rejoined_segments:
        rejoined_segments[-1]["text"] = str(rejoined_segments[-1]["text"]) + pending_character
    return rejoined_segments


def _remove_known_hallucinations(
    segments: list[dict[str, object]],
) -> list[dict[str, object]]:
    cleaned_segments = []
    for segment in segments:
        text = str(segment["text"])
        for phrase in KNOWN_HALLUCINATION_PHRASES:
            text = text.replace(phrase, "")
        text = " ".join(text.split())
        if text:
            cleaned_segment = segment.copy()
            cleaned_segment["text"] = text
            cleaned_segments.append(cleaned_segment)
    return cleaned_segments


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        on_device_fallback: Callable[[str], None] | None = None,
    ) -> None:
        from faster_whisper import WhisperModel

        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.on_device_fallback = on_device_fallback
        self._model_class = WhisperModel
        self._runtime_repair_attempted = False

        try:
            self.model = self._create_model(device, compute_type)
        except (OSError, RuntimeError) as exc:
            if device != "cuda":
                raise
            if try_install_nvidia_runtime():
                try:
                    self.model = self._create_model(device, compute_type)
                    return
                except (OSError, RuntimeError) as retry_exc:
                    exc = retry_exc
            self._fallback_to_cpu(exc)

    def _create_model(self, device: str, compute_type: str) -> object:
        return self._model_class(
            self.model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(model_cache_dir()),
        )

    def _fallback_to_cpu(self, error: BaseException) -> None:
        if self.on_device_fallback:
            self.on_device_fallback(str(error))
        self.device = "cpu"
        self.compute_type = "int8"
        self.model = self._create_model("cpu", "int8")

    def transcribe(
        self,
        input_path: Path,
        source_language: str,
        on_progress: Callable[[float, float], None] | None = None,
    ) -> list[dict[str, object]]:
        try:
            return self._transcribe_once(input_path, source_language, on_progress)
        except (OSError, RuntimeError) as exc:
            if self.device != "cuda":
                raise

            error_text = str(exc).lower()
            missing_cuda_library = any(name in error_text for name in ("cublas", "cudnn", "cuda", ".dll"))
            if missing_cuda_library and not self._runtime_repair_attempted:
                self._runtime_repair_attempted = True
                if try_install_nvidia_runtime():
                    try:
                        self.model = self._create_model("cuda", "int8_float16")
                        return self._transcribe_once(input_path, source_language, on_progress)
                    except (OSError, RuntimeError) as retry_exc:
                        exc = retry_exc

            self._fallback_to_cpu(exc)
            return self._transcribe_once(input_path, source_language, on_progress)

    def _transcribe_once(
        self,
        input_path: Path,
        source_language: str,
        on_progress: Callable[[float, float], None] | None,
        allow_audio_recovery: bool = True,
    ) -> list[dict[str, object]]:
        transcribe_options = {
            "language": source_language,
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": True,
            "vad_parameters": {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "max_speech_duration_s": 15,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 300,
            },
            "condition_on_previous_text": False,
            "temperature": [0.0, 0.2, 0.4],
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "hallucination_silence_threshold": 2.0,
        }
        segments_generator, info = self.model.transcribe(
            str(input_path),
            **transcribe_options,
        )
        total_duration = max(float(getattr(info, "duration", 0.0)), 0.001)
        known_durations = [
            duration
            for duration in (
                get_audio_duration_seconds(input_path),
                get_media_duration_seconds(input_path),
            )
            if duration
        ]
        expected_duration = max(known_durations, default=None)
        if (
            allow_audio_recovery
            and expected_duration
            and total_duration < expected_duration * 0.9
        ):
            print(
                "Audio decodificado parcialmente; recuperando a faixa completa "
                "com FFmpeg antes da transcricao..."
            )
            with recovered_audio_path(input_path) as recovered_path:
                return self._transcribe_once(
                    recovered_path,
                    source_language,
                    on_progress,
                    allow_audio_recovery=False,
                )

        if total_duration > LONG_AUDIO_THRESHOLD_SECONDS:
            return self._transcribe_in_windows(
                input_path,
                total_duration,
                transcribe_options,
                on_progress,
            )
        return self._consume_segments(segments_generator, total_duration, on_progress)

    def _transcribe_in_windows(
        self,
        input_path: Path,
        total_duration: float,
        transcribe_options: dict[str, object],
        on_progress: Callable[[float, float], None] | None,
    ) -> list[dict[str, object]]:
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(input_path), sampling_rate=16000)
        normalized_segments = []
        core_start = 0.0
        while core_start < total_duration:
            core_end = min(core_start + TRANSCRIPTION_WINDOW_SECONDS, total_duration)
            clip_start = max(0.0, core_start - TRANSCRIPTION_WINDOW_OVERLAP_SECONDS)
            clip_end = min(total_duration, core_end + TRANSCRIPTION_WINDOW_OVERLAP_SECONDS)
            sample_start = round(clip_start * 16000)
            sample_end = round(clip_end * 16000)
            segments_generator, _ = self.model.transcribe(
                audio[sample_start:sample_end],
                **transcribe_options,
            )
            window_segments = self._normalize_segments(
                [
                    segment
                    for segment in segments_generator
                    if float(segment.start) + clip_start >= core_start
                    and float(segment.start) + clip_start < core_end
                ],
                timestamp_offset=clip_start,
            )
            normalized_segments.extend(window_segments)
            if on_progress:
                on_progress(core_end, total_duration)
            core_start = core_end

        return self._finalize_segments(normalized_segments)

    def _consume_segments(
        self,
        segments_generator: object,
        total_duration: float,
        on_progress: Callable[[float, float], None] | None,
    ) -> list[dict[str, object]]:
        normalized_segments = []
        for segment in segments_generator:
            normalized_segments.extend(self._normalize_segments([segment]))
            if on_progress:
                on_progress(min(float(segment.end), total_duration), total_duration)

        if on_progress:
            on_progress(total_duration, total_duration)
        return self._finalize_segments(normalized_segments)

    @staticmethod
    def _normalize_segments(
        segments: object,
        timestamp_offset: float = 0.0,
    ) -> list[dict[str, object]]:
        normalized_segments = []
        for segment in segments:
            words = [
                {
                    "word": word.word,
                    "start": float(word.start) + timestamp_offset,
                    "end": float(word.end) + timestamp_offset,
                    "probability": float(word.probability),
                }
                for word in (segment.words or [])
            ]
            split_segments = _split_segment_at_word_gaps(segment, words)
            if not words and timestamp_offset:
                for split_segment in split_segments:
                    split_segment["start"] = float(split_segment["start"]) + timestamp_offset
                    split_segment["end"] = float(split_segment["end"]) + timestamp_offset
            normalized_segments.extend(split_segments)
        return normalized_segments

    @staticmethod
    def _finalize_segments(
        normalized_segments: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        merged_segments = _merge_incomplete_nearby_segments(normalized_segments)
        filtered_segments = _filter_low_confidence_short_segments(merged_segments)
        cleaned_segments = _remove_known_hallucinations(filtered_segments)
        return _rejoin_japanese_boundary_characters(cleaned_segments)


class LegacyWhisperTranscriber:
    def __init__(self, model_name: str, device: str) -> None:
        import whisper

        self.model_name = model_name
        self.device = device
        self.compute_type = "float16" if device == "cuda" else "float32"
        self.model = whisper.load_model(model_name, device=device)

    @staticmethod
    def _speech_clip_timestamps(input_path: Path) -> list[float]:
        import whisper
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        sampling_rate = 16000
        audio = whisper.load_audio(str(input_path), sr=sampling_rate)
        speech_chunks = get_speech_timestamps(
            audio,
            VadOptions(
                threshold=0.5,
                min_speech_duration_ms=250,
                max_speech_duration_s=15,
                min_silence_duration_ms=500,
                speech_pad_ms=300,
            ),
            sampling_rate=sampling_rate,
        )
        return [
            timestamp / sampling_rate
            for chunk in speech_chunks
            for timestamp in (chunk["start"], chunk["end"])
        ]

    def transcribe(
        self,
        input_path: Path,
        source_language: str,
        on_progress: Callable[[float, float], None] | None = None,
    ) -> list[dict[str, object]]:
        whisper_transcribe = importlib.import_module("whisper.transcribe")
        clip_timestamps = self._speech_clip_timestamps(input_path)
        if not clip_timestamps:
            if on_progress:
                on_progress(1.0, 1.0)
            return []

        original_tqdm = whisper_transcribe.tqdm.tqdm

        class ProgressAdapter:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.total = float(kwargs.get("total") or 1)
                self.completed = 0.0

            def __enter__(self) -> "ProgressAdapter":
                return self

            def __exit__(self, *args: object) -> None:
                if on_progress:
                    on_progress(self.total / 100, self.total / 100)

            def update(self, frames: int) -> None:
                self.completed += frames
                if on_progress:
                    on_progress(min(self.completed, self.total) / 100, self.total / 100)

        whisper_transcribe.tqdm.tqdm = ProgressAdapter
        try:
            result = self.model.transcribe(
                str(input_path),
                language=source_language,
                fp16=self.device == "cuda",
                verbose=False,
                word_timestamps=False,
                condition_on_previous_text=False,
                hallucination_silence_threshold=2.0,
                temperature=(0.0, 0.2, 0.4),
                beam_size=5,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                clip_timestamps=clip_timestamps,
            )
        finally:
            whisper_transcribe.tqdm.tqdm = original_tqdm
        return [segment.copy() for segment in result.get("segments", [])]
