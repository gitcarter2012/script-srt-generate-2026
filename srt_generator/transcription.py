import importlib
from pathlib import Path
from typing import Callable

from .runtime import model_cache_dir, try_install_nvidia_runtime


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

        def create_model(target_device: str, target_compute_type: str) -> object:
            return WhisperModel(
                model_name,
                device=target_device,
                compute_type=target_compute_type,
                download_root=str(model_cache_dir()),
            )

        try:
            self.model = create_model(device, compute_type)
        except (OSError, RuntimeError) as exc:
            if device != "cuda":
                raise
            if try_install_nvidia_runtime():
                try:
                    self.model = create_model(device, compute_type)
                    return
                except (OSError, RuntimeError) as retry_exc:
                    exc = retry_exc
            if on_device_fallback:
                on_device_fallback(str(exc))
            self.device = "cpu"
            self.compute_type = "int8"
            self.model = create_model("cpu", "int8")

    def transcribe(
        self,
        input_path: Path,
        source_language: str,
        on_progress: Callable[[float, float], None] | None = None,
    ) -> list[dict[str, object]]:
        segments_generator, info = self.model.transcribe(
            str(input_path),
            language=source_language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 300,
            },
            condition_on_previous_text=False,
            temperature=[0.0, 0.2, 0.4],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            hallucination_silence_threshold=2.0,
        )
        total_duration = max(float(getattr(info, "duration", 0.0)), 0.001)
        normalized_segments = []
        for segment in segments_generator:
            words = [
                {
                    "word": word.word,
                    "start": float(word.start),
                    "end": float(word.end),
                    "probability": float(word.probability),
                }
                for word in (segment.words or [])
            ]
            normalized_segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text.strip(),
                    "words": words,
                }
            )
            if on_progress:
                on_progress(min(float(segment.end), total_duration), total_duration)

        if on_progress:
            on_progress(total_duration, total_duration)
        return normalized_segments


class LegacyWhisperTranscriber:
    def __init__(self, model_name: str, device: str) -> None:
        import whisper

        self.model_name = model_name
        self.device = device
        self.compute_type = "float16" if device == "cuda" else "float32"
        self.model = whisper.load_model(model_name, device=device)

    def transcribe(
        self,
        input_path: Path,
        source_language: str,
        on_progress: Callable[[float, float], None] | None = None,
    ) -> list[dict[str, object]]:
        whisper_transcribe = importlib.import_module("whisper.transcribe")

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
                word_timestamps=True,
                condition_on_previous_text=False,
                hallucination_silence_threshold=2.0,
                temperature=(0.0, 0.2, 0.4),
                beam_size=5,
            )
        finally:
            whisper_transcribe.tqdm.tqdm = original_tqdm
        return [segment.copy() for segment in result.get("segments", [])]
