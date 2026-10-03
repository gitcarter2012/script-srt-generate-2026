import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from gerar_srt import DeepLCloudTranslator, get_deepl_api_key, split_segments_by_word_limit, translate_packed_batch
from srt_generator.runtime import choose_execution_profile, configure_huggingface_downloads
from srt_generator.subtitles import write_srt
from srt_generator.transcription import FasterWhisperTranscriber, LegacyWhisperTranscriber


class FakeWord:
    word = " hello"
    start = 1.0
    end = 1.5
    probability = 0.9


class FakeSegment:
    start = 1.0
    end = 2.0
    text = " Hello"
    words = [FakeWord()]


class FakeInfo:
    duration = 4.0


class FakeModel:
    def __init__(self) -> None:
        self.transcribe_kwargs: dict[str, object] = {}

    def transcribe(self, *args: object, **kwargs: object) -> tuple[object, FakeInfo]:
        self.transcribe_kwargs = kwargs
        return iter([FakeSegment()]), FakeInfo()


class FailingLazyCudaModel:
    def transcribe(self, *args: object, **kwargs: object) -> tuple[object, FakeInfo]:
        def fail_during_iteration() -> object:
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            yield

        return fail_during_iteration(), FakeInfo()


class FakeLegacyModel:
    def __init__(self) -> None:
        self.transcribe_kwargs: dict[str, object] = {}

    def transcribe(self, *args: object, **kwargs: object) -> dict[str, object]:
        whisper_transcribe = importlib.import_module("whisper.transcribe")

        self.transcribe_kwargs = kwargs
        with whisper_transcribe.tqdm.tqdm(total=100) as progress:
            progress.update(40)
            progress.update(60)
        return {"segments": [{"start": 0.0, "end": 1.0, "text": " Hello"}]}


class ExecutionProfileTests(unittest.TestCase):
    def test_maps_all_execution_profiles(self) -> None:
        expected = {
            "": ("faster", "auto"),
            "1": ("faster", "auto"),
            "2": ("faster", "cpu"),
            "3": ("legacy", "cuda"),
            "4": ("legacy", "cpu"),
        }
        for choice, profile in expected.items():
            with (
                self.subTest(choice=choice),
                patch("builtins.input", return_value=choice),
                patch("builtins.print"),
            ):
                self.assertEqual(choose_execution_profile(), profile)

    @patch("builtins.input", return_value="1")
    @patch("builtins.print")
    def test_describes_quality_and_hardware(self, print_mock: Mock, input_mock: Mock) -> None:
        choose_execution_profile()

        menu_text = "\n".join(str(item.args[0]) for item in print_mock.call_args_list)
        self.assertIn("Melhor qualidade", menu_text)
        self.assertIn("GPU NVIDIA", menu_text)
        self.assertIn("Whisper antigo", menu_text)
        self.assertIn("muito lenta", menu_text)

    @patch.dict("os.environ", {}, clear=True)
    def test_configures_quiet_huggingface_downloads(self) -> None:
        configure_huggingface_downloads()

        import os
        from huggingface_hub.utils import logging as hub_logging

        self.assertEqual(os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"], "1")
        self.assertEqual(hub_logging.get_verbosity(), hub_logging.ERROR)


class TranscriptionProgressTests(unittest.TestCase):
    def test_reports_faster_whisper_duration_progress(self) -> None:
        model = FakeModel()
        progress_events = []
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        transcriber.model = model

        result = transcriber.transcribe(
            Path("video.mp4"),
            "en",
            lambda current, total: progress_events.append((current, total)),
        )

        self.assertEqual(result[0]["text"], "Hello")
        self.assertEqual((result[0]["start"], result[0]["end"]), (1.0, 1.5))
        self.assertEqual(progress_events, [(2.0, 4.0), (4.0, 4.0)])
        self.assertTrue(model.transcribe_kwargs["word_timestamps"])
        self.assertTrue(model.transcribe_kwargs["vad_filter"])
        self.assertFalse(model.transcribe_kwargs["condition_on_previous_text"])
        self.assertEqual(model.transcribe_kwargs["hallucination_silence_threshold"], 2.0)
        self.assertEqual(model.transcribe_kwargs["temperature"], [0.0, 0.2, 0.4])
        self.assertEqual(model.transcribe_kwargs["beam_size"], 5)

    @patch("srt_generator.transcription.try_install_nvidia_runtime", return_value=True)
    @patch("faster_whisper.WhisperModel")
    def test_retries_cuda_after_installing_runtime(self, whisper_model: Mock, install_runtime: Mock) -> None:
        loaded_model = object()
        whisper_model.side_effect = [RuntimeError("missing DLL"), loaded_model]

        transcriber = FasterWhisperTranscriber("large-v3", "cuda", "int8_float16")

        self.assertIs(transcriber.model, loaded_model)
        install_runtime.assert_called_once_with()
        self.assertEqual(whisper_model.call_count, 2)
        self.assertTrue(all(item.kwargs["device"] == "cuda" for item in whisper_model.call_args_list))

    @patch("srt_generator.transcription.try_install_nvidia_runtime", return_value=True)
    @patch("faster_whisper.WhisperModel")
    def test_repairs_cuda_failure_during_lazy_transcription(
        self,
        whisper_model: Mock,
        install_runtime: Mock,
    ) -> None:
        whisper_model.side_effect = [FailingLazyCudaModel(), FakeModel()]
        transcriber = FasterWhisperTranscriber("large-v3", "cuda", "int8_float16")

        result = transcriber.transcribe(Path("video.mp4"), "ja")

        self.assertEqual(result[0]["text"], "Hello")
        self.assertEqual(transcriber.device, "cuda")
        install_runtime.assert_called_once_with()
        self.assertEqual(
            [item.kwargs["device"] for item in whisper_model.call_args_list],
            ["cuda", "cuda"],
        )

    @patch("srt_generator.transcription.try_install_nvidia_runtime", return_value=False)
    @patch("faster_whisper.WhisperModel")
    def test_falls_back_to_cpu_when_cuda_cannot_load(self, whisper_model: Mock, install_runtime: Mock) -> None:
        loaded_model = object()
        whisper_model.side_effect = [RuntimeError("missing DLL"), loaded_model]
        errors = []

        transcriber = FasterWhisperTranscriber(
            "large-v3",
            "cuda",
            "int8_float16",
            errors.append,
        )

        self.assertIs(transcriber.model, loaded_model)
        self.assertEqual((transcriber.device, transcriber.compute_type), ("cpu", "int8"))
        self.assertEqual(errors, ["missing DLL"])
        install_runtime.assert_called_once_with()
        self.assertEqual(
            [item.kwargs["device"] for item in whisper_model.call_args_list],
            ["cuda", "cpu"],
        )

    def test_reports_legacy_whisper_progress_and_options(self) -> None:
        whisper_transcribe = importlib.import_module("whisper.transcribe")

        model = FakeLegacyModel()
        progress_events = []
        original_tqdm = whisper_transcribe.tqdm.tqdm
        transcriber = LegacyWhisperTranscriber.__new__(LegacyWhisperTranscriber)
        transcriber.model = model
        transcriber.device = "cpu"

        result = transcriber.transcribe(
            Path("video.mp4"),
            "en",
            lambda current, total: progress_events.append((current, total)),
        )

        self.assertEqual(result[0]["text"], " Hello")
        self.assertEqual(progress_events[-1], (1.0, 1.0))
        self.assertIs(whisper_transcribe.tqdm.tqdm, original_tqdm)
        self.assertFalse(model.transcribe_kwargs["fp16"])
        self.assertFalse(model.transcribe_kwargs["word_timestamps"])
        self.assertFalse(model.transcribe_kwargs["condition_on_previous_text"])
        self.assertEqual(model.transcribe_kwargs["beam_size"], 5)


class SubtitleWordLimitTests(unittest.TestCase):
    def test_starts_new_caption_after_six_words(self) -> None:
        segments = [{
            "start": 0.0,
            "end": 9.0,
            "text": "um dois tres quatro cinco seis sete oito nove",
        }]

        result = split_segments_by_word_limit(segments)

        self.assertEqual(result[0]["text"], "um dois tres quatro cinco seis")
        self.assertEqual(result[1]["text"], "sete oito nove")
        self.assertEqual(result[0]["end"], result[1]["start"])

    def test_uses_real_word_timestamps_for_original_srt(self) -> None:
        words = [
            {"word": word, "start": index * 0.5, "end": index * 0.5 + 0.3}
            for index, word in enumerate("um dois tres quatro cinco seis sete oito nove".split())
        ]
        segments = [{
            "start": 0.0,
            "end": 9.0,
            "text": "um dois tres quatro cinco seis sete oito nove",
            "words": words,
        }]

        result = split_segments_by_word_limit(segments, use_word_timestamps=True)

        self.assertEqual(result[0]["start"], 0.0)
        self.assertEqual(result[0]["end"], 2.8)
        self.assertEqual(result[1]["start"], 3.0)
        self.assertEqual(result[1]["end"], 4.3)

    def test_trims_short_caption_to_spoken_word_timestamps(self) -> None:
        segments = [{
            "start": 10.0,
            "end": 30.0,
            "text": "duas palavras",
            "words": [
                {"word": "duas", "start": 12.0, "end": 12.4},
                {"word": "palavras", "start": 12.5, "end": 13.0},
            ],
        }]

        result = split_segments_by_word_limit(segments, use_word_timestamps=True)

        self.assertEqual((result[0]["start"], result[0]["end"]), (12.0, 13.0))

    def test_srt_numbering_remains_continuous_after_empty_text(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "primeiro"},
            {"start": 1.0, "end": 2.0, "text": ""},
            {"start": 2.0, "end": 3.0, "text": "terceiro"},
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "output.srt"

            write_srt(segments, output_path)

            content = output_path.read_text(encoding="utf-8")
        self.assertIn("\n\n2\n00:00:02,000", content)
        self.assertNotIn("\n\n3\n", content)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self.payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, object]:
        return self.payload


class DeepLCloudTranslatorTests(unittest.TestCase):
    @patch("requests.post")
    def test_uses_free_endpoint_and_brazilian_portuguese(self, post: object) -> None:
        post.return_value = FakeResponse(200, {"translations": [{"text": "Ola, mundo."}]})
        translator = DeepLCloudTranslator("secret:fx", "en", "pt")

        translated = translator.translate("Hello, world.")

        self.assertEqual(translated, "Ola, mundo.")
        request = post.call_args
        self.assertEqual(request.args[0], "https://api-free.deepl.com/v2/translate")
        self.assertEqual(request.kwargs["data"]["text"], ["Hello, world."])
        self.assertEqual(request.kwargs["data"]["target_lang"], "PT-BR")
        self.assertEqual(request.kwargs["data"]["formality"], "prefer_less")
        self.assertEqual(request.kwargs["data"]["model_type"], "prefer_quality_optimized")
        self.assertNotIn("secret:fx", request.kwargs["data"])

    @patch("requests.post")
    def test_preserves_native_batch_alignment(self, post: object) -> None:
        post.return_value = FakeResponse(
            200,
            {"translations": [{"text": "Primeiro"}, {"text": "Segundo"}]},
        )
        translator = DeepLCloudTranslator("paid-secret", "en", "pt")

        translated = translator.translate_many(["First", "Second"], context="Previous dialogue")

        self.assertEqual(translated, ["Primeiro", "Segundo"])
        self.assertEqual(post.call_args.kwargs["data"]["text"], ["First", "Second"])
        self.assertEqual(post.call_args.kwargs["data"]["context"], "Previous dialogue")

    @patch("requests.post")
    def test_reports_exhausted_quota(self, post: object) -> None:
        post.return_value = FakeResponse(456, {})
        translator = DeepLCloudTranslator("paid-secret", "en", "pt")

        with self.assertRaisesRegex(RuntimeError, "Cota mensal"):
            translator.translate("Hello")

    @patch("requests.post")
    def test_does_not_retry_exhausted_quota(self, post: object) -> None:
        post.return_value = FakeResponse(456, {})
        translator = DeepLCloudTranslator("paid-secret", "en", "pt")

        with self.assertRaisesRegex(RuntimeError, "Cota mensal"):
            translate_packed_batch(["Hello"], lambda: translator, 0.0)

        self.assertEqual(post.call_count, 1)


class DeepLConfigTests(unittest.TestCase):
    @patch.dict("os.environ", {"DEEPL_API_KEY": "environment-key"})
    def test_config_key_has_priority_over_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.json"
            config_path.write_text('{"deepl_api_key": "config-key"}', encoding="utf-8")

            self.assertEqual(get_deepl_api_key(config_path), "config-key")

    def test_reports_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.json"
            config_path.write_text("{invalid", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Nao foi possivel ler config.json"):
                get_deepl_api_key(config_path)


if __name__ == "__main__":
    unittest.main()