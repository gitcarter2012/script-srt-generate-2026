import importlib
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from gerar_srt import DeepLCloudTranslator, get_deepl_api_key, split_segments_by_word_limit, translate_packed_batch
from srt_generator.cli import choose_source_language
from srt_generator.runtime import choose_execution_profile, configure_huggingface_downloads
from srt_generator.subtitles import write_srt
from srt_generator.transcription import (
    FasterWhisperTranscriber,
    LegacyWhisperTranscriber,
    _filter_low_confidence_short_segments,
    _merge_incomplete_nearby_segments,
    _remove_known_hallucinations,
    _rejoin_japanese_boundary_characters,
)


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


class FakeGappedSegment:
    start = 1.0
    end = 24.0
    text = " First second"
    words = [
        type("Word", (), {"word": " First", "start": 1.0, "end": 1.5, "probability": 0.9})(),
        type("Word", (), {"word": " second", "start": 22.0, "end": 22.5, "probability": 0.9})(),
    ]


class FakeModeratePauseSegment:
    start = 1.0
    end = 4.0
    text = " I'm not done"
    words = [
        type("Word", (), {"word": " I'm", "start": 1.0, "end": 1.5, "probability": 0.9})(),
        type("Word", (), {"word": " not", "start": 3.5, "end": 3.7, "probability": 0.9})(),
        type("Word", (), {"word": " done", "start": 3.7, "end": 4.0, "probability": 0.9})(),
    ]


class FakeInfo:
    duration = 4.0


class FakeModel:
    def __init__(self) -> None:
        self.transcribe_kwargs: dict[str, object] = {}

    def transcribe(self, *args: object, **kwargs: object) -> tuple[object, FakeInfo]:
        self.transcribe_kwargs = kwargs
        return iter([FakeSegment()]), FakeInfo()


class FakeGappedModel(FakeModel):
    def transcribe(self, *args: object, **kwargs: object) -> tuple[object, FakeInfo]:
        self.transcribe_kwargs = kwargs
        return iter([FakeGappedSegment()]), FakeInfo()


class FakeModeratePauseModel(FakeModel):
    def transcribe(self, *args: object, **kwargs: object) -> tuple[object, FakeInfo]:
        self.transcribe_kwargs = kwargs
        return iter([FakeModeratePauseSegment()]), FakeInfo()


class FakeSplitSentenceModel(FakeModel):
    def transcribe(self, *args: object, **kwargs: object) -> tuple[object, FakeInfo]:
        self.transcribe_kwargs = kwargs
        first = type(
            "Segment",
            (),
            {
                "start": 1.0,
                "end": 1.5,
                "text": " I'm",
                "words": [type("Word", (), {"word": " I'm", "start": 1.0, "end": 1.5, "probability": 0.9})()],
            },
        )()
        second = type(
            "Segment",
            (),
            {
                "start": 3.5,
                "end": 4.0,
                "text": " not done",
                "words": [
                    type("Word", (), {"word": " not", "start": 3.5, "end": 3.7, "probability": 0.9})(),
                    type("Word", (), {"word": " done", "start": 3.7, "end": 4.0, "probability": 0.9})(),
                ],
            },
        )()
        return iter([first, second]), FakeInfo()


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
    def test_language_menu_supports_requested_source_languages(self) -> None:
        expected = {"": "en", "1": "en", "2": "es", "3": "ja"}
        for choice, language in expected.items():
            with self.subTest(choice=choice), patch("builtins.input", return_value=choice), patch("builtins.print"):
                self.assertEqual(choose_source_language("en"), language)

    def test_language_menu_rejects_unsupported_choice(self) -> None:
        with patch("builtins.input", return_value="4"), patch("builtins.print"):
            self.assertEqual(choose_source_language("en"), "en")

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
        self.assertEqual(model.transcribe_kwargs["vad_parameters"]["threshold"], 0.5)
        self.assertEqual(model.transcribe_kwargs["vad_parameters"]["max_speech_duration_s"], 15)
        self.assertFalse(model.transcribe_kwargs["condition_on_previous_text"])
        self.assertEqual(model.transcribe_kwargs["hallucination_silence_threshold"], 2.0)
        self.assertEqual(model.transcribe_kwargs["temperature"], [0.0, 0.2, 0.4])
        self.assertEqual(model.transcribe_kwargs["beam_size"], 5)

    def test_splits_faster_whisper_segment_across_internal_silence(self) -> None:
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        transcriber.model = FakeGappedModel()

        result = transcriber.transcribe(Path("video.mp4"), "en")

        self.assertEqual(
            [(item["start"], item["end"], item["text"]) for item in result],
            [(1.0, 1.5, "First"), (22.0, 22.5, "second")],
        )

    def test_keeps_sentence_together_across_moderate_pause(self) -> None:
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        transcriber.model = FakeModeratePauseModel()

        result = transcriber.transcribe(Path("video.mp4"), "en")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "I'm not done")

    def test_merges_incomplete_sentence_returned_as_separate_segments(self) -> None:
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        transcriber.model = FakeSplitSentenceModel()

        result = transcriber.transcribe(Path("video.mp4"), "en")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "I'm not done")
        self.assertEqual((result[0]["start"], result[0]["end"]), (1.0, 4.0))

    def test_does_not_merge_incomplete_segments_past_duration_limit(self) -> None:
        segments = [
            {"start": 1.0, "end": 4.0, "text": "First", "words": []},
            {"start": 4.2, "end": 9.0, "text": "second", "words": []},
        ]

        result = _merge_incomplete_nearby_segments(segments)

        self.assertEqual(len(result), 2)

    def test_filters_only_short_segments_with_very_low_confidence(self) -> None:
        segments = [
            {
                "start": 1.0,
                "end": 1.5,
                "text": "Walking",
                "words": [{"word": " Walking", "probability": 0.1}],
            },
            {
                "start": 2.0,
                "end": 2.5,
                "text": "Yes",
                "words": [{"word": " Yes", "probability": 0.9}],
            },
            {
                "start": 3.0,
                "end": 5.0,
                "text": "long uncertain phrase",
                "words": [
                    {"word": " long", "probability": 0.1},
                    {"word": " uncertain", "probability": 0.1},
                    {"word": " phrase", "probability": 0.1},
                ],
            },
        ]

        result = _filter_low_confidence_short_segments(segments)

        self.assertEqual([segment["text"] for segment in result], ["Yes", "long uncertain phrase"])

    def test_rejoins_isolated_japanese_boundary_characters(self) -> None:
        segments = [
            {"start": 1.0, "end": 2.0, "text": "昼も 奥", "words": []},
            {"start": 10.0, "end": 11.0, "text": "さんのおっぱい 大", "words": []},
            {"start": 20.0, "end": 21.0, "text": "きくなった", "words": []},
        ]

        result = _rejoin_japanese_boundary_characters(segments)

        self.assertEqual(
            [segment["text"] for segment in result],
            ["昼も", "奥さんのおっぱい", "大きくなった"],
        )
        self.assertEqual(result[1]["start"], 10.0)

    def test_rejoins_repeated_two_character_japanese_fragment(self) -> None:
        segments = [
            {"start": 1.0, "end": 2.0, "text": "いや どん", "words": []},
            {"start": 10.0, "end": 11.0, "text": "どん固くなる", "words": []},
        ]

        result = _rejoin_japanese_boundary_characters(segments)

        self.assertEqual([segment["text"] for segment in result], ["いや", "どんどん固くなる"])

    def test_removes_known_japanese_hallucination_inside_speech(self) -> None:
        segments = [{
            "start": 1.0,
            "end": 2.0,
            "text": "お熱が出てますよ ご視聴ありがとうございました",
            "words": [],
        }]

        result = _remove_known_hallucinations(segments)

        self.assertEqual(result[0]["text"], "お熱が出てますよ")

    @patch("srt_generator.transcription.get_audio_duration_seconds", side_effect=[100.0, 100.0])
    @patch("srt_generator.transcription.get_media_duration_seconds", side_effect=[100.0, 100.0])
    @patch("srt_generator.transcription.recovered_audio_path")
    def test_recovers_audio_when_decoder_returns_truncated_duration(
        self,
        recovered_audio: Mock,
        media_duration: Mock,
        audio_duration: Mock,
    ) -> None:
        truncated_info = type("Info", (), {"duration": 10.0})()
        complete_info = type("Info", (), {"duration": 100.0})()
        model = Mock()
        model.transcribe.side_effect = [
            (iter([]), truncated_info),
            (iter([FakeSegment()]), complete_info),
        ]

        @contextmanager
        def recovered_path(input_path: Path) -> object:
            yield Path("recovered.wav")

        recovered_audio.side_effect = recovered_path
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        transcriber.model = model

        result = transcriber._transcribe_once(Path("broken.mp4"), "ja", None)

        self.assertEqual(result[0]["text"], "Hello")
        self.assertEqual(model.transcribe.call_count, 2)
        self.assertEqual(model.transcribe.call_args_list[1].args[0], "recovered.wav")

    @patch("faster_whisper.audio.decode_audio", return_value=[])
    def test_long_audio_decodes_once_for_all_windows(self, decode_audio: Mock) -> None:
        model = Mock()
        model.transcribe.return_value = (iter([]), FakeInfo())
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        transcriber.model = model

        result = transcriber._transcribe_in_windows(
            Path("long.wav"),
            601.0,
            {"language": "ja"},
            None,
        )

        self.assertEqual(result, [])
        decode_audio.assert_called_once_with("long.wav", sampling_rate=16000)
        self.assertEqual(model.transcribe.call_count, 3)

    @patch("faster_whisper.audio.decode_audio", return_value=[])
    def test_long_audio_keeps_only_window_core_segments(self, decode_audio: Mock) -> None:
        before_core = type("Segment", (), {"start": 4.0, "end": 4.5, "text": " before", "words": []})()
        inside_core = type("Segment", (), {"start": 6.0, "end": 6.5, "text": " inside", "words": []})()
        next_overlap = type("Segment", (), {"start": 305.5, "end": 305.9, "text": " next", "words": []})()
        next_core = type("Segment", (), {"start": 5.5, "end": 5.9, "text": " next", "words": []})()
        model = Mock()
        model.transcribe.side_effect = [
            (iter([]), FakeInfo()),
            (iter([before_core, inside_core, next_overlap]), FakeInfo()),
            (iter([next_core]), FakeInfo()),
        ]
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        transcriber.model = model

        result = transcriber._transcribe_in_windows(
            Path("long.wav"),
            601.0,
            {"language": "ja"},
            None,
        )

        self.assertEqual([segment["text"] for segment in result], ["inside", "next"])

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
        transcriber._speech_clip_timestamps = Mock(return_value=[1.0, 2.0, 4.0, 5.0])

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
        self.assertEqual(model.transcribe_kwargs["clip_timestamps"], [1.0, 2.0, 4.0, 5.0])
        self.assertFalse(model.transcribe_kwargs["condition_on_previous_text"])
        self.assertEqual(model.transcribe_kwargs["beam_size"], 5)

    def test_legacy_whisper_skips_transcription_without_detected_speech(self) -> None:
        model = Mock()
        progress_events = []
        transcriber = LegacyWhisperTranscriber.__new__(LegacyWhisperTranscriber)
        transcriber.model = model
        transcriber.device = "cpu"
        transcriber._speech_clip_timestamps = Mock(return_value=[])

        result = transcriber.transcribe(
            Path("silence.mp4"),
            "en",
            lambda current, total: progress_events.append((current, total)),
        )

        self.assertEqual(result, [])
        model.transcribe.assert_not_called()
        self.assertEqual(progress_events, [(1.0, 1.0)])


class SubtitleWordLimitTests(unittest.TestCase):
    def test_starts_new_caption_after_six_words(self) -> None:
        segments = [{
            "start": 0.0,
            "end": 9.0,
            "text": "um dois tres quatro cinco seis sete oito nove",
        }]

        result = split_segments_by_word_limit(segments)

        self.assertEqual(result[0]["text"], "um dois tres quatro cinco")
        self.assertEqual(result[1]["text"], "seis sete oito nove")
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
        self.assertEqual(result[0]["end"], 2.3)
        self.assertEqual(result[1]["start"], 2.5)
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