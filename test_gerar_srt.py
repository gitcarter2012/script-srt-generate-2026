import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gerar_srt import DeepLCloudTranslator, get_deepl_api_key, split_segments_by_word_limit, transcribe_with_rich_progress, translate_packed_batch


class FakeProgress:
    def __init__(self) -> None:
        self.events = []

    def add_task(self, *args: object, **kwargs: object) -> int:
        return 1

    def update(self, *args: object, **kwargs: object) -> None:
        self.events.append(kwargs)


class FakeModel:
    def __init__(self) -> None:
        self.transcribe_kwargs = {}

    def transcribe(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.transcribe_kwargs = kwargs
        transcribe_module = importlib.import_module("whisper.transcribe")
        with transcribe_module.tqdm.tqdm(total=100, unit="frames", disable=False) as progress:
            progress.update(40)
            progress.update(60)
        return {"segments": []}


class TranscriptionProgressTests(unittest.TestCase):
    def test_reports_whisper_frame_progress(self) -> None:
        progress = FakeProgress()
        model = FakeModel()
        transcribe_module = importlib.import_module("whisper.transcribe")
        original_tqdm = transcribe_module.tqdm.tqdm

        result = transcribe_with_rich_progress(
            model=model,
            input_path=Path("video.mp4"),
            source_language="en",
            use_fp16=False,
            progress=progress,
        )

        self.assertEqual(result, {"segments": []})
        self.assertEqual(sum(event.get("advance", 0) for event in progress.events), 100)
        self.assertEqual(progress.events[-1].get("completed"), 100)
        self.assertIs(transcribe_module.tqdm.tqdm, original_tqdm)
        self.assertTrue(model.transcribe_kwargs["word_timestamps"])
        self.assertFalse(model.transcribe_kwargs["condition_on_previous_text"])
        self.assertEqual(model.transcribe_kwargs["hallucination_silence_threshold"], 2.0)
        self.assertEqual(model.transcribe_kwargs["temperature"], (0.0, 0.2, 0.4))
        self.assertEqual(model.transcribe_kwargs["beam_size"], 5)


class SubtitleWordLimitTests(unittest.TestCase):
    def test_starts_new_caption_after_eight_words(self) -> None:
        segments = [{
            "start": 0.0,
            "end": 9.0,
            "text": "um dois tres quatro cinco seis sete oito nove",
        }]

        result = split_segments_by_word_limit(segments)

        self.assertEqual(result[0]["text"], "um dois tres quatro cinco seis sete oito")
        self.assertEqual(result[1]["text"], "nove")
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
        self.assertEqual(result[0]["end"], 3.8)
        self.assertEqual(result[1]["start"], 4.0)
        self.assertEqual(result[1]["end"], 4.3)


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