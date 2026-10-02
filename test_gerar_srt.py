import importlib
from pathlib import Path
import unittest
from unittest.mock import patch

from gerar_srt import DeepLCloudTranslator, split_segments_by_word_limit, transcribe_with_rich_progress


class FakeProgress:
    def __init__(self) -> None:
        self.events = []

    def add_task(self, *args: object, **kwargs: object) -> int:
        return 1

    def update(self, *args: object, **kwargs: object) -> None:
        self.events.append(kwargs)


class FakeModel:
    def transcribe(self, *args: object, **kwargs: object) -> dict[str, object]:
        transcribe_module = importlib.import_module("whisper.transcribe")
        with transcribe_module.tqdm.tqdm(total=100, unit="frames", disable=False) as progress:
            progress.update(40)
            progress.update(60)
        return {"segments": []}


class TranscriptionProgressTests(unittest.TestCase):
    def test_reports_whisper_frame_progress(self) -> None:
        progress = FakeProgress()
        transcribe_module = importlib.import_module("whisper.transcribe")
        original_tqdm = transcribe_module.tqdm.tqdm

        result = transcribe_with_rich_progress(
            model=FakeModel(),
            input_path=Path("video.mp4"),
            source_language="en",
            use_fp16=False,
            progress=progress,
        )

        self.assertEqual(result, {"segments": []})
        self.assertEqual(sum(event.get("advance", 0) for event in progress.events), 100)
        self.assertEqual(progress.events[-1].get("completed"), 100)
        self.assertIs(transcribe_module.tqdm.tqdm, original_tqdm)


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
        self.assertEqual(request.kwargs["data"]["formality"], "prefer_more")
        self.assertNotIn("secret:fx", request.kwargs["data"])

    @patch("requests.post")
    def test_preserves_native_batch_alignment(self, post: object) -> None:
        post.return_value = FakeResponse(
            200,
            {"translations": [{"text": "Primeiro"}, {"text": "Segundo"}]},
        )
        translator = DeepLCloudTranslator("paid-secret", "en", "pt")

        translated = translator.translate_many(["First", "Second"])

        self.assertEqual(translated, ["Primeiro", "Segundo"])
        self.assertEqual(post.call_args.kwargs["data"]["text"], ["First", "Second"])

    @patch("requests.post")
    def test_reports_exhausted_quota(self, post: object) -> None:
        post.return_value = FakeResponse(456, {})
        translator = DeepLCloudTranslator("paid-secret", "en", "pt")

        with self.assertRaisesRegex(RuntimeError, "Cota mensal"):
            translator.translate("Hello")


if __name__ == "__main__":
    unittest.main()