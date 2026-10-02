import importlib
from pathlib import Path
import unittest

from gerar_srt import transcribe_with_rich_progress


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


if __name__ == "__main__":
    unittest.main()