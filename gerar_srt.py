"""Ponto de entrada compativel do gerador de SRT."""

from srt_generator.cli import main
from srt_generator.config import get_deepl_api_key
from srt_generator.subtitles import split_segments_by_word_limit
from srt_generator.translation import (
    DeepLCloudTranslator,
    DeepLNonRetryableError,
    translate_packed_batch,
    translate_segments_strict,
)

__all__ = [
    "DeepLCloudTranslator",
    "DeepLNonRetryableError",
    "get_deepl_api_key",
    "main",
    "split_segments_by_word_limit",
    "translate_packed_batch",
    "translate_segments_strict",
]


if __name__ == "__main__":
    main()
