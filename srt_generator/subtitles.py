from pathlib import Path
import math


def _balanced_chunk_sizes(word_count: int, max_words: int) -> list[int]:
    chunk_count = math.ceil(word_count / max_words)
    base_size, larger_chunks = divmod(word_count, chunk_count)
    return [
        base_size + (1 if index < larger_chunks else 0)
        for index in range(chunk_count)
    ]


def split_segments_by_word_limit(
    segments: list[dict[str, object]],
    max_words: int = 6,
    use_word_timestamps: bool = False,
    max_duration_seconds: float = 7.5,
) -> list[dict[str, object]]:
    limited_segments = []
    for segment in segments:
        words = str(segment.get("text", "")).split()
        normalized_segment = segment.copy()
        normalized_segment["text"] = " ".join(words)
        timed_words = segment.get("words", []) if use_word_timestamps else []
        has_word_timestamps = isinstance(timed_words, list) and len(timed_words) == len(words)
        if len(words) <= max_words:
            if has_word_timestamps and timed_words:
                normalized_segment["start"] = float(timed_words[0]["start"])
                normalized_segment["end"] = float(timed_words[-1]["end"])
            normalized_segment["end"] = min(
                float(normalized_segment["end"]),
                float(normalized_segment["start"]) + max_duration_seconds,
            )
            limited_segments.append(normalized_segment)
            continue

        start = float(segment["start"])
        duration = float(segment["end"]) - start
        word_offset = 0
        for chunk_size in _balanced_chunk_sizes(len(words), max_words):
            next_word_offset = word_offset + chunk_size
            chunk = words[word_offset:next_word_offset]
            chunk_segment = segment.copy()
            chunk_segment["text"] = " ".join(chunk)
            if has_word_timestamps:
                chunk_segment["start"] = float(timed_words[word_offset]["start"])
                chunk_segment["end"] = float(timed_words[next_word_offset - 1]["end"])
            else:
                chunk_segment["start"] = start + duration * word_offset / len(words)
                chunk_segment["end"] = start + duration * next_word_offset / len(words)
            chunk_segment["end"] = min(
                float(chunk_segment["end"]),
                float(chunk_segment["start"]) + max_duration_seconds,
            )
            limited_segments.append(chunk_segment)
            word_offset = next_word_offset

    return limited_segments


def format_srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: list[dict[str, object]], output_path: Path) -> None:
    blocks = []
    for segment in segments:
        text = " ".join(str(segment.get("text", "")).split())
        if not text:
            continue
        index = len(blocks) + 1
        blocks.append(
            f"{index}\n"
            f"{format_srt_timestamp(float(segment['start']))} --> "
            f"{format_srt_timestamp(float(segment['end']))}\n"
            f"{text}"
        )
    output_path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
