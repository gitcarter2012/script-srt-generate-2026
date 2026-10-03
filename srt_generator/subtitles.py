from pathlib import Path
import re


WEAK_LINE_ENDINGS = {
    "a", "as", "ao", "aos", "com", "da", "das", "de", "do", "dos", "e",
    "em", "na", "nas", "no", "nos", "o", "os", "para", "por", "que", "um", "uma",
}


def _line_length(words: list[str]) -> int:
    return len(" ".join(words))


def _can_wrap_in_two_lines(words: list[str], max_line_characters: int) -> bool:
    if _line_length(words) <= max_line_characters:
        return True
    return any(
        _line_length(words[:index]) <= max_line_characters
        and _line_length(words[index:]) <= max_line_characters
        for index in range(1, len(words))
    )


def _split_into_cue_words(
    words: list[str],
    max_words: int,
    max_line_characters: int,
) -> list[list[str]]:
    chunks = []
    offset = 0
    while offset < len(words):
        end = min(offset + max_words, len(words))
        while end > offset + 1 and not _can_wrap_in_two_lines(
            words[offset:end], max_line_characters
        ):
            end -= 1

        if end < len(words):
            punctuation_candidates = [
                index
                for index in range(max(offset + 1, end - 3), end + 1)
                if re.search(r"[.!?;:,…][\"')\]]?$", words[index - 1])
            ]
            if punctuation_candidates:
                end = punctuation_candidates[-1]
            while (
                end > offset + 1
                and words[end - 1].strip(".,!?;:…\"'()[]").lower() in WEAK_LINE_ENDINGS
            ):
                end -= 1

        chunks.append(words[offset:end])
        offset = end

    if len(chunks) > 1 and len(chunks[-1]) <= 2:
        combined = chunks[-2] + chunks[-1]
        candidates = []
        for index in range(1, len(combined)):
            first_chunk = combined[:index]
            second_chunk = combined[index:]
            if (
                len(first_chunk) > max_words
                or len(second_chunk) > max_words
                or not _can_wrap_in_two_lines(first_chunk, max_line_characters)
                or not _can_wrap_in_two_lines(second_chunk, max_line_characters)
            ):
                continue
            weak_ending = first_chunk[-1].strip(".,!?;:…\"'()[]").lower() in WEAK_LINE_ENDINGS
            score = abs(_line_length(first_chunk) - _line_length(second_chunk))
            score += max_line_characters if weak_ending else 0
            candidates.append((score, index))
        if candidates:
            _, split_index = min(candidates)
            chunks[-2:] = [combined[:split_index], combined[split_index:]]

    semantic_chunks = []
    for chunk in chunks:
        wrapped_lines = _wrap_balanced_lines(chunk, max_line_characters).splitlines()
        has_weak_line_ending = (
            len(wrapped_lines) == 2
            and wrapped_lines[0].split()[-1].strip(".,!?;:…\"'()[]").lower() in WEAK_LINE_ENDINGS
        )
        if not has_weak_line_ending:
            semantic_chunks.append(chunk)
            continue

        punctuation_candidates = [
            index
            for index in range(2, len(chunk) - 1)
            if re.search(r"[.!?;:,…][\"')\]]?$", chunk[index - 1])
            and _can_wrap_in_two_lines(chunk[:index], max_line_characters)
            and _can_wrap_in_two_lines(chunk[index:], max_line_characters)
        ]
        if not punctuation_candidates:
            semantic_chunks.append(chunk)
            continue
        split_index = min(
            punctuation_candidates,
            key=lambda index: abs(_line_length(chunk[:index]) - _line_length(chunk[index:])),
        )
        semantic_chunks.extend((chunk[:split_index], chunk[split_index:]))
    return semantic_chunks


def _wrap_balanced_lines(words: list[str], max_line_characters: int) -> str:
    text = " ".join(words)
    if len(text) <= max_line_characters or len(words) < 2:
        return text

    candidates = []
    for index in range(1, len(words)):
        first_line = " ".join(words[:index])
        second_line = " ".join(words[index:])
        if len(first_line) > max_line_characters or len(second_line) > max_line_characters:
            continue
        weak_ending = words[index - 1].strip(".,!?;:…\"'()[]").lower() in WEAK_LINE_ENDINGS
        punctuation_ending = bool(re.search(r"[.!?;:,…][\"')\]]?$", words[index - 1]))
        score = abs(len(first_line) - len(second_line)) + (max_line_characters if weak_ending else 0)
        score -= max_line_characters * 2 if punctuation_ending else 0
        candidates.append((score, index))

    if not candidates:
        return text
    _, split_index = min(candidates)
    return " ".join(words[:split_index]) + "\n" + " ".join(words[split_index:])


def split_segments_by_word_limit(
    segments: list[dict[str, object]],
    max_words: int = 12,
    use_word_timestamps: bool = False,
    max_duration_seconds: float = 7.5,
    max_line_characters: int = 40,
) -> list[dict[str, object]]:
    limited_segments = []
    for segment in segments:
        words = str(segment.get("text", "")).split()
        normalized_segment = segment.copy()
        normalized_segment["text"] = " ".join(words)
        timed_words = segment.get("words", []) if use_word_timestamps else []
        has_word_timestamps = isinstance(timed_words, list) and len(timed_words) == len(words)
        chunks = _split_into_cue_words(words, max_words, max_line_characters)
        if len(chunks) <= 1:
            normalized_segment["text"] = _wrap_balanced_lines(words, max_line_characters)
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
        for chunk in chunks:
            chunk_size = len(chunk)
            next_word_offset = word_offset + chunk_size
            chunk_segment = segment.copy()
            chunk_segment["text"] = _wrap_balanced_lines(chunk, max_line_characters)
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
        text = "\n".join(
            normalized_line
            for line in str(segment.get("text", "")).splitlines()
            if (normalized_line := " ".join(line.split()))
        )
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
