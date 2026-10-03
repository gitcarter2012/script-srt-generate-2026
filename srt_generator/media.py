from contextlib import contextmanager
from pathlib import Path
import subprocess
import tempfile
from typing import Iterator


def get_media_duration_seconds(input_path: Path) -> float | None:
    try:
        import av

        with av.open(str(input_path)) as container:
            if container.duration is None:
                return None
            return float(container.duration / av.time_base)
    except (ImportError, OSError, ValueError):
        return None


def get_audio_duration_seconds(input_path: Path) -> float | None:
    try:
        import av

        with av.open(str(input_path)) as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            if stream.duration is None:
                return None
            return float(stream.duration * stream.time_base)
    except (ImportError, OSError, ValueError):
        return None


@contextmanager
def recovered_audio_path(input_path: Path) -> Iterator[Path]:
    import imageio_ffmpeg

    with tempfile.TemporaryDirectory(prefix="generate-srt-audio-") as temporary_directory:
        output_path = Path(temporary_directory) / "recovered.wav"
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-v",
                "error",
                "-i",
                str(input_path),
                "-map",
                "0:a:0",
                "-af",
                "aresample=async=1:first_pts=0",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not output_path.exists():
            details = result.stderr.strip().splitlines()
            message = details[-1] if details else "erro desconhecido"
            raise RuntimeError(f"Nao foi possivel recuperar o audio: {message}")
        yield output_path


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    total = int(max(0, round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
