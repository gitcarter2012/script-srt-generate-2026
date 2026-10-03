from pathlib import Path


def get_media_duration_seconds(input_path: Path) -> float | None:
    try:
        import av

        with av.open(str(input_path)) as container:
            if container.duration is None:
                return None
            return float(container.duration / av.time_base)
    except (ImportError, OSError, ValueError):
        return None


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
