from pathlib import Path
import time
from typing import Callable

from .subtitles import split_segments_by_word_limit, write_srt
from .translation import translate_segments_strict


def process_file(
    input_path: Path,
    transcriber: object,
    create_translator: Callable[[], object],
    source_language: str,
    output_base: str,
    console: object,
) -> tuple[float, int]:
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

    console.print(f"\n[bold cyan]Iniciando:[/bold cyan] {input_path.name}")
    started_at = time.perf_counter()

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="green", finished_style="bright_green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        phases_task = progress.add_task(f"{input_path.name} - fases", total=4)
        transcription_task = progress.add_task(
            f"{input_path.name} - preparando transcricao",
            total=1,
        )

        def show_transcription(current: float, total: float) -> None:
            progress.update(
                transcription_task,
                total=total,
                completed=current,
                description=f"{input_path.name} - transcrevendo audio",
            )

        segments = transcriber.transcribe(input_path, source_language, show_transcription)
        progress.update(transcription_task, description=f"{input_path.name} - transcricao concluida")
        progress.update(phases_task, advance=1)

        progress.update(phases_task, description=f"{input_path.name} - gravando SRT original")
        original_segments = split_segments_by_word_limit(
            [segment.copy() for segment in segments],
            use_word_timestamps=True,
        )
        write_srt(original_segments, input_path.parent / f"{output_base}_sem_traducao.srt")
        progress.update(phases_task, advance=1)

        progress.update(phases_task, description=f"{input_path.name} - traducao")
        translation_task = progress.add_task(
            f"{input_path.name} - segmentos",
            total=max(1, len(segments)),
        )
        translated_count = 0

        def show_batch_start(batch_size: int, provider_name: str) -> None:
            progress.update(
                translation_task,
                description=f"{input_path.name} - enviando {batch_size} segmentos ao {provider_name}",
            )

        def show_batch_complete(batch_size: int, provider_name: str) -> None:
            nonlocal translated_count
            translated_count += batch_size
            progress.update(
                translation_task,
                advance=batch_size,
                description=f"{input_path.name} - {translated_count}/{len(segments)} traduzidos via {provider_name}",
            )

        translated_segments = translate_segments_strict(
            segments,
            create_translator,
            on_batch_start=show_batch_start,
            on_batch_complete=show_batch_complete,
        )
        if not segments:
            progress.update(translation_task, completed=1)
        progress.update(phases_task, advance=1)

        progress.update(phases_task, description=f"{input_path.name} - gravando SRT traduzido")
        final_segments = split_segments_by_word_limit(translated_segments)
        write_srt(final_segments, input_path.parent / f"{output_base}.srt")
        progress.update(phases_task, advance=1)

    elapsed = time.perf_counter() - started_at
    console.print(
        f"[bold green]Concluido:[/bold green] {input_path.parent / (output_base + '.srt')} "
        f"([yellow]{elapsed:.1f}s[/yellow])"
    )
    return elapsed, len(final_segments)
