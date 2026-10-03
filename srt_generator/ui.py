from pathlib import Path

from .media import format_seconds


def print_header(console: object, model_name: str, source: str, target: str, device: str) -> None:
    from rich.panel import Panel

    console.print(
        Panel.fit(
            f"[bold cyan]Gerador de SRT[/bold cyan]\n"
            f"Motor: [bold]faster-whisper[/bold] | Modelo: [bold]{model_name}[/bold] | "
            f"Origem: [bold]{source}[/bold] | Destino: [bold]{target}[/bold] | "
            f"Dispositivo: [bold]{device.upper()}[/bold]",
            border_style="bright_blue",
        )
    )


def print_selected_files_table(
    console: object,
    input_paths: list[Path],
    durations: dict[Path, float | None],
) -> None:
    from rich.table import Table

    table = Table(title="Arquivos Selecionados", header_style="bold magenta")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Arquivo", style="white")
    table.add_column("Tamanho", justify="right", style="green")
    table.add_column("Duracao", justify="right", style="yellow")
    for index, input_path in enumerate(input_paths, start=1):
        table.add_row(
            str(index),
            input_path.name,
            f"{input_path.stat().st_size / (1024 * 1024):.1f} MB",
            format_seconds(durations.get(input_path)),
        )
    console.print(table)
