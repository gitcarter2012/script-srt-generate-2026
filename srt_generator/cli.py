import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from .config import get_deepl_api_key
from .runtime import choose_device_mode, ensure_dependencies, resolve_device


def choose_source_language(default_language: str) -> str:
    print("\nIdioma de origem do audio:")
    print(f"  Enter) Ingles padrao ({default_language})")
    print("  1) Japones (ja)")
    print("  2) Espanhol (es)")
    print("  3) Portugues (pt)")
    try:
        choice = input("Escolha Enter, 1, 2 ou 3: ").strip()
    except EOFError:
        return default_language
    return {"": default_language, "1": "ja", "2": "es", "3": "pt"}.get(choice, default_language)


def select_input_files() -> list[str]:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilenames(
        title="Selecione um ou mais videos/audios",
        filetypes=[
            ("Midia suportada", "*.mp4 *.mkv *.mov *.avi *.wmv *.webm *.mp3 *.wav *.m4a *.flac *.aac"),
            ("Todos os arquivos", "*.*"),
        ],
    )
    root.destroy()
    return list(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gerar SRT com faster-whisper e DeepL.")
    parser.add_argument("input", nargs="*", help="Caminho(s) de video/audio")
    parser.add_argument("--model", default="large-v3", help="Modelo faster-whisper (padrao: large-v3)")
    parser.add_argument("--source", default="en", help="Idioma do audio (padrao: en)")
    parser.add_argument("--target", default="pt", help="Idioma da traducao (padrao: pt)")
    parser.add_argument("--output", default=None, help="Nome base da saida, apenas para um arquivo")
    parser.add_argument("--device", choices=["ask", "auto", "cuda", "cpu"], default="ask")
    parser.add_argument("--source-menu", choices=["on", "off"], default="on")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dependencies()

    from rich.console import Console
    from .media import get_media_duration_seconds
    from .pipeline import process_file
    from .transcription import FasterWhisperTranscriber
    from .translation import DeepLCloudTranslator
    from .ui import print_header, print_selected_files_table

    api_key = get_deepl_api_key()
    preferred_mode = choose_device_mode() if args.device == "ask" else args.device
    source_language = choose_source_language(args.source) if args.source_menu == "on" else args.source
    input_values = args.input if args.input else select_input_files()
    if not input_values:
        print("Nenhum arquivo selecionado.")
        return

    input_paths = [Path(value).expanduser().resolve() for value in input_values]
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")
    if args.output and len(input_paths) > 1:
        raise ValueError("Use --output apenas com um arquivo de entrada.")

    device, compute_type, device_name = resolve_device(preferred_mode)
    console = Console(highlight=False)
    print_header(console, args.model, source_language, args.target, device_name)
    console.print(f"[cyan]Compute type:[/cyan] {compute_type}")
    durations = {path: get_media_duration_seconds(path) for path in input_paths}
    print_selected_files_table(console, input_paths, durations)

    def show_fallback(error: str) -> None:
        console.print("[yellow]CUDA falhou ao carregar o modelo; alternando para CPU int8.[/yellow]")
        console.print(f"[bright_black]{error}[/bright_black]")

    console.print(f"[cyan]Carregando modelo {args.model}. Na primeira execucao ele sera baixado automaticamente.[/cyan]")
    transcriber = FasterWhisperTranscriber(args.model, device, compute_type, show_fallback)
    console.print(f"[green]Modelo pronto em {transcriber.device.upper()} ({transcriber.compute_type}).[/green]")
    create_translator = lambda: DeepLCloudTranslator(api_key, source_language, args.target)

    for input_path in input_paths:
        output_base = Path(args.output).stem if args.output else input_path.stem
        process_file(
            input_path=input_path,
            transcriber=transcriber,
            create_translator=create_translator,
            source_language=source_language,
            output_base=output_base,
            console=console,
        )
