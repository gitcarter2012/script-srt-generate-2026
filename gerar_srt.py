import argparse
import importlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tkinter as tk
from tkinter import filedialog
from typing import Callable


def ensure_dependencies() -> None:
    required = {
        "whisper": "openai-whisper",
        "deep_translator": "deep-translator",
        "imageio_ffmpeg": "imageio-ffmpeg",
        "rich": "rich",
    }
    missing_packages = []

    for module_name, package_name in required.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing_packages.append(package_name)

    if not missing_packages:
        return

    print("Instalando dependencias ausentes:", ", ".join(missing_packages))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])


def configure_translation_http_timeout(timeout_seconds: float = 15.0) -> None:
    requests_module = importlib.import_module("requests")
    original_get = requests_module.get

    def get_with_timeout(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("timeout", timeout_seconds)
        return original_get(*args, **kwargs)

    requests_module.get = get_with_timeout


def get_nvidia_gpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return

    print("FFmpeg nao encontrado no sistema. Preparando instalacao automatica...")

    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")

    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not ffmpeg_exe.exists():
        raise RuntimeError("Nao foi possivel obter o executavel do FFmpeg automaticamente.")

    ffmpeg_dir = str(ffmpeg_exe.parent)
    current_path = os.environ.get("PATH", "")
    if ffmpeg_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path

    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg nao foi localizado mesmo apos a instalacao automatica.")

    print(f"FFmpeg pronto para uso: {ffmpeg_exe}")


def ensure_torch_cuda() -> object:
    torch = importlib.import_module("torch")
    if torch.cuda.is_available():
        return torch

    gpu_name = get_nvidia_gpu_name()
    if not gpu_name:
        return torch

    if os.environ.get("GENERAR_SRT_CUDA_ATTEMPTED") == "1":
        print("GPU NVIDIA detectada, mas o PyTorch continua sem CUDA. Usando CPU nesta execucao.")
        return torch

    print(f"GPU NVIDIA detectada: {gpu_name}")
    print("O PyTorch instalado e CPU-only. Tentando instalar uma versao com CUDA automaticamente...")

    cuda_indexes = [
        "https://download.pytorch.org/whl/cu124",
        "https://download.pytorch.org/whl/cu121",
    ]

    install_ok = False
    for index_url in cuda_indexes:
        try:
            print(f"Tentando instalar torch com CUDA em: {index_url}")
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--force-reinstall",
                    "torch",
                    "--index-url",
                    index_url,
                ]
            )
            install_ok = True
            break
        except subprocess.CalledProcessError:
            print(f"Falha ao instalar torch via {index_url}.")

    if not install_ok:
        print("Nao foi possivel instalar PyTorch com CUDA automaticamente. Seguindo em CPU nesta execucao.")
        os.environ["GENERAR_SRT_CUDA_ATTEMPTED"] = "1"
        return torch

    os.environ["GENERAR_SRT_CUDA_ATTEMPTED"] = "1"
    script_path = str(Path(__file__).resolve())
    restart_args = [sys.executable, script_path, *sys.argv[1:]]
    raise SystemExit(subprocess.call(restart_args))


def choose_device_mode() -> str:
    print("\nEscolha o modo de execucao:")
    print("  1) Com CUDA (GPU NVIDIA)")
    print("  2) Sem CUDA (CPU)")
    try:
        choice = input("Digite 1 ou 2 [padrao 1]: ").strip()
    except EOFError:
        return "cuda"

    if choice == "2":
        return "cpu"
    return "cuda"


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

    mapping = {
        "": default_language,
        "1": "ja",
        "2": "es",
        "3": "pt",
    }
    return mapping.get(choice, default_language)


def select_input_files() -> list[str]:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_types = [
        (
            "Midia suportada",
            "*.mp4 *.mkv *.mov *.avi *.wmv *.webm *.mp3 *.wav *.m4a *.flac *.aac",
        ),
        ("Todos os arquivos", "*.*"),
    ]
    selected = filedialog.askopenfilenames(title="Selecione um ou mais videos/audios", filetypes=file_types)
    root.destroy()
    return list(selected)


def print_selected_files(input_paths: list[Path]) -> None:
    # Mantido por compatibilidade; a exibicao principal usa a tabela colorida.
    for index, input_path in enumerate(input_paths, start=1):
        size_mb = input_path.stat().st_size / (1024 * 1024)
        print(f"[{index}] {input_path.name} ({size_mb:.1f} MB)")


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


def get_media_duration_seconds(input_path: Path) -> float | None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return None

    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-i", str(input_path)],
            capture_output=True,
            text=False,
            check=False,
        )
    except OSError:
        return None

    stderr_text = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
    stdout_text = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
    text = stderr_text + "\n" + stdout_text
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        return None

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def default_speed_ratio(model_name: str, is_cuda: bool) -> float:
    cuda_ratio = {
        "tiny": 0.15,
        "base": 0.25,
        "small": 0.40,
        "medium": 0.80,
        "large": 1.40,
    }
    cpu_ratio = {
        "tiny": 0.70,
        "base": 1.20,
        "small": 2.00,
        "medium": 3.50,
        "large": 6.00,
    }
    return (cuda_ratio if is_cuda else cpu_ratio).get(model_name, 2.0)


def estimate_processing_seconds(duration_s: float | None, observed_ratios: list[float], model_name: str, is_cuda: bool) -> float | None:
    if duration_s is None:
        return None
    ratio = (sum(observed_ratios) / len(observed_ratios)) if observed_ratios else default_speed_ratio(model_name, is_cuda)
    return max(10.0, duration_s * ratio)


def print_header(console: object, model_name: str, source: str, target: str, device: str) -> None:
    from rich.panel import Panel

    console.print(
        Panel.fit(
            f"[bold cyan]Gerador de SRT[/bold cyan]\n"
            f"Modelo: [bold]{model_name}[/bold] | Origem: [bold]{source}[/bold] | Destino: [bold]{target}[/bold] | Dispositivo: [bold]{device.upper()}[/bold]",
            border_style="bright_blue",
        )
    )


def print_selected_files_table(console: object, input_paths: list[Path], durations: dict[Path, float | None], estimates: dict[Path, float | None]) -> None:
    from rich.table import Table

    table = Table(title="Arquivos Selecionados", header_style="bold magenta", show_lines=False)
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Arquivo", style="white")
    table.add_column("Tamanho", justify="right", style="green")
    table.add_column("Duracao", justify="right", style="yellow")
    table.add_column("Tempo previsto", justify="right", style="bright_blue")

    for index, input_path in enumerate(input_paths, start=1):
        size_mb = input_path.stat().st_size / (1024 * 1024)
        table.add_row(
            str(index),
            input_path.name,
            f"{size_mb:.1f} MB",
            format_seconds(durations.get(input_path)),
            format_seconds(estimates.get(input_path)),
        )

    console.print(table)


def get_runtime_device(torch_module: object, preferred_mode: str) -> tuple[str, bool]:
    if preferred_mode == "cpu":
        return "cpu", False

    if preferred_mode == "cuda" and not torch_module.cuda.is_available():
        print("CUDA foi solicitado, mas nao esta disponivel. Continuando em CPU.")
        return "cpu", False

    if torch_module.cuda.is_available():
        return "cuda", True
    return "cpu", False


def split_segments_by_word_limit(segments: list[dict[str, object]], max_words: int = 5) -> list[dict[str, object]]:
    limited_segments = []
    for segment in segments:
        words = str(segment.get("text", "")).split()
        normalized_segment = segment.copy()
        normalized_segment["text"] = " ".join(words)
        if len(words) <= max_words:
            limited_segments.append(normalized_segment)
            continue

        start = float(segment["start"])
        duration = float(segment["end"]) - start
        word_offset = 0
        for chunk_start in range(0, len(words), max_words):
            chunk = words[chunk_start:chunk_start + max_words]
            next_word_offset = word_offset + len(chunk)
            chunk_segment = segment.copy()
            chunk_segment["text"] = " ".join(chunk)
            chunk_segment["start"] = start + duration * word_offset / len(words)
            chunk_segment["end"] = start + duration * next_word_offset / len(words)
            limited_segments.append(chunk_segment)
            word_offset = next_word_offset

    return limited_segments


class FallbackTranslator:
    def __init__(self, providers: list[tuple[str, Callable[[], object]]]) -> None:
        self.providers = providers
        self.disabled_providers: set[str] = set()

    def translate(self, text: str) -> str:
        errors = []
        for provider_index, (provider_name, create_provider) in enumerate(self.providers):
            if provider_name in self.disabled_providers:
                continue
            try:
                return create_provider().translate(text)
            except Exception as exc:  # noqa: BLE001 - Failover entre provedores externos.
                errors.append(f"{provider_name}: {exc}")
                has_fallback = provider_index + 1 < len(self.providers)
                if has_fallback:
                    self.disabled_providers.add(provider_name)
                    print(f"Tradutor {provider_name} indisponivel. Alternando para outro provedor...")
                    continue
                raise RuntimeError("Provedor de traducao indisponivel: " + errors[-1]) from exc

        raise RuntimeError("Todos os provedores de traducao estao indisponiveis: " + " | ".join(errors))


def translate_text_strict(
    create_translator: Callable[[], object],
    text: str,
    min_interval_seconds: float = 0.25,
    initial_backoff_seconds: float = 1.0,
    max_retries: int = 2,
) -> str:
    stripped_text = text.strip()
    if not stripped_text:
        return text

    backoff_seconds = initial_backoff_seconds
    last_error = None
    for attempt in range(max_retries + 1):
        if min_interval_seconds > 0:
            time.sleep(min_interval_seconds)
        try:
            translated = create_translator().translate(stripped_text)
            return (translated or stripped_text).strip()
        except Exception as exc:  # noqa: BLE001 - O provedor pode falhar por rede ou limite.
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 8.0)

    raise RuntimeError("Nao foi possivel traduzir um trecho apos varias tentativas.") from last_error


def translate_packed_batch(
    texts: list[str],
    create_translator: Callable[[], object],
    min_interval_seconds: float,
) -> list[str]:
    if len(texts) == 1:
        return [translate_text_strict(create_translator, texts[0], min_interval_seconds)]

    markers = [f"<SRT_{index:06d}>" for index in range(len(texts))]
    packed_text = "\n".join(f"{marker}{text}" for marker, text in zip(markers, texts))

    translated = translate_text_strict(create_translator, packed_text, min_interval_seconds)
    try:
        marker_pattern = re.compile(r"<\s*SRT_(\d{6})\s*>", re.IGNORECASE)
        matches = list(marker_pattern.finditer(translated))
        if len(matches) != len(texts):
            raise ValueError("O tradutor alterou os marcadores do lote.")

        translated_texts = []
        for index, match in enumerate(matches):
            expected_index = int(match.group(1))
            if expected_index != index:
                raise ValueError("O tradutor reordenou os marcadores do lote.")
            text_start = match.end()
            text_end = matches[index + 1].start() if index + 1 < len(matches) else len(translated)
            translated_texts.append(translated[text_start:text_end].strip())

        if any(not text for text in translated_texts):
            raise ValueError("O tradutor retornou um segmento vazio.")
        return translated_texts
    except ValueError:
        middle = len(texts) // 2
        return (
            translate_packed_batch(texts[:middle], create_translator, min_interval_seconds)
            + translate_packed_batch(texts[middle:], create_translator, min_interval_seconds)
        )


def translate_segments_strict(
    segments: list[dict[str, object]],
    create_translator: Callable[[], object],
    min_interval_seconds: float = 0.25,
    max_batch_characters: int = 450,
) -> list[dict[str, object]]:
    translated_segments = [segment.copy() for segment in segments]
    batch_indexes = []
    batch_texts = []
    batch_characters = 0

    def flush_batch() -> None:
        nonlocal batch_indexes, batch_texts, batch_characters
        if not batch_texts:
            return
        translations = translate_packed_batch(batch_texts, create_translator, min_interval_seconds)
        for segment_index, translated_text in zip(batch_indexes, translations):
            translated_segments[segment_index]["text"] = translated_text
        batch_indexes = []
        batch_texts = []
        batch_characters = 0

    for segment_index, segment in enumerate(translated_segments):
        text = str(segment.get("text", "")).strip()
        if not text:
            segment["text"] = ""
            continue

        estimated_characters = len(text) + 20
        if batch_texts and batch_characters + estimated_characters > max_batch_characters:
            flush_batch()
        batch_indexes.append(segment_index)
        batch_texts.append(text)
        batch_characters += estimated_characters

    flush_batch()

    return translated_segments


def process_file(
    input_path: Path,
    whisper_module: object,
    create_translator: Callable[[], object],
    model: object,
    source_language: str,
    output_base: str,
    use_fp16: bool,
    console: object,
    predicted_seconds: float | None,
) -> tuple[float, int]:
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

    console.print(f"\n[bold cyan]Iniciando:[/bold cyan] {input_path.name}")
    if predicted_seconds is not None:
        console.print(f"[bright_black]Tempo previsto: {format_seconds(predicted_seconds)}[/bright_black]")

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

        progress.update(phases_task, description=f"{input_path.name} - transcricao")
        result = model.transcribe(str(input_path), language=source_language, fp16=use_fp16)
        progress.update(phases_task, advance=1)

        segments = result.get("segments", [])
        writer = whisper_module.utils.get_writer("srt", str(input_path.parent))

        progress.update(phases_task, description=f"{input_path.name} - gravando SRT original")
        original_result = result.copy()
        original_result["segments"] = split_segments_by_word_limit([segment.copy() for segment in segments])
        writer(original_result, f"{output_base}_sem_traducao")
        progress.update(phases_task, advance=1)

        progress.update(phases_task, description=f"{input_path.name} - traducao")
        translation_task = progress.add_task(f"{input_path.name} - segmentos", total=max(1, len(segments)))
        translated_segments = translate_segments_strict(
            segments=segments,
            create_translator=create_translator,
        )
        for _ in translated_segments:
            progress.update(translation_task, advance=1)
        result["segments"] = split_segments_by_word_limit(translated_segments)
        progress.update(phases_task, advance=1)
        progress.remove_task(translation_task)

        progress.update(phases_task, description=f"{input_path.name} - gravando SRT traduzido")
        writer(result, output_base)
        progress.update(phases_task, advance=1)

    elapsed = time.perf_counter() - started_at
    console.print(f"[bold green]Concluido:[/bold green] {input_path.parent / (output_base + '.srt')} ([yellow]{format_seconds(elapsed)}[/yellow])")
    return elapsed, len(result.get("segments", []))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerar SRT de um MP4 e traduzir automaticamente.")
    parser.add_argument("input", nargs="*", help="Caminho(s) do(s) arquivo(s) de video/audio")
    parser.add_argument("--model", default="small", help="Modelo Whisper: tiny, base, small, medium, large")
    parser.add_argument("--source", default="en", help="Idioma do áudio (ex: en)")
    parser.add_argument("--target", default="pt", help="Idioma de tradução (ex: pt)")
    parser.add_argument("--output", default=None, help="Nome base do arquivo de saída (sem extensão)")
    parser.add_argument("--device", choices=["ask", "cuda", "cpu"], default="ask", help="Dispositivo: ask (pergunta 1/2), cuda ou cpu")
    parser.add_argument("--source-menu", choices=["on", "off"], default="on", help="Menu de idioma: on (pergunta) ou off")
    args = parser.parse_args()

    ensure_dependencies()
    ensure_ffmpeg()
    configure_translation_http_timeout()

    whisper = importlib.import_module("whisper")
    deep_translator_module = importlib.import_module("deep_translator")
    GoogleTranslator = deep_translator_module.GoogleTranslator
    MyMemoryTranslator = deep_translator_module.MyMemoryTranslator
    selected_device_mode = choose_device_mode() if args.device == "ask" else args.device
    selected_source = choose_source_language(args.source) if args.source_menu == "on" else args.source
    torch = ensure_torch_cuda() if selected_device_mode == "cuda" else importlib.import_module("torch")
    Console = importlib.import_module("rich.console").Console
    console = Console(highlight=False)

    selected_inputs = args.input if args.input else select_input_files()
    if not selected_inputs:
        print("Nenhum arquivo selecionado.")
        return

    input_paths = [Path(item).expanduser().resolve() for item in selected_inputs]
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    if args.output and len(input_paths) > 1:
        raise ValueError("Use --output apenas quando houver um unico arquivo de entrada.")

    device, use_fp16 = get_runtime_device(torch, selected_device_mode)
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print_header(console, args.model, selected_source, args.target, device)
    if device == "cuda":
        console.print(f"[green]Usando GPU CUDA:[/green] {gpu_name}")
    else:
        console.print("[yellow]Usando CPU.[/yellow] Se voce tem GPU NVIDIA, o PyTorch instalado pode estar CPU-only.")

    durations: dict[Path, float | None] = {item: get_media_duration_seconds(item) for item in input_paths}
    observed_ratios: list[float] = []
    estimates: dict[Path, float | None] = {
        item: estimate_processing_seconds(durations.get(item), observed_ratios, args.model, device == "cuda")
        for item in input_paths
    }
    print_selected_files_table(console, input_paths, durations, estimates)

    model = whisper.load_model(args.model, device=device)
    mymemory_language_codes = {
        "en": "en-GB",
        "ja": "ja-JP",
        "es": "es-ES",
        "pt": "pt-PT",
    }
    providers: list[tuple[str, Callable[[], object]]] = [
        ("Google", lambda: GoogleTranslator(source=selected_source, target=args.target)),
    ]
    if selected_source in mymemory_language_codes and args.target in mymemory_language_codes:
        providers.append(
            (
                "MyMemory",
                lambda: MyMemoryTranslator(
                    source=mymemory_language_codes[selected_source],
                    target=mymemory_language_codes[args.target],
                ),
            )
        )
    translator_pool = FallbackTranslator(providers)
    create_translator = lambda: translator_pool

    for input_path in input_paths:
        output_base = Path(args.output).stem if args.output else input_path.stem
        predicted = estimate_processing_seconds(durations.get(input_path), observed_ratios, args.model, device == "cuda")
        elapsed, _segments = process_file(
            input_path=input_path,
            whisper_module=whisper,
            create_translator=create_translator,
            model=model,
            source_language=selected_source,
            output_base=output_base,
            use_fp16=use_fp16,
            console=console,
            predicted_seconds=predicted,
        )
        file_duration = durations.get(input_path)
        if file_duration and file_duration > 0:
            observed_ratios.append(elapsed / file_duration)


if __name__ == "__main__":
    main()
