import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys


COMMON_DEPENDENCIES = {
    "requests": "requests",
    "rich": "rich",
}


def ensure_dependencies() -> None:
    missing_packages = []
    for module_name, package_name in COMMON_DEPENDENCIES.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing_packages.append(package_name)
    if missing_packages:
        print("Instalando dependencias ausentes:", ", ".join(missing_packages))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
        importlib.invalidate_caches()


def ensure_transcription_dependencies(engine: str, preferred_mode: str) -> None:
    if engine == "faster":
        try:
            importlib.import_module("faster_whisper")
        except ModuleNotFoundError:
            print("Instalando faster-whisper...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "faster-whisper==1.2.1"])
        importlib.invalidate_caches()
        return

    legacy_dependencies = {
        "whisper": "openai-whisper",
        "imageio_ffmpeg": "imageio-ffmpeg",
        "torch": "torch",
    }
    missing_packages = []
    for module_name, package_name in legacy_dependencies.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing_packages.append(package_name)
    if missing_packages:
        print("Instalando dependencias do Whisper antigo:", ", ".join(missing_packages))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
        importlib.invalidate_caches()
    configure_legacy_ffmpeg()
    if preferred_mode != "cpu":
        ensure_legacy_torch_cuda()


def configure_legacy_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return
    imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
    ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = str(ffmpeg_path.parent) + os.pathsep + os.environ.get("PATH", "")


def ensure_legacy_torch_cuda() -> None:
    if not get_nvidia_gpu_name():
        return
    torch = importlib.import_module("torch")
    if torch.cuda.is_available():
        return
    if os.environ.get("GENERATE_SRT_LEGACY_CUDA_ATTEMPTED") == "1":
        print("PyTorch CUDA continua indisponivel apos a instalacao. O Whisper antigo usara CPU.")
        return
    os.environ["GENERATE_SRT_LEGACY_CUDA_ATTEMPTED"] = "1"
    print("Preparando PyTorch com CUDA para o Whisper antigo...")
    for index_name in ("cu128", "cu126", "cu124"):
        try:
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
                    f"https://download.pytorch.org/whl/{index_name}",
                ]
            )
            print("PyTorch CUDA instalado. Reiniciando o script para carregar as novas bibliotecas...")
            script_path = str(Path(sys.argv[0]).resolve())
            raise SystemExit(subprocess.call([sys.executable, script_path, *sys.argv[1:]]))
        except subprocess.CalledProcessError:
            print(f"PyTorch CUDA {index_name} nao foi instalado; tentando outra versao.")
    print("Nao foi possivel preparar PyTorch CUDA. O Whisper antigo usara CPU.")


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


def configure_nvidia_dll_paths() -> None:
    try:
        site_packages = Path(importlib.import_module("site").getsitepackages()[0])
    except (ImportError, IndexError):
        return
    candidates = [
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
    ]
    for directory in candidates:
        if not directory.exists():
            continue
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(directory))


def try_install_nvidia_runtime() -> bool:
    if sys.platform != "win32" or not get_nvidia_gpu_name():
        return False
    packages = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12"]
    print("Preparando bibliotecas CUDA 12/cuDNN 9 para faster-whisper...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
    except subprocess.CalledProcessError:
        print("Nao foi possivel instalar as bibliotecas CUDA via pip. O modo CPU continuara disponivel.")
        return False
    configure_nvidia_dll_paths()
    return True


def resolve_device(preferred_mode: str) -> tuple[str, str, str]:
    gpu_name = get_nvidia_gpu_name()
    if preferred_mode == "cpu" or not gpu_name:
        return "cpu", "int8", "CPU"

    configure_nvidia_dll_paths()
    try:
        ctranslate2 = importlib.import_module("ctranslate2")
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "int8_float16", gpu_name
    except (ImportError, OSError, RuntimeError):
        pass

    if try_install_nvidia_runtime():
        try:
            ctranslate2 = importlib.import_module("ctranslate2")
            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "int8_float16", gpu_name
        except (ImportError, OSError, RuntimeError):
            pass

    print("GPU NVIDIA detectada, mas o runtime CUDA nao iniciou. Usando CPU automaticamente.")
    return "cpu", "int8", "CPU"


def resolve_legacy_device(preferred_mode: str) -> tuple[str, str, str]:
    torch = importlib.import_module("torch")
    if preferred_mode != "cpu" and torch.cuda.is_available():
        return "cuda", "float16", str(torch.cuda.get_device_name(0))
    if preferred_mode != "cpu" and get_nvidia_gpu_name():
        print("CUDA nao ficou disponivel no PyTorch. O Whisper antigo usara CPU.")
    return "cpu", "float32", "CPU"


def choose_execution_profile() -> tuple[str, str]:
    print("\nEscolha o motor e o modo de execucao:")
    print("  1) faster-whisper + GPU automatica [RECOMENDADO]")
    print("     Melhor qualidade (large-v3), mais rapido; usa NVIDIA e cai para CPU se necessario.")
    print("  2) faster-whisper + CPU")
    print("     Melhor qualidade (large-v3), funciona sem NVIDIA; mais lento que GPU.")
    print("  3) Whisper antigo + GPU NVIDIA")
    print("     Boa qualidade (medium), compatibilidade com o fluxo antigo; mais pesado e mais lento.")
    print("  4) Whisper antigo + CPU")
    print("     Boa qualidade (medium), maxima compatibilidade; execucao muito lenta.")
    try:
        choice = input("Digite 1, 2, 3 ou 4 [padrao 1]: ").strip()
    except EOFError:
        return "faster", "auto"
    profiles = {
        "2": ("faster", "cpu"),
        "3": ("legacy", "cuda"),
        "4": ("legacy", "cpu"),
    }
    return profiles.get(choice, ("faster", "auto"))


def model_cache_dir() -> Path:
    cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "generate-srt" / "models"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root
