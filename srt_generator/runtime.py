import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys


COMMON_DEPENDENCIES = {
    "faster_whisper": "faster-whisper==1.2.1",
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


def choose_device_mode() -> str:
    print("\nEscolha o modo de execucao:")
    print("  1) Automatico (GPU NVIDIA quando disponivel)")
    print("  2) Somente CPU")
    try:
        choice = input("Digite 1 ou 2 [padrao 1]: ").strip()
    except EOFError:
        return "auto"
    return "cpu" if choice == "2" else "auto"


def model_cache_dir() -> Path:
    cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "generate-srt" / "models"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root
