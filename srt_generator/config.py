import getpass
import json
import os
from pathlib import Path


def get_deepl_api_key(config_path: Path | None = None) -> str:
    resolved_config_path = config_path or Path(__file__).resolve().parent.parent / "config.json"
    if resolved_config_path.exists():
        try:
            config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Nao foi possivel ler {resolved_config_path.name}: {exc}") from exc

        api_key = str(config.get("deepl_api_key", "")).strip()
        if api_key:
            return api_key

    api_key = os.environ.get("DEEPL_API_KEY", "").strip()
    if api_key:
        return api_key

    api_key = getpass.getpass("Chave da API DeepL (entrada oculta): ").strip()
    if not api_key:
        raise RuntimeError(
            "A traducao premium exige uma chave DeepL. Preencha deepl_api_key no config.json, "
            "defina DEEPL_API_KEY ou informe a chave ao iniciar."
        )
    return api_key
