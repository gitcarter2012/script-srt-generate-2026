import importlib
import time
from typing import Callable


class DeepLNonRetryableError(RuntimeError):
    pass


class DeepLCloudTranslator:
    def __init__(self, api_key: str, source_language: str, target_language: str) -> None:
        self.api_key = api_key
        self.source_language = source_language.upper()
        self.target_language = "PT-BR" if target_language.lower() == "pt" else target_language.upper()
        self.api_url = (
            "https://api-free.deepl.com/v2/translate"
            if api_key.endswith(":fx")
            else "https://api.deepl.com/v2/translate"
        )

    def translate(self, text: str) -> str:
        return self.translate_many([text])[0]

    def translate_many(self, texts: list[str], context: str | None = None) -> list[str]:
        requests_module = importlib.import_module("requests")
        request_data: dict[str, object] = {
            "text": texts,
            "source_lang": self.source_language,
            "target_lang": self.target_language,
            "preserve_formatting": "1",
            "formality": "prefer_less",
            "model_type": "prefer_quality_optimized",
        }
        if context:
            request_data["context"] = context
        response = requests_module.post(
            self.api_url,
            headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"},
            data=request_data,
            timeout=30,
        )
        if response.status_code == 400:
            raise DeepLNonRetryableError("Requisicao DeepL invalida. Verifique idiomas e configuracao.")
        if response.status_code == 403:
            raise DeepLNonRetryableError("Chave DeepL invalida ou sem permissao para traducao.")
        if response.status_code == 456:
            raise DeepLNonRetryableError("Cota mensal da conta DeepL esgotada.")
        response.raise_for_status()
        translations = response.json().get("translations", [])
        translated_texts = [str(item.get("text", "")).strip() for item in translations]
        if len(translated_texts) != len(texts) or any(not text for text in translated_texts):
            raise RuntimeError("DeepL nao retornou todas as traducoes esperadas.")
        return translated_texts


def translate_packed_batch(
    texts: list[str],
    create_translator: Callable[[], object],
    min_interval_seconds: float,
    context: str | None = None,
) -> list[str]:
    translator = create_translator()
    last_error = None
    for attempt in range(3):
        if min_interval_seconds > 0:
            time.sleep(min_interval_seconds)
        try:
            return translator.translate_many(texts, context=context)
        except DeepLNonRetryableError:
            raise
        except Exception as exc:  # noqa: BLE001 - Retentativa da API cloud.
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
                translator = create_translator()
    raise RuntimeError("DeepL Cloud falhou apos tres tentativas.") from last_error


def translate_segments_strict(
    segments: list[dict[str, object]],
    create_translator: Callable[[], object],
    min_interval_seconds: float = 0.25,
    max_batch_characters: int = 12000,
    max_batch_items: int = 50,
    on_batch_start: Callable[[int, str], None] | None = None,
    on_batch_complete: Callable[[int, str], None] | None = None,
) -> list[dict[str, object]]:
    translated_segments = [segment.copy() for segment in segments]
    batch_indexes: list[int] = []
    batch_texts: list[str] = []
    batch_characters = 0

    def flush_batch() -> None:
        nonlocal batch_indexes, batch_texts, batch_characters
        if not batch_texts:
            return
        if on_batch_start:
            on_batch_start(len(batch_texts), "DeepL Cloud")
        context_start = max(0, batch_indexes[0] - 3)
        context_end = min(len(segments), batch_indexes[-1] + 4)
        context = "\n".join(
            str(segments[index].get("text", "")).strip()
            for index in range(context_start, context_end)
        )
        translations = translate_packed_batch(
            batch_texts,
            create_translator,
            min_interval_seconds,
            context=context,
        )
        for segment_index, translated_text in zip(batch_indexes, translations):
            translated_segments[segment_index]["text"] = translated_text
        if on_batch_complete:
            on_batch_complete(len(batch_texts), "DeepL Cloud")
        batch_indexes = []
        batch_texts = []
        batch_characters = 0

    for segment_index, segment in enumerate(translated_segments):
        text = str(segment.get("text", "")).strip()
        if not text:
            segment["text"] = ""
            if on_batch_complete:
                on_batch_complete(1, "sem texto")
            continue
        estimated_characters = len(text) + 20
        if batch_texts and (
            batch_characters + estimated_characters > max_batch_characters
            or len(batch_texts) >= max_batch_items
        ):
            flush_batch()
        batch_indexes.append(segment_index)
        batch_texts.append(text)
        batch_characters += estimated_characters

    flush_batch()
    return translated_segments
