# Guia de manutencao

## Arquitetura

`gerar_srt.py` e apenas o entrypoint e uma fachada de compatibilidade. A aplicacao esta dividida por responsabilidade:

- `srt_generator/cli.py`: argumentos, menus, seletor de arquivos e composicao;
- `srt_generator/runtime.py`: dependencias, deteccao NVIDIA, CUDA e fallback CPU;
- `srt_generator/transcription.py`: adaptador do faster-whisper;
- `srt_generator/translation.py`: cliente e lotes DeepL;
- `srt_generator/subtitles.py`: limite de palavras, timestamps e writer SRT;
- `srt_generator/media.py`: duracao da midia via PyAV;
- `srt_generator/ui.py`: paineis e tabela Rich;
- `srt_generator/pipeline.py`: ordem das quatro fases;
- `srt_generator/config.py`: resolucao da chave DeepL.

```mermaid
flowchart TD
    A[CLI] --> B[Instalar dependencias]
    B --> C[Selecionar CUDA ou CPU]
    C --> D[Carregar large-v3]
    D --> E[Transcrever com VAD]
    E --> F[Gravar SRT original]
    F --> G[Traduzir lotes DeepL]
    G --> H[Gravar SRT PT-BR]
    H --> I{Outro arquivo?}
    I -->|Sim| E
```

## Bootstrap e portabilidade

`ensure_dependencies()` instala `faster-whisper==1.2.1`, `requests` e `rich` com o mesmo interpretador que iniciou o script. O pacote faster-whisper traz CTranslate2 e usa PyAV para decodificar midia, portanto nao depende de PyTorch nem de um FFmpeg instalado no sistema.

`resolve_device()` retorna dispositivo, compute type e nome de exibicao:

- CPU: `cpu` e `int8`;
- NVIDIA: `cuda` e `int8_float16`.

No Windows, `try_install_nvidia_runtime()` tenta instalar cuBLAS CUDA 12 e cuDNN 9 pelos pacotes Python oficiais da NVIDIA e registra suas pastas `bin` no caminho de DLL da execucao. Um driver NVIDIA funcional continua sendo requisito externo. Falha de instalacao ou inicializacao nunca impede a execucao em CPU.

`FasterWhisperTranscriber` ainda protege o carregamento do modelo: se CTranslate2 falhar apenas ao criar o modelo CUDA, recria o modelo em CPU int8. O modelo e baixado para `%LOCALAPPDATA%/generate-srt/models` e reutilizado.

## Transcricao

O modelo padrao e `large-v3`. A chamada usa:

- idioma de origem explicito;
- `beam_size=5`;
- timestamps por palavra;
- Silero VAD;
- `condition_on_previous_text=False`;
- temperaturas `0.0`, `0.2` e `0.4`;
- limites de compressao, probabilidade, silencio e alucinacao.

O faster-whisper devolve um gerador lazy. `transcribe()` deve consumi-lo para que o trabalho aconteca. Cada segmento e normalizado para dicionarios independentes da biblioteca. O progresso e calculado por `segment.end / info.duration`, sem monkey-patch de APIs internas.

## Legendas

`split_segments_by_word_limit()` limita cada entrada a oito palavras. Na transcricao original, usa timestamps reais por palavra. Na traducao, reparte o intervalo proporcionalmente, pois a resposta DeepL nao possui alinhamento de audio.

`write_srt()` e interno e grava UTF-8 com timestamps `HH:MM:SS,mmm`. Assim, o writer nao depende de openai-whisper.

O original e sempre gravado antes da chamada DeepL. Uma falha de traducao preserva o resultado mais caro da transcricao.

## Traducao

`DeepLCloudTranslator` seleciona o endpoint Free para chaves `:fx` e o endpoint Pro para as demais. O destino `pt` vira `PT-BR`. As requisicoes usam contexto, `prefer_less`, `prefer_quality_optimized`, timeout de 30 segundos e autorizacao no cabecalho.

`translate_segments_strict()` limita lotes a 50 itens e cerca de 12.000 caracteres. HTTP 400, 403 e 456 nao sao retentados. Falhas transitórias recebem ate tres tentativas. O retorno deve manter quantidade, ordem e textos nao vazios.

A chave e procurada em `config.json`, depois `DEEPL_API_KEY` e por fim entrada oculta. O `config.json` e versionado por decisao do proprietario; uma chave publicada deve ser revogada no DeepL, pois removê-la depois nao apaga o historico Git.

## Testes

```powershell
python -m unittest -v
python -m py_compile gerar_srt.py srt_generator\*.py test_gerar_srt.py
git diff --check
```

Os testes unitarios nao baixam modelos nem consomem a API. Eles cobrem configuracao do faster-whisper e progresso lazy, divisao temporal, contratos DeepL, erros sem retentativa e prioridade de configuracao.

Uma validacao integrada requer um arquivo curto, internet e chave DeepL. Execute uma vez com `--device cpu` e, quando disponivel, outra com `--device cuda`.

## Alteracoes comuns

Para trocar o backend de transcricao, preserve a interface `transcribe(path, language, on_progress) -> list[dict]` usada pelo pipeline.

Para trocar o tradutor, preserve `translate_many(texts, context) -> list[str]`, mantendo tamanho e ordem. Erros de credencial e cota devem continuar explicitos.

Para mudar o limite de palavras, altere o padrao em `subtitles.py` e atualize os testes e documentos.

## Limitacoes

- Python e internet sao necessarios na primeira instalacao e no download do modelo.
- O script nao instala nem atualiza o driver NVIDIA.
- `large-v3` exige memoria e pode ser lento em CPUs modestas; `--model medium` reduz o custo.
- A API DeepL requer internet, chave valida e cota.
- O seletor grafico depende do Tkinter incluído na instalacao Python.
- O idioma de origem e unico por execucao.

## Convencoes

- Manter Python 3.10 como versao minima enquanto for o ambiente principal.
- Nao importar bibliotecas pesadas no entrypoint antes de `ensure_dependencies()`.
- Nao registrar a chave DeepL nem inclui-la no corpo HTTP.
- Nao versionar midias, SRTs, caches ou ambientes virtuais.
- Fazer commit e push direto para `main` apos cada ajuste, incluindo todas as alteracoes presentes.
