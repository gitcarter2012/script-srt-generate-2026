# Gerador de legenda SRT

Transcreve audio e video com faster-whisper ou com o OpenAI Whisper antigo, gera a legenda original e traduz para portugues brasileiro com DeepL Cloud.

## Uso rapido

Requisito: Python 3.10 ou mais recente no `PATH`. O restante e instalado automaticamente na primeira execucao.

```powershell
python .\gerar_srt.py
```

Sem argumentos, uma janela permite selecionar varios arquivos. Tambem e possivel dar duplo clique em `executar_para_selecionar_videos.bat`.

O primeiro menu escolhe o idioma falado no video:

1. ingles;
2. espanhol;
3. japones.

Depois, o menu de execucao apresenta quatro perfis:

1. faster-whisper + GPU automatica: melhor qualidade com `large-v3`, mais rapido e recomendado;
2. faster-whisper + CPU: mesma qualidade, funciona sem NVIDIA e e mais lento;
3. Whisper antigo + GPU NVIDIA: boa qualidade com `medium` e compatibilidade com o fluxo antigo;
4. Whisper antigo + CPU: boa qualidade e maxima compatibilidade, mas muito lento.

Qualquer um dos motores pode processar os tres idiomas. O idioma escolhido e informado explicitamente ao Whisper e ao DeepL; `Enter` usa ingles como padrao.

Para cada `video.mp4`, o script grava na mesma pasta:

- `video_sem_traducao.srt`: transcricao original;
- `video.srt`: traducao premium para portugues brasileiro.

Cada entrada preserva ate 12 palavras em no maximo duas linhas de 40 caracteres. As quebras sao balanceadas, favorecem pontuacao e evitam terminar a primeira linha em artigos ou preposicoes. Frases maiores sao repartidas sem deixar uma cauda isolada de uma ou duas palavras. No faster-whisper, os limites de cada legenda sao recortados pela primeira e pela ultima palavra realmente faladas, e pausas internas acima de 2,5 segundos iniciam uma nova entrada. O terminal mostra o progresso real da transcricao conforme o audio e processado e o progresso de cada lote DeepL.

## Primeira execucao

O script sempre instala automaticamente:

- `requests`, para a API DeepL;
- `rich`, para a interface no terminal.

Depois da escolha, instala apenas o backend necessario:

- faster-whisper: `faster-whisper==1.2.1`, CTranslate2 e PyAV;
- Whisper antigo: `openai-whisper`, PyTorch, `imageio-ffmpeg` e o detector Silero VAD do faster-whisper.

O modelo tambem e baixado automaticamente. O faster-whisper usa `large-v3` por padrao e guarda o cache em `%LOCALAPPDATA%\generate-srt\models`. O Whisper antigo usa `medium` por padrao.

O modelo publico do Hugging Face nao exige `HF_TOKEN`. No Windows sem suporte a symlinks, o cache continua funcionando e pode apenas ocupar mais espaco; esses dois avisos informativos sao ocultados pelo script.

## CPU e NVIDIA

No modo automatico, o script detecta a GPU com `nvidia-smi`:

- NVIDIA disponivel: usa CUDA com `int8_float16`;
- sem NVIDIA: usa CPU com `int8`;
- CUDA indisponivel ou incompatibilidade de DLL/modelo: informa o motivo e continua em CPU.

No Windows com NVIDIA, o script tenta instalar `nvidia-cublas-cu12` e `nvidia-cudnn-cu12` quando necessario. O driver NVIDIA deve estar instalado pelo fabricante; o script nao instala drivers do sistema.

Para o Whisper antigo com GPU, o script instala uma build CUDA compativel do PyTorch e reinicia automaticamente para carregar as novas DLLs. Se CUDA nao ficar disponivel, continua em CPU.

Os dois motores usam Silero VAD para ignorar intervalos sem voz. Os blocos de voz sao limitados a 15 segundos para evitar que trechos mais baixos sejam descartados dentro de janelas longas. No Whisper antigo, somente esses intervalos sao enviados ao modelo, reduzindo alucinacoes em silencio e o tempo de processamento.

No faster-whisper, segmentos de ate duas palavras com confianca media inferior a 20% sao removidos. O limite conservador elimina falsos positivos isolados sem descartar frases longas ou falas curtas reconhecidas com confianca.

Se o decoder interno entregar apenas parte da faixa de audio, o script compara a duracao decodificada com as duracoes do stream e do conteiner. Uma diferenca superior a 10% aciona a recuperacao automatica para WAV temporario pelo FFmpeg; o arquivo temporario e removido ao terminar.

Audios acima de 20 minutos sao decodificados uma vez e transcritos em janelas de quatro minutos com cinco segundos de contexto sobreposto. As bordas de contexto sao descartadas para evitar duplicacoes e alucinacoes. No japones, caracteres isolados no limite entre segmentos sao recompostos com a fala seguinte antes da traducao. A frase espuria recorrente `ご視聴ありがとうございました` tambem e removida antes do envio ao DeepL.

## Chave DeepL

Preencha `config.json`:

```json
{
  "deepl_api_key": "sua-chave"
}
```

Alternativamente, use `$env:DEEPL_API_KEY = "sua-chave"`. Sem ambos, a chave e solicitada com entrada oculta. Contas DeepL API Free e Pro sao detectadas automaticamente.

O projeto envia lotes nativos ao DeepL usando `PT-BR`, as falas anteriores e posteriores como contexto, tom menos formal e modelo otimizado para qualidade. A frase completa e traduzida antes de ser diagramada em ate duas linhas. Credenciais, cota e erros de rede sao reportados sem substituir silenciosamente a traducao.

## Linha de comando

```powershell
python .\gerar_srt.py .\video.mp4
python .\gerar_srt.py .\video1.mp4 .\video2.mp4
python .\gerar_srt.py .\video.mp4 --engine faster --device cpu --source en --source-menu off
python .\gerar_srt.py .\video.mp4 --engine legacy --device cuda
python .\gerar_srt.py .\video.mp4 --engine faster --model medium --output legenda_final
```

Opcoes principais:

- `--engine`: `ask`, `faster` ou `legacy`; `ask` abre o menu completo;
- `--model`: modelo Whisper; padrao `large-v3` no faster ou `medium` no antigo;
- `--device`: `ask`, `auto`, `cuda` ou `cpu`;
- `--source`: idioma do audio; padrao `en`;
- `--target`: idioma de destino; padrao `pt`, enviado como `PT-BR`;
- `--source-menu`: `on` ou `off`;
- `--output`: nome base customizado, somente para uma entrada.

Consulte `python .\gerar_srt.py --help` para a lista completa.

## Manutencao

A arquitetura e os contratos de cada modulo estao em [docs/MAINTENANCE.md](docs/MAINTENANCE.md).

Testes:

```powershell
python -m unittest -v
python -m py_compile gerar_srt.py srt_generator\*.py test_gerar_srt.py
```
